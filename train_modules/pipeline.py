import os, json, gc, torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from train_modules.configs import CONFIG
from train_modules.utils import safe_atomic_save
from train_modules.trainer import train_one_epoch, evaluate
from train_modules.loss import compute_class_weights, build_loss_criterion
from architectures.DWTMamba import DWTMamba
from fine_tuning.peft_wrapper import apply_peft, CONV_TARGET_MODULES

def build_optimizer_and_scheduler(model, effective_lr, weight_decay, warmup_epochs, max_epochs, lr_conv_ratio=0.1):
    conv_params = []
    dora_magnitude_params = []
    peft_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if "lora_magnitude_vector" in name:
                dora_magnitude_params.append(param)
            elif any(k in name for k in CONV_TARGET_MODULES):
                conv_params.append(param)
            else:
                peft_params.append(param)

    conv_lr = effective_lr * lr_conv_ratio

    optimizer_grouped_parameters = [
        {'params': conv_params, 'lr': conv_lr, 'weight_decay': weight_decay},
        {'params': peft_params, 'lr': effective_lr, 'weight_decay': weight_decay},
        {'params': dora_magnitude_params, 'lr': effective_lr, 'weight_decay': 0.0} # Proteksi DoRA WD=0
    ]

    optimizer = optim.AdamW(optimizer_grouped_parameters, eps=1e-6)

    print(f"⚙️ [Optimizer Grouping Aktif]")
    print(f"   └─ Conv Layers (0.1x LR) : {len(conv_params)} tensor params | LR: {effective_lr * 0.1:.6f}")
    print(f"   └─ PEFT/Head   (1.0x LR) : {len(peft_params)} tensor params | LR: {effective_lr:.6f}")
    print(f"   └─ DoRA Mag    (WD=0.0)  : {len(dora_magnitude_params)} tensor params | LR: {effective_lr:.6f}")

    scheduler_warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    scheduler_decay = CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs, eta_min=effective_lr * 0.01)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_epochs])

    return optimizer, scheduler, dora_magnitude_params

def evaluate_best_checkpoints(
    logger, device, criterion, class_counts, class_weights,
    best_mcc_epoch, best_loss_epoch, best_val_mcc, best_val_loss,
    path_best_val_mcc, path_best_val_loss,
    train_eval_loader, val_loader, test_loader, gpu_eval_transform
):
    print("\n🧪 Mengevaluasi Performance pada Test Set untuk Kedua Model Terbaik...")

    def eval_single_checkpoint(ckpt_path, desc_tag):
        if not os.path.exists(ckpt_path):
            print(f"⚠️ Checkpoint tidak ditemukan: {ckpt_path}")
            return None, None, None
            
        print(f"  --> Evaluasi Model '{desc_tag}'...")
        
        # 1. Alokasi ulang memori GPU & bersihkan cache sebelum inisialisasi
        gc.collect()
        torch.cuda.empty_cache()

        # 2. Inisialisasi Model TANPA channels_last (mencegah crash Triton/Mamba kernel)
        eval_m = DWTMamba(
            in_channels=CONFIG["in_channels"], num_classes=CONFIG["num_classes"],
            embed_dim=CONFIG["embed_dim"], depth=CONFIG["depth"],
            mamba_d_state=CONFIG["mamba_d_state"], mamba_d_conv=CONFIG["mamba_d_conv"],
            mamba_expand=CONFIG["mamba_expand"], se_reduction=CONFIG["se_reduction"],
            mb_gsf_reduction=CONFIG["mb_gsf_reduction"], latent_dim=CONFIG["latent_dim"],
            proj_dim=CONFIG["proj_dim"]
        ).to(device)

        active_r = CONFIG["prodial_r_eps"] if CONFIG["peft_method"] == "prodial" else CONFIG["lora_r"]

        # 3. Suntikkan PEFT
        eval_m = apply_peft(
            eval_m, method=CONFIG["peft_method"],
            target_modules=CONFIG.get("target_modules", None),
            r=active_r, alpha=CONFIG["lora_alpha"],
            dropout=CONFIG["lora_dropout"], r_b=CONFIG["prodial_r_b"]
        ).to(device)

        # 4. Load weights & Kunci ke Mode Evaluasi Eksplisit (.eval())
        state_dict = torch.load(ckpt_path, map_location=device)
        eval_m.load_state_dict(state_dict)
        eval_m.eval() # WAJIB: Mematikan Dropout/BatchNorm & Graph Autograd

        # 5. Jalankan Evaluasi dalam blok torch.no_grad()
        with torch.no_grad():
            _, _, tr_eval = evaluate(eval_m, train_eval_loader, criterion, device, gpu_eval_transform, desc=f"Eval Train ({desc_tag})")
            _, _, va_eval = evaluate(eval_m, val_loader, criterion, device, gpu_eval_transform, desc=f"Eval Val ({desc_tag})")
            
            te_eval = None
            if test_loader is not None:
                _, _, te_eval = evaluate(eval_m, test_loader, criterion, device, gpu_eval_transform, desc=f"Testing ({desc_tag})")
        
        # 6. Bersihkan model dari VRAM secara bersih
        del eval_m, state_dict
        gc.collect()
        torch.cuda.empty_cache()

        return tr_eval, va_eval, te_eval

    same_best_epoch = (best_mcc_epoch == best_loss_epoch)
    
    # Evaluasi Model 1 (Best MCC)
    tr_mcc, va_mcc, te_mcc = eval_single_checkpoint(path_best_val_mcc, f"Best MCC (Ep {best_mcc_epoch})")
    
    # Evaluasi Model 2 (Best Loss)
    if same_best_epoch:
        print("💡 Epoch terbaik MCC dan Loss sama! Menggunakan hasil evaluasi yang sama.")
        tr_loss, va_loss, te_loss = tr_mcc, va_mcc, te_mcc
    else:
        tr_loss, va_loss, te_loss = eval_single_checkpoint(path_best_val_loss, f"Best Loss (Ep {best_loss_epoch})")

    logger.export_csv()
    if te_mcc and te_mcc[0] is not None:
        logger.plot_learning_curves(test_eval=te_mcc, filename="training_dashboard.png")

    def wrap_eval_results(eval_tuple):
        if eval_tuple is None or eval_tuple[0] is None:
            return None
        metrics_dict, cm = logger.compute_metrics(*eval_tuple)
        return {
            "confusion_matrix": cm.tolist() if hasattr(cm, "tolist") else cm,
            "global_metrics": metrics_dict["global_metrics"],
            "per_class_metrics": metrics_dict["per_class_metrics"]
        }

    summary_json_path = os.path.join(logger.save_dir, "best_model_metrics.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset_info": {
                "num_classes": CONFIG["num_classes"],
                "class_labels": logger.class_names if hasattr(logger, "class_names") else ["Normal", "Ischemia", "Bleeding"],
                "train_class_counts": class_counts.tolist(),
                "class_weights": class_weights.cpu().tolist()
            },
            "hyperparameters": logger.hparams if hasattr(logger, "hparams") else CONFIG,
            "model_by_val_mcc": {
                "criterion": "best_val_mcc", "best_epoch": best_mcc_epoch, "best_metric_value": best_val_mcc,
                "train_results": wrap_eval_results(tr_mcc), "val_results": wrap_eval_results(va_mcc), "test_results": wrap_eval_results(te_mcc)
            },
            "model_by_val_loss": {
                "criterion": "best_val_loss", "best_epoch": best_loss_epoch, "best_metric_value": best_val_loss,
                "train_results": wrap_eval_results(tr_loss), "val_results": wrap_eval_results(va_loss), "test_results": wrap_eval_results(te_loss)
            }
        }, f, indent=4)

    if te_mcc and te_mcc[0] is not None:
        _, cm_mcc = logger.compute_metrics(*te_mcc)
        logger.plot_confusion_matrix(cm_mcc, filename="test_confusion_matrix_by_mcc.png")
    if te_loss and te_loss[0] is not None:
        _, cm_loss = logger.compute_metrics(*te_loss)
        logger.plot_confusion_matrix(cm_loss, filename="test_confusion_matrix_by_loss.png")

    print(f"✨ Eksperimen Selesai! Seluruh metrik disimpan di: {summary_json_path}")


def run_training_pipeline(model, raw_model, loaders, transforms, amp_params, logger, device, train_eval_loader):
    train_loader, val_loader, test_loader = loaders
    gpu_train_transform, gpu_eval_transform = transforms
    amp_dtype, use_bf16, scaler = amp_params

    class_weights, class_counts = compute_class_weights(train_loader, CONFIG["num_classes"], device)
    criterion = build_loss_criterion(CONFIG, class_weights=class_weights)

    # 2. Setup Optimizer & Scheduler
    optimizer, scheduler, dora_magnitude_params = build_optimizer_and_scheduler(
        model, CONFIG["effective_lr"], CONFIG["weight_decay"], CONFIG["warmup_epochs"], CONFIG["max_epochs"], CONFIG["lr_conv_ratio"]
    )

    # 3. Setup Checkpoint Paths & State Variables
    start_epoch, best_val_mcc, best_val_loss = 1, -1.0, float("inf")
    best_mcc_epoch, best_loss_epoch, patience_counter = -1, -1, 0

    checkpoint_path = os.path.join(logger.save_dir, "last_checkpoint.pth")
    path_best_val_mcc = os.path.join(logger.save_dir, "best_model_val_mcc.pth")
    path_best_val_loss = os.path.join(logger.save_dir, "best_model_val_loss.pth")

    # 4. Auto-Resume Logic
    if os.path.exists(checkpoint_path):
        print(f"🔄 Checkpoint terputus ditemukan! Memuat state dari: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        raw_model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        if scaler and checkpoint.get("scaler_state"):
            scaler.load_state_dict(checkpoint["scaler_state"])
        
        start_epoch = checkpoint["epoch"] + 1
        best_val_mcc = checkpoint.get("best_val_mcc", -1.0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        best_mcc_epoch = checkpoint.get("best_mcc_epoch", -1)
        best_loss_epoch = checkpoint.get("best_loss_epoch", -1)
        patience_counter = checkpoint.get("patience_counter", 0)
        logger.load_history_from_csv()
        print(f"✅ Auto-resume sukses! Melanjutkan dari Epoch {start_epoch}")

    # 5. Training Loop Utama
    try:
        print("🔥 Memulai Pelatihan Aktif...")
        for epoch in range(start_epoch, CONFIG["max_epochs"] + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, 
                gpu_train_transform, amp_dtype, use_bf16, scaler, 
                dora_magnitude_params, accumulation_steps=CONFIG["accumulation_steps"]
            )
            val_loss, val_acc, val_eval = evaluate(model, val_loader, criterion, device, gpu_eval_transform, desc="Validation")

            scheduler.step()
            val_metrics, _ = logger.compute_metrics(*val_eval)
            val_f1, val_mcc = val_metrics["global_metrics"]["f1_score_macro"], val_metrics["global_metrics"]["mcc"]
            current_lr = optimizer.param_groups[1]['lr']

            is_mcc_improved = val_mcc > (best_val_mcc + CONFIG["early_stop_delta"])
            is_loss_improved = val_loss < (best_val_loss - CONFIG["early_stop_delta"])

            status_notes = []
            if is_mcc_improved:
                best_val_mcc, best_mcc_epoch = val_mcc, epoch
                safe_atomic_save(raw_model.state_dict(), path_best_val_mcc)
                status_notes.append(f"🎯 Best Val MCC Saved ({val_mcc:.4f})")

            if is_loss_improved:
                best_val_loss, best_loss_epoch = val_loss, epoch
                safe_atomic_save(raw_model.state_dict(), path_best_val_loss)
                status_notes.append(f"📉 Best Val Loss Saved ({val_loss:.4f})")

            if is_mcc_improved or is_loss_improved:
                patience_counter = 0
                status_msg = "  --> " + " | ".join(status_notes)
            else:
                patience_counter += 1
                status_msg = f"  --> ⏳ Patience: [{patience_counter}/{CONFIG['early_stop_patience']}]"

            logger.log_epoch(epoch, train_loss, val_loss, train_acc, val_acc, val_f1, val_mcc, current_lr, patience_counter)
            logger.export_csv()

            print(f"Epoch [{epoch:03d}/{CONFIG['max_epochs']}] | LR: {current_lr:.6f} | "
                  f"Train Loss: {train_loss:.4f} - Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} - Acc: {val_acc:.4f} - F1: {val_f1:.4f} - MCC: {val_mcc:.4f}{status_msg}")

            safe_atomic_save({
                "epoch": epoch, "model_state": raw_model.state_dict(),
                "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if scaler else None,
                "best_val_mcc": best_val_mcc, "best_val_loss": best_val_loss,
                "best_mcc_epoch": best_mcc_epoch, "best_loss_epoch": best_loss_epoch,
                "patience_counter": patience_counter,
            }, checkpoint_path)

            if patience_counter >= CONFIG["early_stop_patience"]:
                print(f"\n🛑 Early stopping dipicu pada epoch {epoch}.")
                break

    except KeyboardInterrupt:
        print("\n⚠️ Pelatihan dihentikan secara manual (Ctrl + C)!")
        logger.log_error("Pelatihan dihentikan paksa oleh user (KeyboardInterrupt).", context="USER_INTERRUPT")
        logger.export_csv()
        logger.plot_learning_curves()
        return

    except Exception as e:
        print(f"\n🚨 Pelatihan mengalami crash/error: {e}")
        logger.log_error(e, context="TRAINING_CRASH")
        logger.export_csv()
        logger.plot_learning_curves()
        raise e
    
    gc.collect()
    torch.cuda.empty_cache()

    # 6. Eksekusi Evaluasi Akhir
    evaluate_best_checkpoints(
        logger, device, criterion, class_counts, class_weights,
        best_mcc_epoch, best_loss_epoch, best_val_mcc, best_val_loss,
        path_best_val_mcc, path_best_val_loss,
        train_eval_loader, val_loader, test_loader, gpu_eval_transform
    )