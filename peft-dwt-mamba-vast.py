import warnings, os, json, argparse, gc, torch, random
warnings.filterwarnings('ignore')
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch._dynamo
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch._dynamo.config.suppress_errors = True # type: ignore

# 1. Inisialisasi Patch Triton/Mamba Terlebih Dahulu
from architectures.patcher import apply_mamba_patch
apply_mamba_patch()

from loaders.StrokeClassificationLoader import get_stroke_dataloaders
from architectures.DWTMamba import DWTMamba
from fine_tuning.peft_wrapper import apply_peft, CONV_TARGET_MODULES
from loggers.experiment_logger import ExperimentLogger
from tqdm import tqdm

CONFIG = {
    "experiment_code": "dora-801010-1ch-v1",   
    "seed": 42,
    
    "npz_path": "data/turkey_1channel.npz", # JANGAN LUPA PILIH CHANNEL
    "pretrained_path": "weights/pretrained_weights.pth",
    "batch_size": 128,
    "eval_batch_size": 2048,
    "accumulation_steps": 1,
    "num_workers": 32,
    "img_size": 224,
    
    # Arsitektur DWT-Mamba
    "in_channels": 1,
    "embed_dim": 256,
    "depth": 3,
    "mamba_d_state": 16,
    "mamba_d_conv": 4,
    "mamba_expand": 2,
    "se_reduction": 16,
    "mb_gsf_reduction": 4,
    "latent_dim": 128,
    "proj_dim": 256,
    
    # Configuration PEFT
    "peft_method": "dora",       # 'lora', 'dora', 'prodial', atau 'full'
    "target_modules": [          
        "in_proj", 
        "out_proj", 
        "proj_fused", 
        "proj_latent", 
        "classifier", 
    ],
    "lora_r": 32,                 # rank
    "lora_alpha": 64,            # LoRA / DoRA only
    "lora_dropout": 0,        # LoRA / DoRA only

    "prodial_r_eps": 32,  
    "prodial_r_b": 128,
    
    "optimizer": "AdamW",
    "learning_rate": 1e-4,
    "effective_lr": 1e-4,
    "weight_decay": 1e-3,
    "max_epochs": 150,
    "warmup_epochs": 15,
    "early_stop_patience": 25,
    "early_stop_delta": 0.000,
}

def safe_atomic_save(obj, path):
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)

def sanitize_peft_hparams(config: dict) -> dict:
    """Helper function untuk menyaring hyperparameter yang relevan sesuai metode PEFT."""
    clean_cfg = config.copy()
    method = str(clean_cfg.get("peft_method", "full")).lower()

    if method in ["full", "none"]:
        clean_cfg["peft_method"] = "full"
        for key in ["lora_r", "lora_alpha", "lora_dropout", "prodial_r_eps", "prodial_r_b"]:
            clean_cfg.pop(key, None)

    elif method in ["lora", "dora"]:
        clean_cfg.pop("prodial_r_eps", None)
        clean_cfg.pop("prodial_r_b", None)

    elif method == "prodial":
        clean_cfg.pop("lora_r", None)
        clean_cfg.pop("lora_alpha", None)
        clean_cfg.pop("lora_dropout", None)

    return clean_cfg


def run_mock_test(model, train_loader, device, gpu_transform):
    """Pengujian simulasi (Dry-Run) 1 batch untuk memastikan kelancaran VRAM dan shape."""
    print("\n🔍 Memulai Mock Test (Dry-Run 1 Batch)...")
    model.eval()
    try:
        images, labels = next(iter(train_loader))
        images = images.to(device, non_blocking=True)
        # Squeeze dimensi (B, 1) -> (B,) dan konversi ke long
        labels = labels.to(device, non_blocking=True).view(-1).long()

        use_bf16 = torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
            images = gpu_transform(images)
            images = images.to(memory_format=torch.channels_last)
            outputs = model(images)
                    
        assert outputs.shape == (images.size(0), CONFIG["num_classes"]), "Shape output tidak sesuai!"
        print(f"✅ Mock Test Berhasil!")
        print(f"    - Input Shape  : {images.shape}")
        print(f"    - Output Shape : {outputs.shape}")
        print(f"    - Memory VRAM  : {torch.cuda.memory_allocated() / 1e6:.2f} MB\n")
    except Exception as e:
        print(f"❌ Mock Test Gagal: {e}")
        raise e


def train_one_epoch(model, dataloader, criterion, optimizer, device, gpu_transform, amp_dtype, use_bf16, scaler, accumulation_steps=1):
    model.train()
    running_loss = torch.tensor(0.0, device=device)
    correct = torch.tensor(0, device=device)
    total = 0

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(dataloader, desc="Training", leave=False)

    trainable_params = [p for group in optimizer.param_groups for p in group['params']]

    for i, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).view(-1).long()

        with torch.no_grad():
            images = gpu_transform(images)
        
        images = images.to(memory_format=torch.channels_last)

        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_scaled = loss / accumulation_steps

        # 🛡️ JARINGAN PENGAMAN 1: Lewati batch jika Loss bernilai NaN / Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⚠️ [Warning] Loss NaN/Inf terdeteksi pada iterasi {i}! Membatalkan step batch ini...")
            optimizer.zero_grad(set_to_none=True)
            continue

        if use_bf16:
            loss_scaled.backward()
        else:
            scaler.scale(loss_scaled).backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            if use_bf16:
                # 🛡️ JARINGAN PENGAMAN 2: Periksa norm gradien pada BF16
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"\n⚠️ [Warning] Gradien NaN/Inf terdeteksi pada iterasi {i}! Optimizer step dilewati.")
                    optimizer.zero_grad(set_to_none=True)
                else:
                    optimizer.step()
            else:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                
                # GradScaler pada FP16 otomatis melompati step jika gradien Inf/NaN saat scaler.step()
                scaler.step(optimizer)
                scaler.update()

            # 🌟 [PROTEKSI DORA] Pasang Clamping Tepat Setelah Optimizer Step!
            for name, param in model.named_parameters():
                if "lora_magnitude_vector" in name and param.requires_grad:
                    param.data.clamp_(min=1e-6)

            optimizer.zero_grad(set_to_none=True)

        batch_size = images.size(0)
        running_loss += loss.detach() * batch_size
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
        total += batch_size

        if i % 10 == 0:
            pbar.set_postfix({
                "loss": f"{(running_loss / total).item():.4f}",
                "acc": f"{(correct.float() / total).item():.4f}"
            })

    epoch_loss = (running_loss / total).item()
    epoch_acc = (correct.float() / total).item()
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device, gpu_transform, desc="Validation"):
    model.eval()
    running_loss = torch.tensor(0.0, device=device)
    correct = torch.tensor(0, device=device)
    total = 0

    all_targets = []
    all_preds = []
    all_probs = []

    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # Bungkus dataloader dengan tqdm
    pbar = tqdm(dataloader, desc=desc, leave=False)

    with torch.no_grad():
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            # Squeeze dimensi (B, 1) -> (B,) dan konversi ke long
            labels = labels.to(device, non_blocking=True).view(-1).long()

            images = gpu_transform(images)
            images = images.to(memory_format=torch.channels_last) # Pastikan channels_last aktif

            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Softmax dieksekusi dalam presisi FP32 agar akurasi probabilitas stabil
            probs = torch.softmax(outputs.float(), dim=1)
            _, preds = outputs.max(1)

            # Akumulasi tensor murni di GPU tanpa .item()
            batch_size = images.size(0)
            running_loss += loss.detach() * batch_size
            correct += preds.eq(labels).sum()
            total += batch_size

            all_targets.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

            # Update status live loss & acc pada progress bar tqdm
            if i % 5 == 0 or (i + 1) == len(dataloader):
                pbar.set_postfix({
                    "loss": f"{(running_loss / total).item():.4f}",
                    "acc": f"{(correct.float() / total).item():.4f}"
                })

    # Konversi statistik ke CPU 1x saja di akhir
    epoch_loss = (running_loss / total).item()
    epoch_acc = (correct.float() / total).item()
    
    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_probs = torch.cat(all_probs).numpy()

    return epoch_loss, epoch_acc, (y_true, y_pred, y_probs)

def run_training_pipeline(model, raw_model, loaders, transforms, amp_params, logger, device, train_eval_loader):
    """Fungsi modul khusus untuk menangani siklus pelatihan penuh, validation, dan final testing (Dual-Track Model Saver)."""
    train_loader, val_loader, test_loader = loaders
    gpu_train_transform, gpu_eval_transform = transforms
    amp_dtype, use_bf16, scaler = amp_params

    if hasattr(train_loader.dataset, 'labels'):
        train_labels = torch.tensor(train_loader.dataset.labels)
    elif hasattr(train_loader.dataset, 'targets'):
        train_labels = torch.tensor(train_loader.dataset.targets)
    else:
        train_labels = torch.tensor([label for _, label in train_loader.dataset])

    train_labels = train_labels.view(-1).long()
    
    class_counts = torch.bincount(train_labels)
    total_samples = len(train_labels)
    num_classes = CONFIG["num_classes"]
    
    raw_weights = torch.sqrt(total_samples / (num_classes * class_counts.float()))
    class_weights = torch.clamp(raw_weights, min=0.5, max=3.0).to(device)
    
    print(f"⚖️ [Class Imbalance Handled] Distribusi Label Train: {class_counts.tolist()}")
    print(f"⚖️ [Class Weights Dihitung]  : {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

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

    optimizer_grouped_parameters = [
        {
            'params': conv_params,
            'lr': CONFIG["effective_lr"] * 0.1,  # Scaled LR (10% / 0.1x) untuk Layer Konvolusi
            'weight_decay': CONFIG["weight_decay"]
        },
        {
            'params': peft_params,
            'lr': CONFIG["effective_lr"],        # Full LR (100% / 1.0x) untuk Adaptor PEFT & Classifier
            'weight_decay': CONFIG["weight_decay"]
        },
        {
            'params': dora_magnitude_params,
            'lr': CONFIG["effective_lr"],        # Full LR untuk Magnitude DoRA
            'weight_decay': 0.0                  # 🌟 0.0 Weight Decay (Proteksi dari loss=nan)
        }
    ]

    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        eps=1e-6
    )

    print(f"⚙️ [Optimizer Grouping Aktif]")
    print(f"   └─ Conv Layers (0.1x LR) : {len(conv_params)} tensor params | LR: {CONFIG['effective_lr'] * 0.1:.6f}")
    print(f"   └─ PEFT/Head   (1.0x LR) : {len(peft_params)} tensor params | LR: {CONFIG['effective_lr']:.6f}")
    print(f"   └─ DoRA Mag    (WD=0.0)     : {len(dora_magnitude_params)} tensor params | LR: {CONFIG['effective_lr']:.6f}")

    scheduler_warmup = LinearLR(
        optimizer, 
        start_factor=0.1, 
        end_factor=1.0, 
        total_iters=CONFIG["warmup_epochs"]
    )

    scheduler_decay = CosineAnnealingLR(
        optimizer, 
        T_max=CONFIG["max_epochs"] - CONFIG["warmup_epochs"], 
        eta_min= CONFIG["effective_lr"] * 0.01
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_decay],
        milestones=[CONFIG["warmup_epochs"]]
    )

    start_epoch = 1
    best_val_mcc = -1.0
    best_val_loss = float("inf")
    best_mcc_epoch = -1
    best_loss_epoch = -1
    patience_counter = 0

    checkpoint_path = os.path.join(logger.save_dir, "last_checkpoint.pth")
    path_best_val_mcc = os.path.join(logger.save_dir, "best_model_val_mcc.pth")
    path_best_val_loss = os.path.join(logger.save_dir, "best_model_val_loss.pth")

    # Auto-resume logic
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
        
        # ✨ Muat riwayat dari CSV
        logger.load_history_from_csv()
        print(f"✅ Auto-resume sukses! Melanjutkan dari Epoch {start_epoch} | Patience Sisa: [{patience_counter}/{CONFIG['early_stop_patience']}]")

    # Training loop
    try:
        print("🔥 Memulai Pelatihan Aktif (Dual-Track Model Saving: Best Val MCC & Best Val Loss)...")
        for epoch in range(start_epoch, CONFIG["max_epochs"] + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, 
                gpu_train_transform, amp_dtype, use_bf16, scaler, 
                accumulation_steps=CONFIG["accumulation_steps"]
            )
            val_loss, val_acc, val_eval = evaluate(
                model, val_loader, criterion, device, gpu_eval_transform, desc="Validation"
            )

            scheduler.step()
            
            val_metrics, _ = logger.compute_metrics(*val_eval)
            val_f1 = val_metrics["global_metrics"]["f1_score_macro"]
            val_mcc = val_metrics["global_metrics"]["mcc"]
            current_lr = optimizer.param_groups[1]['lr']

            is_mcc_improved = val_mcc > (best_val_mcc + CONFIG["early_stop_delta"])
            is_loss_improved = val_loss < (best_val_loss - CONFIG["early_stop_delta"])

            status_notes = []

            if is_mcc_improved:
                best_val_mcc = val_mcc
                best_mcc_epoch = epoch
                safe_atomic_save(raw_model.state_dict(), path_best_val_mcc)
                status_notes.append(f"🎯 Best Val MCC Saved ({val_mcc:.4f})")

            if is_loss_improved:
                best_val_loss = val_loss
                best_loss_epoch = epoch
                safe_atomic_save(raw_model.state_dict(), path_best_val_loss)
                status_notes.append(f"📉 Best Val Loss Saved ({val_loss:.4f})")

            if is_mcc_improved or is_loss_improved:
                patience_counter = 0
                status_msg = "  --> " + " | ".join(status_notes)
            else:
                patience_counter += 1
                status_msg = f"  --> ⏳ Patience: [{patience_counter}/{CONFIG['early_stop_patience']}]"

            # 1. Log ke memory
            logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_acc=train_acc,
                val_acc=val_acc,
                val_f1=val_f1,
                val_mcc=val_mcc,
                lr=current_lr,
                patience=patience_counter
            )

            # ✨ 2. PERBAIKAN KRUSIAL: Ekspor CSV langsung tiap epoch agar disk selalu ter-update!
            logger.export_csv()

            print(f"Epoch [{epoch:03d}/{CONFIG['max_epochs']}] | LR: {current_lr:.6f} | "
                  f"Train Loss: {train_loss:.4f} - Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} - Acc: {val_acc:.4f} - F1: {val_f1:.4f} - MCC: {val_mcc:.4f}"
                  f"{status_msg}")

            safe_atomic_save(
                {
                    "epoch": epoch,
                    "model_state": raw_model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict() if scaler else None,
                    "best_val_mcc": best_val_mcc,
                    "best_val_loss": best_val_loss,
                    "best_mcc_epoch": best_mcc_epoch, 
                    "best_loss_epoch": best_loss_epoch,
                    "patience_counter": patience_counter,
                },
                checkpoint_path,
            )

            gc.collect()
            torch.cuda.empty_cache()

            if patience_counter >= CONFIG["early_stop_patience"]:
                print(f"\n🛑 Early stopping dipicu pada epoch {epoch} (Val MCC & Val Loss stagnan selama {patience_counter} epoch).")
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
            
    # Final Evaluation & Testing Kedua Model (Safe PEFT Merge)
    print("\n🧪 Mengevaluasi Performance pada Test Set untuk Kedua Model Terbaik...")

    def eval_and_merge_checkpoint(ckpt_path, desc_tag):
        if not os.path.exists(ckpt_path):
            return None, None, None
        print(f"  --> Evaluasi Model '{desc_tag}'...")
        
        eval_m = DWTMamba(
            in_channels=CONFIG["in_channels"],
            num_classes=CONFIG["num_classes"],
            embed_dim=CONFIG["embed_dim"],
            depth=CONFIG["depth"],
            mamba_d_state=CONFIG["mamba_d_state"],
            mamba_d_conv=CONFIG["mamba_d_conv"],
            mamba_expand=CONFIG["mamba_expand"],
            se_reduction=CONFIG["se_reduction"],
            mb_gsf_reduction=CONFIG["mb_gsf_reduction"],
            latent_dim=CONFIG["latent_dim"],
            proj_dim=CONFIG["proj_dim"]
        ).to(device).to(memory_format=torch.channels_last) # type: ignore

        active_r = CONFIG["prodial_r_eps"] if CONFIG["peft_method"] == "prodial" else CONFIG["lora_r"]

        eval_m = apply_peft(
            eval_m,
            method=CONFIG["peft_method"],
            target_modules= CONFIG.get("target_modules", None), # type: ignore
            r=active_r,
            alpha=CONFIG["lora_alpha"],
            dropout=CONFIG["lora_dropout"],
            r_b=CONFIG["prodial_r_b"]
        )

        eval_m = eval_m.to(device)

        eval_m.load_state_dict(torch.load(ckpt_path, map_location=device))
        eval_m = eval_m.to(device).to(memory_format=torch.channels_last) # type: ignore

        _, _, tr_eval = evaluate(eval_m, train_eval_loader, criterion, device, gpu_eval_transform, desc=f"Eval Train ({desc_tag})")
        _, _, va_eval = evaluate(eval_m, val_loader, criterion, device, gpu_eval_transform, desc=f"Eval Val ({desc_tag})")
        
        te_eval = None
        if test_loader is not None:
            _, _, te_eval = evaluate(eval_m, test_loader, criterion, device, gpu_eval_transform, desc=f"Testing ({desc_tag})")
        
        del eval_m
        gc.collect()
        torch.cuda.empty_cache()

        return tr_eval, va_eval, te_eval

    same_best_epoch = (best_mcc_epoch == best_loss_epoch)
    
    tr_mcc, va_mcc, te_mcc = eval_and_merge_checkpoint(path_best_val_mcc, f"Best MCC (Ep {best_mcc_epoch})")
    
    if same_best_epoch:
        print("💡 Epoch terbaik MCC dan Loss sama! Menggunakan hasil evaluasi yang sama untuk menghemat waktu.")
        tr_loss, va_loss, te_loss = tr_mcc, va_mcc, te_mcc
    else:
        tr_loss, va_loss, te_loss = eval_and_merge_checkpoint(path_best_val_loss, f"Best Loss (Ep {best_loss_epoch})")

    # ✨ PERBAIKAN: Pastikan CSV final diekspor dan kurva digambar di alur normal!
    logger.export_csv()

    if te_mcc is not None and te_mcc[0] is not None:
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

    final_json_data = {
        "dataset_info": {
            "num_classes": CONFIG["num_classes"],
            "class_labels": logger.class_names if hasattr(logger, "class_names") else ["Normal", "Ischemia", "Bleeding"],
            "train_class_counts": class_counts.tolist(),
            "class_weights": class_weights.cpu().tolist()
        },
        "hyperparameters": logger.hparams if hasattr(logger, "hparams") else CONFIG,
        
        "model_by_val_mcc": {
            "criterion": "best_val_mcc",
            "best_epoch": best_mcc_epoch,
            "best_metric_value": best_val_mcc,
            "train_results": wrap_eval_results(tr_mcc),
            "val_results": wrap_eval_results(va_mcc),
            "test_results": wrap_eval_results(te_mcc)
        },
        
        "model_by_val_loss": {
            "criterion": "best_val_loss",
            "best_epoch": best_loss_epoch,
            "best_metric_value": best_val_loss,
            "train_results": wrap_eval_results(tr_loss),
            "val_results": wrap_eval_results(va_loss),
            "test_results": wrap_eval_results(te_loss)
        }
    }

    summary_json_path = os.path.join(logger.save_dir, "best_model_metrics.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(final_json_data, f, indent=4)

    if te_mcc is not None and te_mcc[0] is not None:
        _, cm_mcc = logger.compute_metrics(*te_mcc)
        logger.plot_confusion_matrix(cm_mcc, filename="test_confusion_matrix_by_mcc.png")
        
    if te_loss is not None and te_loss[0] is not None:
        _, cm_loss = logger.compute_metrics(*te_loss)
        logger.plot_confusion_matrix(cm_loss, filename="test_confusion_matrix_by_loss.png")

    print(f"✨ Eksperimen Selesai! Seluruh metrik (Train, Val, Test) untuk kedua model berhasil disimpan di: {summary_json_path}")

def main():
    args = parse_args()

    CONFIG["batch_size"] = args.batch_size
    CONFIG["accumulation_steps"] = args.accumulation_steps
    CONFIG["num_workers"] = args.num_workers
    CONFIG["max_epochs"] = args.max_epochs
    CONFIG["learning_rate"] = args.learning_rate
    CONFIG["weight_decay"] = args.weight_decay
    CONFIG["experiment_code"] = args.experiment_code
    CONFIG["warmup_epochs"] = args.warmup_epochs
    CONFIG["eval_batch_size"] = args.eval_batch_size

    # Sinkronisasi parameter PEFT & Dataset Path dari CLI
    CONFIG["npz_path"] = args.npz_path
    CONFIG["pretrained_path"] = args.pretrained_path
    CONFIG["peft_method"] = args.peft_method
    CONFIG["lora_r"] = args.lora_r
    CONFIG["lora_alpha"] = args.lora_alpha
    CONFIG["lora_dropout"] = args.lora_dropout
    CONFIG["prodial_r_eps"] = args.prodial_r_eps
    CONFIG["prodial_r_b"] = args.prodial_r_b
    CONFIG["in_channels"] = args.in_channels

    if args.target_modules:
        CONFIG["target_modules"] = args.target_modules

    # Auto-detect 3-channel jika nama file mengandung '3channel'
    if "3channel" in CONFIG["npz_path"] and CONFIG["in_channels"] == 1:
        CONFIG["in_channels"] = 3
        print("💡 Auto-detected 3-channel dataset dari npz_path! Mengubah in_channels=3.")

    CONFIG["effective_lr"] = CONFIG["learning_rate"] * (CONFIG["batch_size"] / 16) ** 0.5

    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    random.seed(CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Memulai Eksekusi pada Perangkat: {device}")
    print(f"💡 Base LR: {CONFIG['learning_rate']} | Effective LR (Batch Size {CONFIG['batch_size']}): {CONFIG['effective_lr']:.6f}")

    # Setup AMP & Scaler
    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16) # type: ignore

    # 1. Load Data
    # 1. Panggil loader dengan menyertakan num_channels dan config split
    loader_output = get_stroke_dataloaders(
        npz_path=CONFIG["npz_path"],
        config=CONFIG.get("split_config", "split_80_10_10"),
        num_channels=CONFIG["in_channels"], # <--- Sinkron dengan 1-ch atau 3-ch
        batch_size=CONFIG["batch_size"],
        eval_batch_size=CONFIG["eval_batch_size"],
        num_workers=CONFIG["num_workers"],
    )

    # 2. Unpacking dinamis (Aman untuk mode Holdout maupun K-Fold)
    if len(loader_output) == 6:
        train_loader, val_loader, test_loader, num_classes, gpu_train_transform, gpu_eval_transform = loader_output
    else:
        train_loader, val_loader, num_classes, gpu_train_transform, gpu_eval_transform = loader_output
        test_loader = None
    
    train_eval_loader = DataLoader(
        train_loader.dataset,
        batch_size=CONFIG["eval_batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True
    )
    
    CONFIG["num_classes"] = num_classes
    class_names = ["Normal", "Ischemia", "Bleeding"]

    # 1.1 Inisialisasi Model DWT-Mamba Base
    model = DWTMamba(
        in_channels=CONFIG["in_channels"],
        num_classes=CONFIG["num_classes"], # type: ignore
        embed_dim=CONFIG["embed_dim"],
        depth=CONFIG["depth"],
        mamba_d_state=CONFIG["mamba_d_state"],
        mamba_d_conv=CONFIG["mamba_d_conv"],
        mamba_expand=CONFIG["mamba_expand"],
        se_reduction=CONFIG["se_reduction"],
        mb_gsf_reduction=CONFIG["mb_gsf_reduction"],
        latent_dim=CONFIG["latent_dim"],
        proj_dim=CONFIG["proj_dim"]
    ).to(device).to(memory_format=torch.channels_last) # type: ignore

    # 1.2 Memuat Pretrained Weights OrganCMNIST (Otomatis Weight Inflation/Deflation jika channel berbeda)
    model.load_pretrained_weights(CONFIG["pretrained_path"], device=device)

    active_r = CONFIG["prodial_r_eps"] if CONFIG["peft_method"] == "prodial" else CONFIG["lora_r"]

    model = apply_peft(
        model,
        method=CONFIG["peft_method"],
        r=active_r,
        alpha=CONFIG["lora_alpha"],
        dropout=CONFIG["lora_dropout"],
        r_b=CONFIG["prodial_r_b"],
        target_modules=CONFIG.get("target_modules", None), # type: ignore
    )
    model = model.to(device).to(memory_format=torch.channels_last) # type: ignore

    # 🔍 [DIAGNOSTIC HOOK] Melacak path lengkap layer pemicu NaN
    # def make_nan_hook(mod_name):
    #     def debug_nan_hook(module, input, output):
    #         if isinstance(output, torch.Tensor):
    #             if torch.isnan(output).any() or torch.isinf(output).any():
    #                 print(f"🚨 [NaN Detected Output] Module: {mod_name} ({module.__class__.__name__})")
    #         elif isinstance(output, (tuple, list)):
    #             for i, out in enumerate(output):
    #                 if isinstance(out, torch.Tensor) and (torch.isnan(out).any() or torch.isinf(out).any()):
    #                     print(f"🚨 [NaN Detected Output Tuple {i}] Module: {mod_name} ({module.__class__.__name__})")
    #     return debug_nan_hook

    # for name, module in model.named_modules():
    #     module.register_forward_hook(make_nan_hook(name))

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)

    raw_model = model

    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"📊 [PEFT Param Summary] Trainable: {trainable_p:,} / {total_p:,} ({100 * trainable_p / total_p:.2f}%)")

    # 2. Sanitasi Hparams & Inisialisasi Logger
    clean_hparams = sanitize_peft_hparams(CONFIG)
    logger = ExperimentLogger(
        save_dir=f"./logs/{CONFIG['experiment_code']}",
        class_names=class_names,
        hparams=clean_hparams
    )

    # 4. Mode Percabangan: Dry-Run vs Full Training
    run_mock_test(model, train_loader, device, gpu_eval_transform)

    if args.dry_run:
        print("💡 Mode --dry_run selesai. Program keluar tanpa melakukan pelatihan.")
        return

    gc.collect()
    torch.cuda.empty_cache()

    # 5. Jalankan Training Pipeline Penuh
    loaders = (train_loader, val_loader, test_loader)
    transforms = (gpu_train_transform, gpu_eval_transform)
    amp_params = (amp_dtype, use_bf16, scaler)

    run_training_pipeline(model, raw_model, loaders, transforms, amp_params, logger, device, train_eval_loader)

def parse_args():
    parser = argparse.ArgumentParser(description="DWT-Mamba Modular Fine-Tuning Pipeline with PEFT")
    parser.add_argument("--batch_size", type=int, default=CONFIG["batch_size"], help="Batch size per iterasi GPU")
    parser.add_argument("--accumulation_steps", type=int, default=CONFIG["accumulation_steps"], help="Langkah akumulasi gradien")
    parser.add_argument("--num_workers", type=int, default=CONFIG["num_workers"], help="Jumlah worker DataLoader")
    parser.add_argument("--max_epochs", type=int, default=CONFIG["max_epochs"], help="Jumlah maksimum epoch pelatihan")
    parser.add_argument("--warmup_epochs", type=int, default=CONFIG["warmup_epochs"], help="Jumlah epoch LR warmup")
    parser.add_argument("--learning_rate", type=float, default=CONFIG["learning_rate"], help="Learning rate AdamW")
    parser.add_argument("--weight_decay", type=float, default=CONFIG["weight_decay"], help="Weight decay AdamW")
    parser.add_argument("--experiment_code", type=str, default=CONFIG["experiment_code"], help="Kode/Folder eksperimen")
    parser.add_argument("--dry_run", action="store_true", help="Eksekusi mock test 1 batch lalu keluar tanpa training")
    parser.add_argument("--eval_batch_size", type=int, default=CONFIG["eval_batch_size"], help="Batch size evaluasi")
    
    # Argumen Tambahan Khusus Fine-Tuning & PEFT
    parser.add_argument("--npz_path", type=str, default=CONFIG["npz_path"], help="Path ke dataset .npz target")
    parser.add_argument("--in_channels", type=int, default=CONFIG["in_channels"], choices=[1, 3], help="Jumlah channel input (1 atau 3)")
    parser.add_argument("--pretrained_path", type=str, default=CONFIG["pretrained_path"], help="Path checkpoint pre-trained")
    parser.add_argument("--peft_method", type=str, default=CONFIG["peft_method"], choices=["lora", "dora", "prodial", "full"], help="Metode PEFT")
    parser.add_argument("--lora_r", type=int, default=CONFIG["lora_r"], help="Rank LoRA / DoRA")
    parser.add_argument("--lora_alpha", type=int, default=CONFIG["lora_alpha"], help="Alpha LoRA / DoRA")
    parser.add_argument("--lora_dropout", type=float, default=CONFIG["lora_dropout"], help="Dropout LoRA / DoRA")
    parser.add_argument("--prodial_r_eps", type=int, default=CONFIG["prodial_r_eps"], help="Rank r_eps ProDiaL")
    parser.add_argument("--prodial_r_b", type=int, default=CONFIG["prodial_r_b"], help="Pembagi blok diagonal ProDiaL")
    parser.add_argument("--target_modules", nargs="+", default=CONFIG["target_modules"], help="Daftar nama modul target PEFT")
    return parser.parse_args()

if __name__ == "__main__":
    main()