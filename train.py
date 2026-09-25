import warnings, gc, torch, random
import numpy as np
warnings.filterwarnings('ignore')

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch._dynamo.config.suppress_errors = True  # type: ignore

# 1. Triton / Mamba Patch
from architectures.patcher import apply_mamba_patch
apply_mamba_patch()

from torch.utils.data import DataLoader
from loaders.StrokeClassificationLoader import get_stroke_dataloaders
from architectures.DWTMamba import DWTMamba
from fine_tuning.peft_wrapper import apply_peft
from loggers.experiment_logger import ExperimentLogger

# Import modul internal kita
from train_modules.configs import CONFIG, parse_args
from train_modules.utils import sanitize_peft_hparams, run_mock_test
from train_modules.pipeline import run_training_pipeline

def main():
    args = parse_args()

    # Sync argumen CLI ke CONFIG
    CONFIG["batch_size"] = args.batch_size
    CONFIG["accumulation_steps"] = args.accumulation_steps
    CONFIG["num_workers"] = args.num_workers
    CONFIG["max_epochs"] = args.max_epochs
    CONFIG["learning_rate"] = args.learning_rate
    CONFIG["weight_decay"] = args.weight_decay
    CONFIG["experiment_code"] = args.experiment_code
    CONFIG["warmup_epochs"] = args.warmup_epochs
    CONFIG["eval_batch_size"] = args.eval_batch_size

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

    if "3channel" in CONFIG["npz_path"] and CONFIG["in_channels"] == 1:
        CONFIG["in_channels"] = 3
        print("💡 Auto-detected 3-channel dataset dari npz_path! Mengubah in_channels=3.")

    CONFIG["effective_lr"] = CONFIG["learning_rate"] * (CONFIG["batch_size"] / 16) ** 0.5

    # Seed reproducibility
    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    random.seed(CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Memulai Eksekusi pada Perangkat: {device}")
    print(f"💡 Base LR: {CONFIG['learning_rate']} | Effective LR (Batch Size {CONFIG['batch_size']}): {CONFIG['effective_lr']:.6f}")

    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16) # type: ignore

    # Load Data
    loader_output = get_stroke_dataloaders(
        npz_path=CONFIG["npz_path"],
        config=CONFIG.get("split_config", "split_80_10_10"),
        num_channels=CONFIG["in_channels"],
        batch_size=CONFIG["batch_size"],
        eval_batch_size=CONFIG["eval_batch_size"],
        num_workers=CONFIG["num_workers"],
    )

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

    # Inisialisasi Model DWT-Mamba
    model = DWTMamba(
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

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    raw_model = model

    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"📊 [PEFT Param Summary] Trainable: {trainable_p:,} / {total_p:,} ({100 * trainable_p / total_p:.2f}%)")

    clean_hparams = sanitize_peft_hparams(CONFIG)
    logger = ExperimentLogger(
        save_dir=f"./logs/{CONFIG['experiment_code']}",
        class_names=class_names,
        hparams=clean_hparams
    )

    run_mock_test(model, train_loader, device, gpu_eval_transform)

    if args.dry_run:
        print("💡 Mode --dry_run selesai. Program keluar tanpa melakukan pelatihan.")
        return

    gc.collect()
    torch.cuda.empty_cache()

    loaders = (train_loader, val_loader, test_loader)
    transforms = (gpu_train_transform, gpu_eval_transform)
    amp_params = (amp_dtype, use_bf16, scaler)

    # Jalankan Siklus Pelatihan Penuh
    run_training_pipeline(model, raw_model, loaders, transforms, amp_params, logger, device, train_eval_loader)

if __name__ == "__main__":
    main()