import os
import torch
from .configs import CONFIG

def safe_atomic_save(obj, path):
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)

def sanitize_peft_hparams(config: dict) -> dict:
    """Membersihkan parameter PEFT yang tidak relevan agar JSON log tetap rapi."""
    clean_cfg = config.copy()
    method = str(clean_cfg.get("peft_method", "full")).lower()

    if method in ["full", "none"]:
        clean_cfg["peft_method"] = "full"
        # 🌟 Hapus seluruh parameter spesifik PEFT (termasuk target_modules)
        for key in [
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "prodial_r_eps",
            "prodial_r_b",
            "target_modules",
            "lr_conv_ratio",
        ]:
            clean_cfg.pop(key, None)

    elif method in ["lora", "dora"]:
        # Hapus parameter spesifik ProDiAL
        clean_cfg.pop("prodial_r_eps", None)
        clean_cfg.pop("prodial_r_b", None)

    elif method == "prodial":
        # Hapus parameter spesifik LoRA/DoRA (Tetap simpan lora_dropout jika dipakai oleh ProDiAL)
        clean_cfg.pop("lora_r", None)
        clean_cfg.pop("lora_alpha", None)

    return clean_cfg

def sanitize_loss_hparams(config: dict) -> dict:
    clean_cfg = config.copy()
    loss_type = str(clean_cfg.get("loss_type", "ce")).lower()

    # Daftar seluruh hyperparameter spesifik loss
    all_loss_keys = [
        "focal_gamma",
        "asl_gamma_pos",
        "asl_gamma_neg",
        "asl_margin",
        "fece_gamma",
    ]

    if loss_type in ["ce", "cross_entropy"]:
        # CE tidak membutuhkan hyperparameter tambahan
        for key in all_loss_keys:
            clean_cfg.pop(key, None)

    elif loss_type in ["focal", "focal_loss"]:
        # Hanya simpan focal_gamma
        for key in ["asl_gamma_pos", "asl_gamma_neg", "asl_margin", "fece_gamma"]:
            clean_cfg.pop(key, None)

    elif loss_type in ["asl", "asymmetric"]:
        # Hanya simpan asl_*
        for key in ["focal_gamma", "fece_gamma"]:
            clean_cfg.pop(key, None)

    elif loss_type in ["fece", "f_ece", "focal_ece"]:
        # Hanya simpan fece_gamma
        for key in ["focal_gamma", "asl_gamma_pos", "asl_gamma_neg", "asl_margin"]:
            clean_cfg.pop(key, None)

    return clean_cfg

def sanitize_class_weight_hparams(config: dict) -> dict:
    """Membersihkan parameter class weight yang tidak relevan agar JSON log tetap rapi."""
    clean_cfg = config.copy()
    if not clean_cfg.get("use_class_weights", False):
        for key in ["weight_mode", "weight_beta", "weight_delta"]:
            clean_cfg.pop(key, None)
    else:
        mode = clean_cfg.get("weight_mode", "sqrt")
        if mode not in ["s-b", "d-b-mcc"]:
            clean_cfg.pop("weight_beta", None)
        if mode != "d-b-mcc":
            clean_cfg.pop("weight_delta", None)
    return clean_cfg

def sanitize_all_hparams(config: dict) -> dict:
    """Master function untuk membersihkan hparams PEFT, Loss, dan Class Weights secara sekaligus."""
    clean_cfg = sanitize_peft_hparams(config)
    clean_cfg = sanitize_loss_hparams(clean_cfg)
    clean_cfg = sanitize_class_weight_hparams(clean_cfg)
    return clean_cfg

def run_mock_test(model, train_loader, device, gpu_transform):
    print("\n🔍 Memulai Mock Test (Dry-Run 1 Batch)...")
    model.eval()
    try:
        images, labels = next(iter(train_loader))
        images = images.to(device, non_blocking=True)
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