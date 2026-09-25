import os
import torch
from .configs import CONFIG

def safe_atomic_save(obj, path):
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)

def sanitize_peft_hparams(config: dict) -> dict:
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