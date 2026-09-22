import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel
from .ProDiaL import apply_prodial_to_model, merge_prodial_weights

LIST_TARGET_MODULES = [
    "in_proj", 
    "out_proj", 
    "x_proj", 
    "dt_proj", 
    "proj_fused", 
    "proj_latent", 
    "classifier", 
    "gate_fc", 
    "fc"
]

def apply_lora(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: list = LIST_TARGET_MODULES
) -> nn.Module:
    """Injeksi LoRA standar via Hugging Face PEFT."""
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        use_dora=False
    )
    peft_model = get_peft_model(model, config)
    print(f"✅ [LoRA Injected] r={r}, alpha={alpha} pada {target_modules}")
    peft_model.print_trainable_parameters()
    return peft_model


def apply_dora(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: list = LIST_TARGET_MODULES
) -> nn.Module:
    """Injeksi DoRA (Weight-Decomposed LoRA) via Hugging Face PEFT."""
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        use_dora=True
    )
    peft_model = get_peft_model(model, config)
    print(f"✅ [DoRA Injected] r={r}, alpha={alpha} pada {target_modules}")
    peft_model.print_trainable_parameters()
    return peft_model


def apply_prodial(
    model: nn.Module,
    r_eps: int = 8,
    r_b: int = 16,
    target_modules: list = LIST_TARGET_MODULES
) -> nn.Module:
    """Injeksi ProDiaL (Diagonal-centric Transformation) kustom KAIST 2025."""
    peft_model = apply_prodial_to_model(
        model,
        target_modules=target_modules,
        r_b=r_b,
        r_eps=r_eps
    )
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"⚡ [ProDiaL Injected] r_b={r_b}, r_eps={r_eps} pada {target_modules}")
    print(f"📊 Trainable Parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    return peft_model


# ==========================================
# 2. UNIFIED FACTORY ROUTER (Pintu Utama)
# ==========================================

def apply_peft(
    model: nn.Module,
    method: str = "dora",
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    r_b: int = 16,
    target_modules: list = None # type: ignore
) -> nn.Module:
    """Router utama untuk memilih metode PEFT berdasarkan string konfigurasi."""
    if method is None or str(method).lower() in ["full", "none"]:
        print("🔓 [PEFT Config] Mode Full Fine-Tuning (seluruh parameter dilatih).")
        return model

    method = method.lower()
    targets = target_modules or LIST_TARGET_MODULES

    if method == "lora":
        return apply_lora(model, r=r, alpha=alpha, dropout=dropout, target_modules=targets)
    elif method == "dora":
        return apply_dora(model, r=r, alpha=alpha, dropout=dropout, target_modules=targets)
    elif method == "prodial":
        return apply_prodial(model, r_eps=r, r_b=r_b, target_modules=targets)
    else:
        raise ValueError(f"❌ Metode PEFT '{method}' tidak dikenal! Pilih: 'lora', 'dora', 'prodial', atau 'full'.")


def merge_peft_and_unload(model: nn.Module) -> nn.Module:
    """Menggabungkan (merge) weights PEFT ke backbone utama untuk zero-latency inference."""
    if isinstance(model, PeftModel):
        model = model.merge_and_unload()
        print("✅ [PEFT Merged] LoRA/DoRA weights berhasil digabungkan ke backbone.")
        return model
    elif hasattr(model, "is_prodial") or any("ProDiaLLinear" in m.__class__.__name__ for m in model.modules()):
        model = merge_prodial_weights(model)
        print("✅ [PEFT Merged] ProDiaL weights berhasil digabungkan ke backbone.")
        return model
    else:
        # Mode Full Fine-Tuning atau tanpa PEFT (tidak perlu di-merge)
        print("🔓 [Full FT] Tidak ada adaptor PEFT yang perlu digabungkan.")
        return model