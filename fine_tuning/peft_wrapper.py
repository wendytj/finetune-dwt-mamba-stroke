import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel
from .ProDiaL import apply_prodial_to_model, merge_prodial_weights
from .CoLoRA import apply_colora_to_model, merge_colora_weights, CoLoRAConv2d

# ==============================================================================
# MASTER LIST SELURUH MODUL BERBOBOT DI DWT-MAMBA
# ==============================================================================

# Master List Seluruh Modul Linear/Dense (Tanpa peduli dipakai/di-skip)
LINEAR_TARGET_MODULES = [
    "in_proj",      # Proyeksi input Mamba (row_mamba & col_mamba)
    "out_proj",     # Proyeksi output Mamba (row_mamba & col_mamba)
    "x_proj",       # Proyeksi parameter SSM Mamba (row_mamba & col_mamba)
    "dt_proj",      # Proyeksi delta Mamba
    "gate_fc.0",  # Linear pertama di MB_GSF
    "gate_fc.2",  # Linear kedua di MB_GSF
    "fc.2",       # Linear pertama di SqueezeAndExcitation
    "fc.4",        # Linear kedua di SqueezeAndExcitation  
    "proj_fused",   # Proyeksi fitur DWT-Mamba global sebelum Classifier
    "proj_latent",  # Proyeksi vektor LatentEncoder sebelum Classifier
    "classifier"    # Layer Linear Classifier utama (3 kelas stroke)
]

# Master List Layer Konvolusi (2D & 1D) -> Untuk Differential LR 0.1x pada Optimizer
CONV_TARGET_MODULES = [
    "pe_branches",    # Conv2d Patch Embedding pada 4 subband DWT
    "pm_branches",    # Conv2d Patch Match (Downsampling) pada 4 cabang DWT
    "conv_branch",    # Conv2d Spasial 3x3 pada modul SpaSE_SSM
    "conv1d",         # Conv1d Temporal bawaan internal Mamba SSM (d_conv=4)
    "mb_gsf.proj",    # Conv2d 1x1 Fusi Subband pada MB_GSF (Spesifik agar tidak kena in_proj, out_proj, x_proj)
    "latent_encoder"  # Conv2d Sequential pada LatentEncoder 
]

# Master List Layer Normalisasi
NORM_TARGET_MODULES = [
    "norm",           # LayerNorm pada SpaSE_SSM
    "ln"              # LayerNorm pada MB_GSF
]

# Combined Master List
LIST_TARGET_MODULES = LINEAR_TARGET_MODULES + CONV_TARGET_MODULES + NORM_TARGET_MODULES


def apply_lora(model: nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.05, target_modules: list = LINEAR_TARGET_MODULES) -> nn.Module:
    config = LoraConfig(r=r, lora_alpha=alpha, target_modules=target_modules, lora_dropout=dropout, bias="none", use_dora=False)
    peft_model = get_peft_model(model, config)
    print(f"✅ [LoRA Injected] r={r}, alpha={alpha} pada {target_modules}")
    return peft_model

def apply_dora(model: nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.05, target_modules: list = LINEAR_TARGET_MODULES) -> nn.Module:
    config = LoraConfig(r=r, lora_alpha=alpha, target_modules=target_modules, lora_dropout=dropout, bias="none", use_dora=True)
    peft_model = get_peft_model(model, config)
    print(f"✅ [DoRA Injected] r={r}, alpha={alpha} pada {target_modules}")
    return peft_model

def apply_prodial(model: nn.Module, r_eps: int = 8, r_b: int = 16, target_modules: list = LINEAR_TARGET_MODULES) -> nn.Module:
    peft_model = apply_prodial_to_model(model, target_modules=target_modules, r_b=r_b, r_eps=r_eps)
    print(f"⚡ [ProDiaL Injected] r_b={r_b}, r_eps={r_eps} pada {target_modules}")
    return peft_model

def apply_peft(
    model: nn.Module,
    method: str = "dora",
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    r_b: int = 16,
    target_modules: list = None, # type: ignore
    use_colora_for_conv: bool = True
) -> nn.Module:
    if method is None or str(method).lower() in ["full", "none"]:
        print("🔓 [PEFT Config] Mode Full Fine-Tuning.")
        return model

    method = method.lower()

    # 1. Bekukan seluruh parameter dasar model
    for param in model.parameters():
        param.requires_grad = False

    # 2. Track 1: Terapkan CoLoRA pada Layer Konvolusi
    if use_colora_for_conv:
        print("🎨 [Dual-Track PEFT] Mengaplikasikan CoLoRA pada layer Konvolusi (2D Spasial)...")
        model = apply_colora_to_model(model, target_modules=CONV_TARGET_MODULES)

    # 3. Track 2: Terapkan Adaptor Linear Pilihan
    raw_targets = LINEAR_TARGET_MODULES if target_modules is None else target_modules
    linear_targets = [m for m in raw_targets if m != "dt_proj"]

    if method == "lora":
        model = apply_lora(model, r=r, alpha=alpha, dropout=dropout, target_modules=linear_targets)
    elif method == "dora":
        model = apply_dora(model, r=r, alpha=alpha, dropout=dropout, target_modules=linear_targets)
    elif method == "prodial":
        model = apply_prodial(model, r_eps=r, r_b=r_b, target_modules=linear_targets)

    # 4. 🚨 RE-ENABLE CoLoRA (Mencegah ter-freeze ulang oleh get_peft_model / ProDiaL)
    if use_colora_for_conv:
        for m in model.modules():
            if isinstance(m, CoLoRAConv2d):
                for p in m.depthwise.parameters():
                    p.requires_grad = True
                for p in m.pointwise.parameters():
                    p.requires_grad = True

    return model


def merge_peft_and_unload(model: nn.Module) -> nn.Module:
    merged_any = False
    if any(isinstance(m, CoLoRAConv2d) for m in model.modules()):
        model = merge_colora_weights(model)
        merged_any = True

    if hasattr(model, "is_prodial") or any("ProDiaLLinear" in m.__class__.__name__ for m in model.modules()):
        model = merge_prodial_weights(model)
        merged_any = True

    if isinstance(model, PeftModel):
        model = model.merge_and_unload()
        merged_any = True

    return model