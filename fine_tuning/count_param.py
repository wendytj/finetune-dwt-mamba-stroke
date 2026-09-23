import os
import sys
import argparse

# Memastikan root project masuk ke sys.path agar impor modul berjalan lancar
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from architectures.DWTMamba import DWTMamba
from fine_tuning.peft_wrapper import apply_peft, CONV_TARGET_MODULES, LINEAR_TARGET_MODULES

# Konfigurasi standar arsitektur DWTMamba (8.14M Base)
MODEL_CONFIG = {
    "in_channels": 1,
    "num_classes": 3,
    "embed_dim": 256,
    "depth": 3,
    "mamba_d_state": 16,
    "mamba_d_conv": 4,
    "mamba_expand": 2,
    "se_reduction": 16,
    "mb_gsf_reduction": 4,
    "latent_dim": 128,
    "proj_dim": 256,
}

def create_base_model():
    """Membuat instansiasi model baru DWTMamba."""
    return DWTMamba(
        in_channels=MODEL_CONFIG["in_channels"],
        num_classes=MODEL_CONFIG["num_classes"],
        embed_dim=MODEL_CONFIG["embed_dim"],
        depth=MODEL_CONFIG["depth"],
        mamba_d_state=MODEL_CONFIG["mamba_d_state"],
        mamba_d_conv=MODEL_CONFIG["mamba_d_conv"],
        mamba_expand=MODEL_CONFIG["mamba_expand"],
        se_reduction=MODEL_CONFIG["se_reduction"],
        mb_gsf_reduction=MODEL_CONFIG["mb_gsf_reduction"],
        latent_dim=MODEL_CONFIG["latent_dim"],
        proj_dim=MODEL_CONFIG["proj_dim"]
    )

def count_dual_track_parameters(model, conv_keywords):
    """Menghitung rincian parameter terlatih antara Linear PEFT dan Conv CoLoRA."""
    linear_peft_trainable = 0
    conv_colora_trainable = 0
    total_params = 0

    for name, param in model.named_parameters():
        numel = param.numel()
        total_params += numel
        if param.requires_grad:
            if any(k in name for k in conv_keywords):
                conv_colora_trainable += numel
            else:
                linear_peft_trainable += numel

    total_trainable = linear_peft_trainable + conv_colora_trainable
    pct = (total_trainable / total_params) * 100.0 if total_params > 0 else 0.0
    return linear_peft_trainable, conv_colora_trainable, total_trainable, total_params, pct

def main():
    parser = argparse.ArgumentParser(description="Hitung Rincian Trainable Parameters Dual-Track PEFT (LoRA, DoRA, ProDiaL + CoLoRA)")
    
    # Arguments untuk LoRA / DoRA
    parser.add_argument("--lora_r", type=int, default=64, help="Rank untuk LoRA / DoRA (default: 64)")
    parser.add_argument("--lora_a", type=int, default=128, help="Alpha untuk LoRA / DoRA (default: 128)")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="Dropout untuk LoRA / DoRA (default: 0.0)")
    
    # Arguments untuk ProDiaL
    parser.add_argument("--prodial_r_eps", type=int, default=64, help="Rank r_eps untuk off-diagonal ProDiaL (default: 64)")
    parser.add_argument("--prodial_r_b", type=int, default=128, help="Block size r_b untuk diagonal ProDiaL (default: 128)")

    args = parser.parse_args()

    print("=" * 100)
    print(" 📊 DWTMamba Dual-Track PEFT Parameter Counter & Breakdown")
    print("=" * 100)
    print(f"🎯 Target Linear PEFT : {LINEAR_TARGET_MODULES}")
    print(f"🎨 Target Conv CoLoRA  : {CONV_TARGET_MODULES}")
    print(f"⚙️ Config LoRA/DoRA    : r={args.lora_r}, alpha={args.lora_a}, dropout={args.lora_dropout}")
    print(f"⚙️ Config ProDiaL      : r_eps={args.prodial_r_eps}, r_b={args.prodial_r_b}")
    print("=" * 100)

    methods = [
        ("LoRA + CoLoRA", "lora", args.lora_r, args.lora_a, args.prodial_r_b),
        ("DoRA + CoLoRA", "dora", args.lora_r, args.lora_a, args.prodial_r_b),
        ("ProDiaL + CoLoRA", "prodial", args.prodial_r_eps, args.lora_a, args.prodial_r_b),
    ]

    results = []

    for name, method, r, alpha, r_b in methods:
        # 1. Inisialisasi model bersih
        model = create_base_model()
        
        # 2. Terapkan Dual-Track PEFT wrapper (CoLoRA + Linear PEFT secara otomatis)
        peft_model = apply_peft(
            model=model,
            method=method,
            target_modules=LINEAR_TARGET_MODULES,
            r=r,
            alpha=alpha,
            dropout=args.lora_dropout,
            r_b=r_b,
            use_colora_for_conv=True
        )

        # 3. Hitung rincian parameter (Tanpa unfreezing manual!)
        lin_p, conv_p, total_tr, total_p, pct = count_dual_track_parameters(peft_model, CONV_TARGET_MODULES)
        results.append({
            "name": name,
            "linear_trainable": lin_p,
            "conv_trainable": conv_p,
            "total_trainable": total_tr,
            "total_params": total_p,
            "percentage": pct
        })

    # Cetak hasil perbandingan rincian parameter
    print("\n" + "=" * 100)
    header = f"{'Metode PEFT':<18} | {'Linear PEFT Params':<18} | {'Conv CoLoRA Params':<18} | {'Total Trainable':<16} | {'Total Model':<14} | {'Trainable (%)':<12}"
    print(header)
    print("-" * len(header))
    for res in results:
        print(f"{res['name']:<18} | {res['linear_trainable']:<18,d} | {res['conv_trainable']:<18,d} | {res['total_trainable']:<16,d} | {res['total_params']:<14,d} | {res['percentage']:.4f}%")
    print("=" * 100)

    # Menghitung selisih parameter Linear PEFT murni terhadap LoRA
    lora_lin_params = results[0]["linear_trainable"]
    print("💡 ANALISIS KAPASITAS ADAPTOR LINIER MURNI (Eksklusi Layer CoLoRA):")
    for res in results[1:]:
        diff = res["linear_trainable"] - lora_lin_params
        diff_pct = (diff / lora_lin_params) * 100.0 if lora_lin_params > 0 else 0.0
        sign = "+" if diff >= 0 else ""
        print(f"    └─ {res['name']} vs LoRA: {sign}{diff:,d} Linear params ({sign}{diff_pct:.2f}%)")
    print("=" * 100)

if __name__ == "__main__":
    main()

# python fine_tuning/count_param.py --lora_r 64 --lora_a 128 --lora_dropout 0.0 --prodial_r_eps 64 --prodial_r_b 128