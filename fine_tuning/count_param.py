import os
import sys
import argparse

# Memastikan root project masuk ke sys.path agar impor modul berjalan lancar
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from architectures.DWTMamba import DWTMamba
from fine_tuning.peft_wrapper import apply_peft

# Konfigurasi standar arsitektur DWTMamba (8.2M)
MODEL_CONFIG = {
    "in_channels": 3,
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

TARGET_MODULES = [
    "in_proj",
    "out_proj",
    "proj_fused",
    "proj_latent",
    "classifier"
]

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

def count_parameters(model):
    """Menghitung total parameter dan parameter yang dapat dilatih."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    percentage = (trainable / total) * 100.0 if total > 0 else 0.0
    return trainable, total, percentage

def main():
    parser = argparse.ArgumentParser(description="Hitung Trainable Parameters untuk PEFT (LoRA, DoRA, ProDiaL)")
    
    # Arguments untuk LoRA / DoRA
    parser.add_argument("--lora_r", type=int, default=64, help="Rank untuk LoRA / DoRA (default: 64)")
    parser.add_argument("--lora_a", type=int, default=128, help="Alpha untuk LoRA / DoRA (default: 128)")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="Dropout untuk LoRA / DoRA (default: 0.0)")
    
    # Arguments untuk ProDiaL
    parser.add_argument("--prodial_r_eps", type=int, default=32, help="Rank r_eps untuk off-diagonal ProDiaL (default: 32)")
    parser.add_argument("--prodial_r_b", type=int, default=128, help="Block size r_b untuk diagonal ProDiaL (default: 128)")

    args = parser.parse_args()

    print("=" * 80)
    print(" 📊 DWTMamba PEFT Parameter Counter & Capacity Comparison")
    print("=" * 80)
    print(f"🎯 Target Modules : {TARGET_MODULES}")
    print(f"⚙️ Config LoRA/DoRA: r={args.lora_r}, alpha={args.lora_a}, dropout={args.lora_dropout}")
    print(f"⚙️ Config ProDiaL  : r_eps={args.prodial_r_eps}, r_b={args.prodial_r_b}")
    print("=" * 80)

    methods = [
        ("LoRA", "lora", args.lora_r, args.lora_a, args.prodial_r_b),
        ("DoRA", "dora", args.lora_r, args.lora_a, args.prodial_r_b),
        ("ProDiaL", "prodial", args.prodial_r_eps, args.lora_a, args.prodial_r_b),
    ]

    results = []

    for name, method, r, alpha, r_b in methods:
        # Inisialisasi model bersih
        model = create_base_model()
        
        # Terapkan PEFT wrapper
        peft_model = apply_peft(
            model=model,
            method=method,
            target_modules=TARGET_MODULES,
            r=r,
            alpha=alpha,
            dropout=args.lora_dropout,
            r_b=r_b
        )
        
        trainable, total, pct = count_parameters(peft_model)
        results.append({
            "name": name,
            "trainable": trainable,
            "total": total,
            "percentage": pct
        })

    # Cetak hasil perbandingan
    print(f"{'Metode':<12} | {'Trainable Params':<18} | {'Total Params':<18} | {'Trainable (%)':<15}")
    print("-" * 75)
    for res in results:
        print(f"{res['name']:<12} | {res['trainable']:<18,d} | {res['total']:<18,d} | {res['percentage']:.4f}%")
    print("=" * 80)

    # Menghitung selisih parameter terhadap LoRA
    lora_params = results[0]["trainable"]
    for res in results[1:]:
        diff = res["trainable"] - lora_params
        diff_pct = (diff / lora_params) * 100.0 if lora_params > 0 else 0.0
        sign = "+" if diff >= 0 else ""
        print(f"💡 {res['name']} vs LoRA: {sign}{diff:,d} params ({sign}{diff_pct:.2f}%)")
    print("=" * 80)

if __name__ == "__main__":
    main()