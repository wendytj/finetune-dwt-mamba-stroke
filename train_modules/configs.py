import argparse

CONFIG = {
    "experiment_code": "dora-801010-3ch-v1",   
    "seed": 42,
    
    "num_classes": 3,
    "npz_path": "data/turkey_3channel.npz",
    "pretrained_path": "weights/pretrained_weights.pth",
    "batch_size": 64,
    "eval_batch_size": 1024,
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
    
    # Konfigurasi PEFT
    "peft_method": "dora",       # 'lora', 'dora', 'prodial', atau 'full'
    "target_modules": [          
        "in_proj", 
        "out_proj", 
        "proj_fused", 
        "proj_latent", 
        "classifier", 
    ],
    "lora_r": 32,
    "lora_alpha": 64,   
    "lora_dropout": 0.0,

    "prodial_r_eps": 32,  
    "prodial_r_b": 128,

    # Konfigurasi Loss Function (Dinamis dari loss.py)
    "loss_type": "ce",           # Opsi: 'ce', 'focal', 'asl', 'fece'
    "use_class_weights": True,   # Multiplexer pembobotan kelas statis
    "focal_gamma": 2.0,          # Hyperparameter Focal Loss (Lin et al., 2017)
    "asl_gamma_pos": 1.0,        # Hyperparameter ASL Positif (Ridnik et al., 2021)
    "asl_gamma_neg": 4.0,        # Hyperparameter ASL Negatif
    "asl_margin": 0.05,          # Probability Margin Shift ASL
    "fece_gamma": 2.0,           # Hyperparameter F-ECE Loss Negatif (Liu et al., 2026)
    
    # Konfigurasi Optimizer & Pelatihan
    "optimizer": "AdamW",
    "learning_rate": 1e-4,
    "effective_lr": 1e-4,
    "lr_conv_ratio": 0.1,        # 🌟 Rasio LR untuk Layer Konvolusi CoLoRA (default: 0.1x Effective LR)
    "weight_decay": 1e-3,
    "max_epochs": 150,
    "warmup_epochs": 15,
    "early_stop_patience": 25,
    "early_stop_delta": 0.000,
}

def parse_args():
    parser = argparse.ArgumentParser(description="DWT-Mamba Modular Fine-Tuning Pipeline with PEFT")
    parser.add_argument("--batch_size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--accumulation_steps", type=int, default=CONFIG["accumulation_steps"])
    parser.add_argument("--num_workers", type=int, default=CONFIG["num_workers"])
    parser.add_argument("--max_epochs", type=int, default=CONFIG["max_epochs"])
    parser.add_argument("--warmup_epochs", type=int, default=CONFIG["warmup_epochs"])
    parser.add_argument("--learning_rate", type=float, default=CONFIG["learning_rate"])
    parser.add_argument("--lr_conv_ratio", type=float, default=CONFIG["lr_conv_ratio"], help="Rasio Learning Rate layer Conv CoLoRA terhadap Effective LR")
    parser.add_argument("--weight_decay", type=float, default=CONFIG["weight_decay"])
    parser.add_argument("--experiment_code", type=str, default=CONFIG["experiment_code"])
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--eval_batch_size", type=int, default=CONFIG["eval_batch_size"])
    
    parser.add_argument("--npz_path", type=str, default=CONFIG["npz_path"])
    parser.add_argument("--in_channels", type=int, default=CONFIG["in_channels"], choices=[1, 3])
    parser.add_argument("--pretrained_path", type=str, default=CONFIG["pretrained_path"])

    parser.add_argument("--peft_method", type=str, default=CONFIG["peft_method"], choices=["lora", "dora", "prodial", "full"])
    parser.add_argument("--lora_r", type=int, default=CONFIG["lora_r"])
    parser.add_argument("--lora_alpha", type=int, default=CONFIG["lora_alpha"])
    parser.add_argument("--lora_dropout", type=float, default=CONFIG["lora_dropout"])
    parser.add_argument("--prodial_r_eps", type=int, default=CONFIG["prodial_r_eps"])
    parser.add_argument("--prodial_r_b", type=int, default=CONFIG["prodial_r_b"])
    parser.add_argument("--target_modules", nargs="+", default=CONFIG["target_modules"])

    parser.add_argument("--loss_type", type=str, default=CONFIG["loss_type"], choices=["ce", "focal", "asl", "fece"], help="Pilihan fungsi loss")
    parser.add_argument("--use_class_weights", action=argparse.BooleanOptionalAction, default=CONFIG["use_class_weights"], help="Gunakan pembobotan kelas statis")
    parser.add_argument("--focal_gamma", type=float, default=CONFIG["focal_gamma"], help="Gamma parameter untuk Focal Loss")
    parser.add_argument("--asl_gamma_pos", type=float, default=CONFIG["asl_gamma_pos"], help="Gamma positif untuk Asymmetric Loss")
    parser.add_argument("--asl_gamma_neg", type=float, default=CONFIG["asl_gamma_neg"], help="Gamma negatif untuk Asymmetric Loss")
    parser.add_argument("--asl_margin", type=float, default=CONFIG["asl_margin"], help="Probability margin shift untuk ASL")
    parser.add_argument("--fece_gamma", type=float, default=CONFIG["fece_gamma"], help="Gamma parameter untuk F-ECE Loss")
    return parser.parse_args()