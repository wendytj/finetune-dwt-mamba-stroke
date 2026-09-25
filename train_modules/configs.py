import argparse

CONFIG = {
    "experiment_code": "dora-801010-3ch-v1",   
    "seed": 42,
    
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
    "peft_method": "lora",       # 'lora', 'dora', 'prodial', atau 'full'
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
    
    "optimizer": "AdamW",
    "learning_rate": 1e-4,
    "effective_lr": 1e-4,
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
    return parser.parse_args()