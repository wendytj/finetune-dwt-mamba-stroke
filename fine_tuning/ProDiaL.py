import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProDiaLLinear(nn.Module):
    """
    ProDiaL (Projector-targeted Diagonal-centric Linear Transformation) Wrapper Layer.
    Paper: "Parameter Efficient Mamba Tuning via Projector-targeted Diagonal-centric 
           Linear Transformation" (KAIST, 2025).
    
    Formula: W' = s * W * D_b + B_eps * A_eps
    """
    def __init__(self, original_linear: nn.Linear, r_b: int = 16, r_eps: int = 8):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.r_b = r_b
        self.r_eps = r_eps

        # 1. Bekukan (Freeze) bobot layer linear asli
        self.original_linear = original_linear
        for param in self.original_linear.parameters():
            param.requires_grad = False

        # Validasi pembagian blok
        if self.in_features % r_b != 0:
            raise ValueError(
                f"❌ in_features ({self.in_features}) harus habis dibagi r_b ({r_b})!"
            )
        self.block_size = self.in_features // r_b

        # 2. Block-Diagonal Matrix D_b (r_b blok berukuran block_size x block_size)
        # Diinisialisasi sebagai Matriks Identitas
        self.D_blocks = nn.Parameter(
            torch.eye(self.block_size).unsqueeze(0).repeat(r_b, 1, 1)
        )

        # 3. Scaling Factor s (vektor sepanjang out_features)
        # Diinisialisasi dengan angka 1
        self.s = nn.Parameter(torch.ones(self.out_features))

        # 4. Off-Diagonal LoRA Matrices (A_eps & B_eps)
        self.A_eps = nn.Parameter(torch.empty(r_eps, self.in_features))
        self.B_eps = nn.Parameter(torch.zeros(self.out_features, r_eps))

        # Inisialisasi A_eps dengan Kaiming Uniform / Gaussian noise
        nn.init.kaiming_uniform_(self.A_eps, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        
        # A. Reshape input x untuk operasi perkalian Block-Diagonal D_b
        # Shape: [Batch * SeqLen, r_b, block_size]
        x_reshaped = x.view(-1, self.r_b, self.block_size)

        # x @ D_b^T via Tensor Einsum secara efisien
        x_trans = torch.einsum('b r k, r m k -> b r m', x_reshaped, self.D_blocks)
        x_trans = x_trans.view(orig_shape)

        # B. Forward melalui bobot W asli (frozen)
        base_out = F.linear(x_trans, self.original_linear.weight, bias=None)

        # C. Element-wise Scaling per-channel menggunakan s
        scaled_out = base_out * self.s

        # D. Adaptasi Off-Diagonal LoRA: (x @ A_eps^T) @ B_eps^T
        lora_out = F.linear(F.linear(x, self.A_eps), self.B_eps)

        # E. Penggabungan akhir + bias asli jika ada
        out = scaled_out + lora_out
        if self.original_linear.bias is not None:
            out = out + self.original_linear.bias

        return out

    def merge_weights(self) -> None:
        """
        Menggabungkan W' = s * (W @ D_b) + B_eps @ A_eps langsung ke layer linear asli
        agar saat inference tidak ada overhead komputasi tambahan.
        """
        W = self.original_linear.weight.data  # [out_features, in_features]

        # Construct matriks penuh D_b dari blok-blok kecil
        D_b = torch.block_diag(*[self.D_blocks[i] for i in range(self.r_b)])

        # W @ D_b
        W_Db = torch.matmul(W, D_b)

        # Scaling s pada setiap baris
        scaled_W = self.s.unsqueeze(1) * W_Db

        # Off-diagonal LoRA matrix
        eps = torch.matmul(self.B_eps, self.A_eps)

        # Timpa bobot asli dan kembalikan ke mode trainable biasa
        self.original_linear.weight.data = scaled_W + eps
        self.original_linear.weight.requires_grad = True


def apply_prodial_to_model(
    model: nn.Module,
    target_modules: list = None, # type: ignore
    r_b: int = 16,
    r_eps: int = 8
) -> nn.Module:
    """
    Injektor otomatis ProDiaL ke layer-layer linear target dalam model.
    """
    # 1. ✨ BEKUKAN SELURUH BASE MODEL DULU SEBELUM INJEKSI!
    for param in model.parameters():
        param.requires_grad = False

    if target_modules is None:
        # Target default ProDiaL: Projectors & Classifier Heads
        target_modules = ["in_proj", "out_proj", "proj_fused", "proj_latent", "classifier"]

    # 2. Fungsi rekursif pembantu agar pembekuan di atas tidak terpanggil berulang kali
    def _inject_recursive(module_to_check):
        for name, child in module_to_check.named_children():
            if isinstance(child, nn.Linear) and any(target in name for target in target_modules):
                try:
                    setattr(module_to_check, name, ProDiaLLinear(child, r_b=r_b, r_eps=r_eps))
                except ValueError as e:
                    print(f"  ⚠️ [ProDiaL Skipped] {name}: {e}")
            else:
                _inject_recursive(child)

    _inject_recursive(model)
    return model


def merge_prodial_weights(model: nn.Module) -> nn.Module:
    """
    Menggabungkan seluruh layer ProDiaL kembali ke modul linear standar
    sebelum menyimpan checkpoint final.
    """
    for name, module in model.named_children():
        if isinstance(module, ProDiaLLinear):
            module.merge_weights()
            setattr(model, name, module.original_linear)
            print(f"  ✅ [ProDiaL Merged] -> {name}")
        else:
            merge_prodial_weights(module)
    return model