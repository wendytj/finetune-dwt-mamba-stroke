import torch
import torch.nn as nn
import torch.nn.functional as F

import torch

def compute_class_weights(
    train_loader,
    num_classes,
    device,
    use_class_weights=True,
    weight_mode="sqrt",
    beta=0.999,
    delta=0.5,
    per_class_mcc=None,
    class_counts=None,  # 🌟 1. Disamakan nama parameternya menjadi plural
):
    """Menghitung bobot kelas berdasarkan strategi statis maupun dinamis (MCC Feedback)."""

    # 🌟 2. Logika Caching: Ekstrak label HANYA jika class_counts belum ada di memori
    if class_counts is None:
        if hasattr(train_loader.dataset, "labels"):
            train_labels = torch.tensor(train_loader.dataset.labels)
        elif hasattr(train_loader.dataset, "targets"):
            train_labels = torch.tensor(train_loader.dataset.targets)
        else:
            train_labels = torch.tensor(
                [label for _, label in train_loader.dataset]
            )

        train_labels = train_labels.view(-1).long()
        class_counts = torch.bincount(train_labels, minlength=num_classes)

    total_samples = int(class_counts.sum().item())
    N_c = class_counts.float().to(device)

    # Jika use_class_weights == False, kembalikan bobot netral [1.0, 1.0, 1.0]
    if not use_class_weights:
        class_weights = torch.ones(num_classes, device=device)
        return class_weights, class_counts

    # Kalkulasi Berdasarkan weight_mode
    if weight_mode == "sqrt":
        raw_weights = torch.sqrt(total_samples / (num_classes * N_c))

    elif weight_mode == "linear":
        raw_weights = total_samples / (num_classes * N_c)

    elif weight_mode in ["s-b", "d-b-mcc"]:
        # Class-Balanced berbasis Effective Number of Samples (Cui et al., 2019)
        effective_num = (1.0 - torch.pow(beta, N_c)) / (1.0 - beta)
        raw_weights = 1.0 / effective_num
        raw_weights = raw_weights / raw_weights.sum() * num_classes

        # Tambahkan Penyesuaian Dinamis MCC jika mode 'd-b-mcc' dan per_class_mcc tersedia
        if weight_mode == "d-b-mcc" and per_class_mcc is not None:
            # 🌟 3. Sanitasi Safe-Parsing: Ubah None/NaN menjadi 0.0 (netral) agar gradien tidak terkontaminasi NaN
            clean_mcc = [
                0.0 if (m is None or torch.isnan(torch.tensor(m))) else float(m)
                for m in per_class_mcc
            ]
            mcc_tensor = torch.tensor(
                clean_mcc, dtype=torch.float32, device=device
            )

            # Normalisasi error MCC dari [-1, 1] ke rentang [0, 1]
            error_mcc = (1.0 - mcc_tensor) / 2.0
            # Suku adaptif: (1 + delta * error_mcc)
            adaptive_factor = 1.0 + (delta * error_mcc)
            raw_weights = raw_weights * adaptive_factor

    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")

    # Clamp batas aman agar tidak terjadi pergeseran gradien ekstrem
    class_weights = torch.clamp(raw_weights, min=0.1, max=10.0).to(device)

    return class_weights, class_counts

class FocalLoss(nn.Module):
    """Multi-class Focal Loss (Lin et al., 2017)."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor = None, # type: ignore
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer("weight", weight)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        # logits: [B, C], targets: [B]
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Ambil log_prob & prob untuk target kelas asli
        target_log_probs = log_probs.gather(
            dim=-1, index=targets.unsqueeze(1)
        ).squeeze(1)
        target_probs = probs.gather(
            dim=-1, index=targets.unsqueeze(1)
        ).squeeze(1)

        focal_weight = (1.0 - target_probs) ** self.gamma
        loss = -focal_weight * target_log_probs

        if self.weight is not None:
            # 🌟 Proteksi Device Mismatch
            weight_dev = self.weight.to(logits.device)
            class_w = weight_dev[targets] # type: ignore
            loss = loss * class_w

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss (ASL) for Multi-class / Multi-label (Ridnik et al., 2021)."""

    def __init__(
        self,
        gamma_pos: float = 1.0,
        gamma_neg: float = 4.0,
        margin: float = 0.05,
        weight: torch.Tensor = None, # type: ignore
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.margin = margin
        self.reduction = reduction
        self.register_buffer("weight", weight)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1).clamp(min=1e-6, max=1.0 - 1e-6)
        targets_onehot = F.one_hot(targets, num_classes=logits.size(-1)).float()

        # Asymmetric probabilities for negative samples with probability shifting (margin)
        probs_neg = (probs - self.margin).clamp(min=0.0)

        # Positive & Negative Loss Terms
        loss_pos = (
            -targets_onehot
            * ((1.0 - probs) ** self.gamma_pos)
            * torch.log(probs)
        )
        loss_neg = (
            -(1.0 - targets_onehot)
            * (probs_neg**self.gamma_neg)
            * torch.log((1.0 - probs_neg).clamp(min=1e-6))
        )

        loss = loss_pos + loss_neg

        if self.weight is not None:
            # 🌟 Proteksi Device Mismatch
            weight_dev = self.weight.to(logits.device)
            loss = loss * weight_dev.unsqueeze(0) # type: ignore

        loss = loss.sum(dim=-1)  # Sum over classes

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class FECELoss(nn.Module):
    """Focal Balanced Exponential Cross Entropy (F-ECE) Loss (Liu et al., Neurocomputing 2026)."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor = None, # type: ignore
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer("weight", weight)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        targets_onehot = F.one_hot(targets, num_classes=logits.size(-1)).float()

        # 🌟 Positive Term: Exponential CE -> (1 / p) - 1 (Clamped to avoid exploding gradients in early epochs)
        probs_pos = probs.clamp(min=1e-4)
        loss_pos = targets_onehot * ((1.0 / probs_pos) - 1.0)

        # 🌟 Negative Term: Focal Negative -> - p^gamma * log(1 - p)
        probs_neg = probs.clamp(max=1.0 - 1e-6)
        loss_neg = (1.0 - targets_onehot) * (
            -(probs_neg**self.gamma) * torch.log(1.0 - probs_neg)
        )

        loss = loss_pos + loss_neg

        if self.weight is not None:
            # 🌟 Proteksi Device Mismatch
            weight_dev = self.weight.to(logits.device)
            loss = loss * weight_dev.unsqueeze(0) # type: ignore

        loss = loss.sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
    
def build_loss_criterion(config: dict, class_weights: torch.Tensor = None) -> nn.Module: # type: ignore
    """Factory Function untuk membangun kriteria Loss Function berdasarkan CONFIG."""
    loss_type = config.get("loss_type", "ce").lower()
    use_weights = config.get("use_class_weights", True)
    weights = class_weights if use_weights else None

    if loss_type in ["ce", "cross_entropy"]:
        print(f"🎯 [Loss Criterion] Active: CrossEntropyLoss (Weighted={use_weights})")
        return nn.CrossEntropyLoss(weight=weights)

    elif loss_type in ["focal", "focal_loss"]:
        gamma = config.get("focal_gamma", 2.0)
        print(f"🎯 [Loss Criterion] Active: FocalLoss (gamma={gamma}, Weighted={use_weights})")
        return FocalLoss(gamma=gamma, weight=weights) # type: ignore

    elif loss_type in ["asl", "asymmetric"]:
        g_pos = config.get("asl_gamma_pos", 1.0)
        g_neg = config.get("asl_gamma_neg", 4.0)
        margin = config.get("asl_margin", 0.05)
        print(f"🎯 [Loss Criterion] Active: AsymmetricLoss (g_pos={g_pos}, g_neg={g_neg}, margin={margin})")
        return AsymmetricLoss(gamma_pos=g_pos, gamma_neg=g_neg, margin=margin, weight=weights) # type: ignore

    elif loss_type in ["fece", "f_ece", "focal_ece"]:
        gamma = config.get("fece_gamma", 2.0)
        print(f"🎯 [Loss Criterion] Active: F-ECE Loss (gamma={gamma}, Weighted={use_weights})")
        return FECELoss(gamma=gamma, weight=weights) # type: ignore

    else:
        raise ValueError(f"🚨 Invalid loss_type: '{loss_type}'. Pilih opsi: ['ce', 'focal', 'asl', 'fece'].")