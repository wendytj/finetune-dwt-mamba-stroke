import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_class_weights(train_loader, num_classes, device):
    """Menghitung bobot kelas berbasis inversi frekuensi ter-clamp."""
    if hasattr(train_loader.dataset, 'labels'):
        train_labels = torch.tensor(train_loader.dataset.labels)
    elif hasattr(train_loader.dataset, 'targets'):
        train_labels = torch.tensor(train_loader.dataset.targets)
    else:
        train_labels = torch.tensor([label for _, label in train_loader.dataset])

    train_labels = train_labels.view(-1).long()
    class_counts = torch.bincount(train_labels, minlength=num_classes)
    total_samples = len(train_labels)
    
    raw_weights = torch.sqrt(total_samples / (num_classes * class_counts.float()))
    class_weights = torch.clamp(raw_weights, min=0.5, max=3.0).to(device)
    
    print(f"⚖️ [Class Imbalance] Distribusi Label Train: {class_counts.tolist()}")
    print(f"⚖️ [Class Weights]   : {class_weights.tolist()}")
    return class_weights, class_counts


class FocalLoss(nn.Module):
    """Multi-class Focal Loss (Lin et al., 2017)."""
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None, reduction: str = 'mean'): # type: ignore
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, C], targets: [B]
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        
        # Ambil log_prob & prob untuk target kelas asli
        target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        
        focal_weight = (1.0 - target_probs) ** self.gamma
        loss = -focal_weight * target_log_probs

        if self.weight is not None:
            class_w = self.weight[targets]
            loss = loss * class_w

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
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
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.margin = margin
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1).clamp(min=1e-6, max=1.0 - 1e-6)
        targets_onehot = F.one_hot(targets, num_classes=logits.size(-1)).float()

        # Asymmetric probabilities for negative samples with probability shifting (margin)
        probs_neg = (probs - self.margin).clamp(min=0.0)

        # Positive & Negative Loss Terms
        loss_pos = -targets_onehot * ((1.0 - probs) ** self.gamma_pos) * torch.log(probs)
        loss_neg = -(1.0 - targets_onehot) * (probs_neg ** self.gamma_neg) * torch.log((1.0 - probs_neg).clamp(min=1e-6))

        loss = loss_pos + loss_neg

        if self.weight is not None:
            loss = loss * self.weight.unsqueeze(0)

        loss = loss.sum(dim=-1) # Sum over classes

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class FECELoss(nn.Module):
    """Focal Balanced Exponential Cross Entropy (F-ECE) Loss (Liu et al., Neurocomputing 2026)."""
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None, reduction: str = 'mean'): # type: ignore
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        targets_onehot = F.one_hot(targets, num_classes=logits.size(-1)).float()

        # 🌟 Positive Term: Exponential CE -> (1 / p) - 1 (Clamped to avoid exploding gradients in early epochs)
        probs_pos = probs.clamp(min=1e-4)
        loss_pos = targets_onehot * ((1.0 / probs_pos) - 1.0)

        # 🌟 Negative Term: Focal Negative -> - p^gamma * log(1 - p)
        probs_neg = probs.clamp(max=1.0 - 1e-6)
        loss_neg = (1.0 - targets_onehot) * (-(probs_neg ** self.gamma) * torch.log(1.0 - probs_neg))

        loss = loss_pos + loss_neg

        if self.weight is not None:
            loss = loss * self.weight.unsqueeze(0)

        loss = loss.sum(dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
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