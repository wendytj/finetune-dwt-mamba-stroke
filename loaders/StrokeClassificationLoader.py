import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as v2
from .debugger import unpack_split_info

class StrokeNPZDataset(Dataset):
    """Dataset class untuk memuat array NPZ CT Scan ke PyTorch Tensor."""

    def __init__(self, images_array, labels_array):
        # Konversi ke PyTorch Tensor
        images_tensor = torch.from_numpy(images_array)

        # Standardisasi Dimensi ke [N, C, H, W] saja (tanpa duplikasi/slicing channel di RAM)
        if images_tensor.ndim == 3:  # [N, H, W] -> [N, 1, H, W]
            images_tensor = images_tensor.unsqueeze(1)
        elif images_tensor.ndim == 4 and images_tensor.shape[-1] in [1, 3]:  # [N, H, W, C] -> [N, C, H, W]
            images_tensor = images_tensor.permute(0, 3, 1, 2)

        self.images = images_tensor
        self.labels = torch.from_numpy(labels_array).long().reshape(-1)

    def __len__(self): 
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def get_stroke_dataloaders(
    npz_path="data/turkey_1channel.npz",
    config="split_80_10_10",
    fold_idx=0,
    num_channels=1,
    batch_size=16,
    eval_batch_size=None,
    num_workers=2,
):
    """
    Fungsi pembuat DataLoader untuk Holdout dan K-Fold Cross Validation.
    
    Returns:
        - Mode Holdout : (train_loader, val_loader, test_loader, num_classes, gpu_train_transform, gpu_eval_transform)
        - Mode K-Fold  : (train_loader, val_loader, num_classes, gpu_train_transform, gpu_eval_transform)
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"❌ File NPZ tidak ditemukan di path: {npz_path}")

    if eval_batch_size is None:
        eval_batch_size = batch_size * 4

    # 1. Load Data NPZ & Ekstrak Indeks Split dari debugger.py
    data = np.load(npz_path, allow_pickle=True)
    images = data["images"]
    labels = data["labels"]

    if config not in data:
        raise KeyError(f"❌ Key config '{config}' tidak ditemukan dalam file NPZ.")

    split_info = unpack_split_info(data[config], fold_idx=fold_idx)
    train_idx = split_info["train"]  # type: ignore
    val_idx = split_info["val"]      # type: ignore
    test_idx = split_info["test"]    # type: ignore

    # 2. Deteksi Mode: K-Fold vs Holdout
    is_kfold = "kfold" in config.lower() or config.startswith("folds_")

    # 3. Inisialisasi Sub-Dataset (Bersih dari target_channels)
    train_dataset = StrokeNPZDataset(images[train_idx], labels[train_idx])
    val_dataset = StrokeNPZDataset(images[val_idx], labels[val_idx])
    
    test_dataset = None
    if not is_kfold and len(test_idx) > 0:
        test_dataset = StrokeNPZDataset(images[test_idx], labels[test_idx])

    # 4. Parameter Kwargs DataLoader
    use_persistent = num_workers > 0
    base_kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": use_persistent,
    }
    if use_persistent:
        base_kwargs["prefetch_factor"] = 1

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **base_kwargs
    )
    val_loader = DataLoader(
        val_dataset, batch_size=eval_batch_size, shuffle=False, **base_kwargs
    )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset, batch_size=eval_batch_size, shuffle=False, **base_kwargs
        )

    # 5. Transformasi GPU On-The-Fly (Anatomy-Preserving CT Scan)
    gpu_train_transform = v2.Compose([
        v2.Lambda(
            lambda x: x.repeat(1, 3, 1, 1)
            if (x.ndim == 4 and x.shape[1] == 1 and num_channels == 3)
            else x
        ),
        v2.ToDtype(torch.float32, scale=False), 
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomAffine(
            degrees=5,  # type: ignore
            translate=(0.03, 0.03), 
            scale=(0.95, 1.05),
            interpolation=v2.InterpolationMode.BILINEAR,
            fill=0
        ),
        v2.RandomApply([
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))
        ], p=0.3),
    ])

    gpu_eval_transform = v2.Compose([
        v2.Lambda(
            lambda x: x.repeat(1, 3, 1, 1)
            if (x.ndim == 4 and x.shape[1] == 1 and num_channels == 3)
            else x
        ),
        v2.ToDtype(torch.float32, scale=False),
    ])

    num_classes = len(np.unique(labels))

    # 6. Return Output Membedakan Mode Split
    if is_kfold:
        return (
            train_loader,
            val_loader,
            num_classes,
            gpu_train_transform,
            gpu_eval_transform,
        )
    else:
        return (
            train_loader,
            val_loader,
            test_loader,
            num_classes,
            gpu_train_transform,
            gpu_eval_transform,
        )

if __name__ == "__main__":
    npz_path = "data/turkey_1channel.npz"

    # --- TEST HOLDOUT MODE ---
    holdout_config = "split_80_10_10"
    print("=" * 60)
    print(f"🧪 TEST LOADER: HOLDOUT ({holdout_config})")
    print("=" * 60)
    tr, val, ts, n_cls, tr_tf, ev_tf = get_stroke_dataloaders( # type: ignore
        npz_path=npz_path, config=holdout_config, num_channels=1, batch_size=16
    ) 
    print(f"✅ Mode Holdout Loaded Successfully!")
    print(f"   - Total Classes  : {n_cls}")
    print(f"   - Train Samples  : {len(tr.dataset)}")  # type: ignore
    print(f"   - Val Samples    : {len(val.dataset)}")  # type: ignore
    print(f"   - Test Samples   : {len(ts.dataset)}")  # type: ignore

    # --- TEST KFOLD MODE ---
    kfold_config = "folds_kfold_5"
    print("\n" + "=" * 60)
    print(f"🧪 TEST LOADER: KFOLD ({kfold_config} - Fold 0)")
    print("=" * 60)
    tr_kf, val_kf, n_cls_kf, _, _ = get_stroke_dataloaders(  # type: ignore
        npz_path=npz_path, config=kfold_config, fold_idx=0, num_channels=3, batch_size=16
    )
    print(f"✅ Mode K-Fold Loaded Successfully!")
    print(f"   - Total Classes  : {n_cls_kf}")
    print(f"   - Train Samples  : {len(tr_kf.dataset)}")  # type: ignore
    print(f"   - Val Samples    : {len(val_kf.dataset)}")  # type: ignore

