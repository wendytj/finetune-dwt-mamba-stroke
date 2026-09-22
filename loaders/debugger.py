import os, sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from architectures.DWTMamba import DWTMamba

def _to_tensor_img(arr):
    """Mengonversi numpy array citra ke PyTorch Tensor [B, C, H, W] berkisar [0, 1]."""
    if len(arr) == 0:
        return torch.empty(0)

    tensor = torch.tensor(arr, dtype=torch.float32)
    
    # Penyesuaian Dimensi -> Target: [B, C, H, W]
    if tensor.ndim == 3:  # [B, H, W] -> [B, 1, H, W]
        tensor = tensor.unsqueeze(1)
    elif tensor.ndim == 4 and tensor.shape[-1] in [1, 3]:  # [B, H, W, C] -> [B, C, H, W]
        tensor = tensor.permute(0, 3, 1, 2)
    
    # FIX DOUBLE NORMALIZATION: Hanya bagi 255 jika max > 1.5 (asumsi data 0-255)
    if tensor.numel() > 0 and tensor.max() > 1.5:
        tensor /= 255.0
        
    return tensor

def unpack_split_info(split_entry, fold_idx=0):
    """Membongkar indeks split secara fleksibel untuk Holdout & K-Fold."""
    # 1. Jika tersimpan sebagai 0D object / dict
    if split_entry.shape == () or split_entry.size == 1:
        entry_item = split_entry.item()
        if isinstance(entry_item, dict):
            if 'train' in entry_item:
                return entry_item
            elif fold_idx in entry_item or str(fold_idx) in entry_item:
                return entry_item.get(fold_idx, entry_item.get(str(fold_idx)))
        split_entry = entry_item

    # 2. Jika tersimpan sebagai tuple/list 3 array [train_idx, val_idx, test_idx]
    if isinstance(split_entry, (list, np.ndarray)) and len(split_entry) in [2, 3] and split_entry.dtype == object: # type: ignore
        return {
            'train': split_entry[0],
            'val': split_entry[1],
            'test': split_entry[2] if len(split_entry) > 2 else split_entry[1]
        }

    # 3. Jika tersimpan sebagai K-Fold list
    if isinstance(split_entry, (list, np.ndarray)) and len(split_entry) > 0 and len(split_entry) <= 10:
        selected_fold = split_entry[fold_idx % len(split_entry)]
        if isinstance(selected_fold, dict):
            return {
                'train': selected_fold.get('train', np.array([])),
                'val': selected_fold.get('val', selected_fold.get('test', np.array([]))),
                'test': selected_fold.get('test', np.array([]))
            }
        elif isinstance(selected_fold, (list, tuple, np.ndarray)) and len(selected_fold) >= 2:
            return {
                'train': selected_fold[0],
                'val': selected_fold[1],
                'test': selected_fold[2] if len(selected_fold) > 2 else selected_fold[1]
            }

    # 4. Jika tersimpan sebagai Array 1D penanda per sampel (Length == 6551)
    if isinstance(split_entry, np.ndarray) and split_entry.ndim == 1 and len(split_entry) > 10:
        # Pengecekan jika array berisi STRING ('train', 'val', 'test')
        if np.issubdtype(split_entry.dtype, np.character) or isinstance(split_entry[0], str):
            train_idx = np.where(split_entry == 'train')[0]
            val_idx   = np.where(split_entry == 'val')[0]
            test_idx  = np.where(split_entry == 'test')[0]
            return {'train': train_idx, 'val': val_idx, 'test': test_idx}
        # Jika array berisi INTEGER penanda fold (0, 1, 2, ...)
        else:
            val_idx = np.where(split_entry == fold_idx)[0]
            train_idx = np.where(split_entry != fold_idx)[0]
            return {'train': train_idx, 'val': val_idx, 'test': val_idx}

    return {'train': np.array([]), 'val': np.array([]), 'test': np.array([])}

def verify_stratification(npz_path="data/turkey_1channel.npz", config="split_80_10_10", class_names=None):
    if class_names is None:
        class_names = {0: "Normal", 1: "Ischemia", 2: "Bleeding"}

    if not os.path.exists(npz_path):
        print(f"❌ File {npz_path} tidak ditemukan!")
        return

    data = np.load(npz_path, allow_pickle=True)
    labels = data['labels'].squeeze()
    total_samples = len(labels)

    if config not in data:
        print(f"❌ Key '{config}' tidak ditemukan di NPZ!")
        return

    split_info = unpack_split_info(data[config])
    train_idx = split_info['train']  # type: ignore
    val_idx = split_info['val']  # type: ignore
    test_idx = split_info['test']  # type: ignore

    # Ekstrak label per subset
    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx] if len(test_idx) > 0 else np.array([])

    # Buat Rangkuman Ringkasan Tabel
    df_summary = pd.DataFrame(index=[f"Class {k} ({v})" for k, v in class_names.items()])

    # Total Dataset (.values ditambahkan agar tidak mismatch index)
    total_counts = pd.Series(labels).value_counts().sort_index()
    df_summary['Total (Count)'] = total_counts.values
    df_summary['Total (%)'] = ((total_counts / total_samples) * 100).round(2).values

    # Train Split
    train_counts = pd.Series(y_train).value_counts().sort_index()
    df_summary['Train (Count)'] = train_counts.values
    df_summary['Train (%)'] = ((train_counts / len(y_train)) * 100).round(2).values

    # Val Split
    val_counts = pd.Series(y_val).value_counts().sort_index()
    df_summary['Val (Count)'] = val_counts.values
    df_summary['Val (%)'] = ((val_counts / len(y_val)) * 100).round(2).values

    # Test Split (jika ada dan beda dari Val)
    if len(y_test) > 0 and not np.array_equal(val_idx, test_idx):
        test_counts = pd.Series(y_test).value_counts().sort_index()
        df_summary['Test (Count)'] = test_counts.values
        df_summary['Test (%)'] = ((test_counts / len(y_test)) * 100).round(2).values

    print("=" * 75)
    print(f"📊 ANALISIS STRATIFIKASI LABEL: {config}")
    print(f"📂 Source File: {npz_path} (Total Data: {total_samples} Irisan)")
    print("=" * 75)
    print(df_summary.to_string())
    print("-" * 75)

    # Evaluasi Toleransi Stratifikasi
    train_diff = np.abs(df_summary['Total (%)'].values - df_summary['Train (%)'].values) # type: ignore
    val_diff = np.abs(df_summary['Total (%)'].values - df_summary['Val (%)'].values)  # type: ignore

    max_diff = max(train_diff.max(), val_diff.max())
    print(f"🔍 Selisih Proporsi Maksimum terhadap Populasi Total: {max_diff:.2f}%")

    if max_diff <= 1.0:
        print("✅ VERDIK: Pembagian TERSTRATIFIKASI Sempurna! Proporsi kelas di setiap subset identik dengan populasi asli.")
    else:
        print("⚠️ VERDIK: Terdapat sedikit pergeseran (*skew*) proporsi antar subset.")

def run_diagnostics(npz_path="data/turkey_1channel.npz", config='split_80_10_10', embed_dim=256, depth=3):
    print("=" * 60)
    print("🔍 DWTMAMBA DIAGNOSTIC TOOL")
    print("=" * 60)

    # -------------------------------------------------------------
    # TEST 1: PENGECEKAN FILE NPZ ASLI & SLICING
    # -------------------------------------------------------------
    if not os.path.exists(npz_path):
        print(f"❌ File {npz_path} tidak ditemukan! Pastikan path benar.")
        return

    data = np.load(npz_path, allow_pickle=True)
    print(f"\n📂 1. NPZ File Inspection ({npz_path}):")
    print(f"   - Keys tersedia       : {list(data.keys())}")

    if 'images' not in data or 'labels' not in data:
        print("❌ Key 'images' atau 'labels' tidak ditemukan di NPZ!")
        return

    images = data['images']  # Shape: (N, H, W) atau (N, H, W, C)
    labels = data['labels']  # Shape: (N,)

    if config not in data:
        print(f"❌ Config split '{config}' tidak ditemukan di file NPZ!")
        return

    # Unpack & Ekstrak Indeks
    split_info = unpack_split_info(data[config])
    train_idx, val_idx, test_idx = split_info['train'], split_info['val'], split_info['test'] # type: ignore

    # Slicing Array Gambar & Label (NumPy Views - Hemat RAM)
    train_images, train_labels = images[train_idx], labels[train_idx]
    val_images, val_labels = images[val_idx], labels[val_idx]
    test_images, test_labels = images[test_idx], labels[test_idx]

    print(f"\n📊 {config} Info:")
    print(f"   - Train indices count : {len(train_idx)}")
    print(f"   - Val indices count   : {len(val_idx)}")
    print(f"   - Test indices count  : {len(test_idx)}")

    # -------------------------------------------------------------
    # TEST 2: PENGECEKAN PRAPROSES & NORMALISASI POPULASI
    # -------------------------------------------------------------
    print("\n🔄 2. DataLoader & Normalization Test:")
    
    if len(train_images) == 0:
        print("❌ Train images kosong! Periksa kembali key config split.")
        return

    # Hitung Min/Max dari SELURUH Populasi NumPy (Cepat, 100% Akurat, 0 RAM Tambahan)
    tr_min, tr_max = float(train_images.min()), float(train_images.max())
    print(f"   - Train NumPy Range (Seluruh Data) : Min {tr_min:.4f} | Max {tr_max:.4f}")

    has_val = len(val_images) > 0
    if has_val:
        val_min, val_max = float(val_images.min()), float(val_images.max())
        print(f"   - Val NumPy Range (Seluruh Data)   : Min {val_min:.4f} | Max {val_max:.4f}")
        if abs(tr_max - val_max) > 0.5:
            print("   ⚠️  WARNING DETECTED: Skala gambar Train dan Val BERBEDA!")
        else:
            print("   ✅ Skala gambar Train dan Val Identik.")
    else:
        print("   ⚠️ Val images kosong pada config ini.")

    # Ambil 32 Sampel HANYA untuk Pembuatan Tensor & Testing Forward Pass Model
    sample_size = min(32, len(val_images) if has_val else len(train_images))
    eval_imgs = val_images[:sample_size] if has_val else train_images[:sample_size]
    eval_lbls = val_labels[:sample_size] if has_val else train_labels[:sample_size]

    sample_x = _to_tensor_img(eval_imgs)
    # Gunakan .reshape(-1) agar aman dari squeeze bug pada batch_size=1
    sample_y = torch.tensor(eval_lbls, dtype=torch.long).reshape(-1)

    in_channels = sample_x.shape[1]

    print(f"   - Batch Sample Tensor Shape        : {sample_x.shape} (Channels: {in_channels})")
    print(f"   - Batch Sample Label Shape         : {sample_y.shape} (Vector 1D [B])")

    # -------------------------------------------------------------
    # TEST 3: PENGECEKAN MODEL BEHAVIOR (train() vs eval())
    # -------------------------------------------------------------
    print("\n🧠 3. Model Forward & Mode Test (train vs eval):")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   - Running on Device   : {device}")

    num_classes = len(np.unique(labels))
    print(f"   - Terdeteksi Jml Kelas: {num_classes}")

    model = DWTMamba(
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        se_reduction=16,
        mb_gsf_reduction=4,
        latent_dim=128,
        proj_dim=256
    ).to(device)  # type: ignore

    criterion = nn.CrossEntropyLoss()

    # Siapkan DataLoader 1 Batch
    val_dataset = TensorDataset(sample_x, sample_y)
    val_loader = DataLoader(val_dataset, batch_size=sample_size, shuffle=False)
    batch_x, batch_y = next(iter(val_loader))
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

    # Test Mode TRAIN
    model.train()
    with torch.no_grad():
        out_train_mode = model(batch_x)
        loss_train_mode = criterion(out_train_mode, batch_y).item()
        preds_train = out_train_mode.argmax(dim=1)
        acc_train = (preds_train == batch_y).float().mean().item()

    # Test Mode EVAL
    model.eval()
    with torch.no_grad():
        out_eval_mode = model(batch_x)
        loss_eval_mode = criterion(out_eval_mode, batch_y).item()
        preds_eval = out_eval_mode.argmax(dim=1)
        acc_eval = (preds_eval == batch_y).float().mean().item()

    print(f"   - Loss mode model.train() : {loss_train_mode:.4f} | Acc: {acc_train*100:.2f}%")
    print(f"   - Loss mode model.eval()  : {loss_eval_mode:.4f} | Acc: {acc_eval*100:.2f}%")

    # -------------------------------------------------------------
    # SUMMARY DIAGNOSTIK
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📋 HASIL EVALUASI DIAGNOSTIK:")
    print("=" * 60)

    if loss_eval_mode > 5.0 and loss_train_mode < 3.0:
        print("🚨 MASALAH DITEMUKAN: BATCHNORM / EVAL MODE EXPLOSION!")
        print("   Penyebab: BatchNorm2d pada model merusak aktivasi fitur saat mode eval() dipanggil.")
        print("   Solusi  : Gunakan LayerNorm atau latih model 3-5 epoch terlebih dahulu agar statistik stabil.")
    elif abs(tr_max - (val_max if has_val else tr_max)) > 0.5:
        print("🚨 MASALAH DITEMUKAN: MISMATCH NORMALISASI DATA!")
        print("   Penyebab: Skala data Train dan Val berbeda jauh.")
    else:
        print("✅ Struktur data, dimensi channel, dan model dasar berfungsi normal.")

if __name__ == "__main__":
    npz_relative_path = "data/turkey_3channel.npz"
    config = "split_90_5_5"
    run_diagnostics(npz_relative_path, config)
    verify_stratification(npz_relative_path, config)