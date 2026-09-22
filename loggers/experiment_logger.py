import os
import json
import traceback
from datetime import datetime
import numpy as np
import pandas as pd

# Wajib dipanggil sebelum import pyplot agar aman di server headless/non-GUI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore

from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, cohen_kappa_score, matthews_corrcoef
)

class ExperimentLogger:
    def __init__(self, save_dir="./logs", class_names=["Normal", "Ischemia", "Bleeding"], hparams=None):
        self.save_dir = save_dir
        self.class_names = class_names
        self.hparams = hparams or {}
        os.makedirs(save_dir, exist_ok=True)

        # History mencatat val_mcc, lr, dan patience
        self.history = {
            'epoch': [], 'lr': [], 'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [], 'val_f1': [], 'val_mcc': [],
            'patience': []
        }
        
        if self.hparams:
            self.save_hparams()

    def save_hparams(self, filename="hparams.json"):
        """Menyimpan konfigurasi hyperparameter ke JSON."""
        path = os.path.join(self.save_dir, filename)
        with open(path, 'w') as f:
            json.dump(self.hparams, f, indent=4, default=str)
        print(f"⚙️ Hyperparameters disimpan di: {path}")

    def log_epoch(self, epoch, train_loss, val_loss, train_acc, val_acc, val_f1, val_mcc, lr=0.0, patience=0):
        """Catat seluruh metrik riwayat pelatihan per epoch."""
        self.history['epoch'].append(int(epoch))
        self.history['lr'].append(float(lr))
        self.history['train_loss'].append(float(train_loss))
        self.history['val_loss'].append(float(val_loss))
        self.history['train_acc'].append(float(train_acc))
        self.history['val_acc'].append(float(val_acc))
        self.history['val_f1'].append(float(val_f1))
        self.history['val_mcc'].append(float(val_mcc))
        self.history['patience'].append(int(patience))

    def export_csv(self, filename="training_history.csv"):
        """Ekspor riwayat epoch ke CSV."""
        if not self.history['epoch']:
            print("⚠️ Belum ada data history untuk diekspor ke CSV.")
            return None
        df = pd.DataFrame(self.history)
        path = os.path.join(self.save_dir, filename)
        df.to_csv(path, index=False)
        print(f"💾 File rekapitulasi CSV disimpan di: {path}")
        return path
    
    def load_history_from_csv(self, filename="training_history.csv"):
        """Memuat kembali riwayat pelatihan dari CSV jika training di-resume."""
        path = os.path.join(self.save_dir, filename)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                self.history = df.to_dict(orient='list')
                print(f"🔄 Riwayat log sebelumnya berhasil dimuat ({len(self.history['epoch'])} epoch ditemukan).")
            except Exception as e:
                print(f"⚠️ Gagal membaca history CSV sebelumnya: {e}")

    def log_error(self, error, context="RUNTIME_ERROR"):
        """Menangkap traceback exception atau interupsi dan mencatatnya ke error.txt."""
        error_path = os.path.join(self.save_dir, "error.txt")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        header = f"============================================================\n"
        header += f"🚨 ERROR / INTERRUPT CAPTURED [{timestamp}]\n"
        header += f"Context: {context}\n"
        header += f"============================================================\n"
        
        if isinstance(error, Exception):
            err_details = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        else:
            err_details = str(error) + "\n"

        with open(error_path, "a", encoding="utf-8") as f:
            f.write(header + err_details + "\n\n")

        print(f"\n🚨 [EMERGENCY LOG] Rincian error/interupsi berhasil dicatat di: {error_path}")

    def compute_metrics(self, y_true, y_pred, y_probs):
        """Hitung Confusion Matrix NxN dan seluruh metrik evaluasi."""
        if hasattr(y_true, 'cpu'): y_true = y_true.cpu().numpy()
        if hasattr(y_pred, 'cpu'): y_pred = y_pred.cpu().numpy()
        if hasattr(y_probs, 'cpu'): y_probs = y_probs.cpu().numpy()
        labels_idx = list(range(len(self.class_names)))
        
        cm = confusion_matrix(y_true, y_pred, labels=labels_idx)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        try:
            auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro', labels=labels_idx)
        except Exception:
            auc = 0.0

        kappa = cohen_kappa_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        prec_per_class = precision_score(y_true, y_pred, labels=labels_idx, average=None, zero_division=0)
        rec_per_class = recall_score(y_true, y_pred, labels=labels_idx, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, labels=labels_idx, average=None, zero_division=0)

        metrics = {
            "confusion_matrix": cm.tolist(),
            "global_metrics": {
                "accuracy": round(float(acc), 4),
                "precision_macro": round(float(prec), 4),
                "recall_macro": round(float(rec), 4),
                "f1_score_macro": round(float(f1), 4),
                "auc_roc_ovr": round(float(auc), 4),
                "cohens_kappa": round(float(kappa), 4),
                "mcc": round(float(mcc), 4)
            },
            "per_class_metrics": {
                name: {
                    "precision": round(float(prec_per_class[i]), 4), # type: ignore
                    "recall": round(float(rec_per_class[i]), 4), # type: ignore
                    "f1_score": round(float(f1_per_class[i]), 4) # type: ignore
                } for i, name in enumerate(self.class_names)
            }
        }
        return metrics, cm

    def save_best_results_json(self, train_eval, val_eval, test_eval, filename="best_model_metrics.json"):
        """Menyimpan snapshot performa terbaik ke format JSON dengan dukungan multiple test evaluations."""
        
        # 1. Hitung metrik Train & Validation
        train_metrics = self.compute_metrics(*train_eval)[0] if train_eval is not None else None
        val_metrics = self.compute_metrics(*val_eval)[0] if val_eval is not None else None

        # Struktur dasar JSON
        summary_json = {
            "dataset_info": {
                "num_classes": len(self.class_names),
                "class_labels": self.class_names
            },
            "hyperparameters": self.hparams,
            "best_train_results": train_metrics,
            "best_val_results": val_metrics,
        }

        # 2. Pastikan test_eval selalu berupa list (backward compatibility)
        if not isinstance(test_eval, list):
            test_eval_list = [test_eval] if test_eval is not None else []
        else:
            test_eval_list = test_eval

        # 3. Looping untuk membaca array test_eval dan mengisi JSON secara dinamis
        for i, single_test_eval in enumerate(test_eval_list, start=1):
            if single_test_eval is None:
                continue
            
            # Hitung metrik & confusion matrix per kandidat test
            test_metrics, cm_test = self.compute_metrics(*single_test_eval)
            
            # Tambahkan entri dinamis ke JSON: "best_test_1_results", "best_test_2_results", dst.
            summary_json[f"best_test_{i}_results"] = test_metrics
            
            # Simpan Confusion Matrix terpisah untuk tiap kandidat (test_confusion_matrix_1.png, test_confusion_matrix_2.png)
            cm_filename = f"test_confusion_matrix_{i}.png"
            self.plot_confusion_matrix(cm_test, filename=cm_filename)
            print(f"📊 Confusion Matrix Test {i} disimpan di: {os.path.join(self.save_dir, cm_filename)}")

        # 4. Tulis struktur JSON ke file
        json_path = os.path.join(self.save_dir, filename)
        with open(json_path, 'w') as f:
            json.dump(summary_json, f, indent=4, default=str)

        print(f"📄 Snapshot JSON (Train/Val/Multi-Test) berhasil disimpan di: {json_path}")
        return summary_json

    def plot_confusion_matrix(self, cm, filename="confusion_matrix.png"):
        """Visualisasi Confusion Matrix NxN Test Set."""
        n_classes = len(self.class_names)
        fig_size = max(7, int(n_classes * 0.8))
        font_size = max(6, 12 - int(n_classes * 0.4))
        
        plt.figure(figsize=(fig_size, fig_size))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.class_names, yticklabels=self.class_names,
            annot_kws={"size": font_size}
        )
        plt.title(f'Confusion Matrix {n_classes}x{n_classes} (Test Set)', fontweight='bold')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300)
        plt.close()

    def plot_learning_curves(self, test_eval=None, filename="training_dashboard.png", save_individual=True):
        """Visualisasi grafik pelatihan: 1 Dashboard Gabungan (6-in-1) + 6 File Grafik Terpisah (Total 7 PNG)."""
        if not self.history['epoch']:
            print("⚠️ Belum ada riwayat epoch untuk digambar grafiknya.")
            return
        sns.set_theme(style="whitegrid")
        
        epochs = self.history['epoch']
        best_idx = int(np.argmax(self.history['val_mcc'])) if self.history['val_mcc'] else 0
        best_epoch = epochs[best_idx] if epochs else 1
        best_mcc_val = self.history['val_mcc'][best_idx] if self.history['val_mcc'] else 0.0
        patience_limit = self.hparams.get('early_stop_patience', 10)

        # String ringkasan performa untuk Panel 6
        summary_text = (
            f"  === TRAINING STATUS SUMMARY ===\n\n"
            f" • Total Epochs Executed : {len(epochs)} / {self.hparams.get('max_epochs', '-')}\n"
            f" • Peak Model Saved Epoch: {best_epoch}\n"
            f" • Best Val MCC          : {best_mcc_val:.4f}\n"
            f" • Best Val F1 (Macro)   : {self.history['val_f1'][best_idx]:.4f}\n"
            f" • Best Val Accuracy     : {self.history['val_acc'][best_idx]:.4f}\n\n"
        )
        if test_eval is not None:
            test_metrics, _ = self.compute_metrics(*test_eval)
            g = test_metrics["global_metrics"]
            summary_text += (
                f"  === FINAL TEST BENCHMARK ===\n\n"
                f" • Test Accuracy  : {g['accuracy']:.4f}\n"
                f" • Test F1 Macro   : {g['f1_score_macro']:.4f}\n"
                f" • Test MCC        : {g['mcc']:.4f}\n"
                f" • Test ROC-AUC    : {g['auc_roc_ovr']:.4f}\n"
                f" • Cohen's Kappa   : {g['cohens_kappa']:.4f}"
            )

        # -------------------------------------------------------------
        # 1. RENDER DASHBOARD GABUNGAN 6-IN-1 (File ke-1)
        # -------------------------------------------------------------
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Panel 1: Loss Dynamics
        axes[0, 0].plot(epochs, self.history['train_loss'], label='Train Loss', color='crimson', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], label='Val Loss', color='navy', linestyle='--', linewidth=2)
        axes[0, 0].axvline(x=best_epoch, color='gold', linestyle=':', linewidth=2, label=f'Best Epoch ({best_epoch})')
        axes[0, 0].set_title('Loss Dynamics', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss'); axes[0, 0].legend()

        # Panel 2: Accuracy Progression
        axes[0, 1].plot(epochs, self.history['train_acc'], label='Train Acc', color='forestgreen', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_acc'], label='Val Acc', color='darkorange', linestyle='--', linewidth=2)
        axes[0, 1].axvline(x=best_epoch, color='gold', linestyle=':', linewidth=2, label=f'Best Epoch ({best_epoch})')
        axes[0, 1].set_title('Accuracy Progression', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Accuracy'); axes[0, 1].legend()

        # Panel 3: Val Metrics (F1 vs MCC)
        axes[0, 2].plot(epochs, self.history['val_f1'], label='Val F1 (Macro)', color='teal', linewidth=2)
        axes[0, 2].plot(epochs, self.history['val_mcc'], label='Val MCC', color='purple', linewidth=2)
        axes[0, 2].scatter([best_epoch], [best_mcc_val], color='gold', s=120, zorder=5, edgecolors='black', label=f'Peak MCC: {best_mcc_val:.4f}')
        axes[0, 2].axvline(x=best_epoch, color='gold', linestyle=':', linewidth=2)
        axes[0, 2].set_title('Validation Quality (F1 vs MCC)', fontweight='bold')
        axes[0, 2].set_xlabel('Epoch'); axes[0, 2].set_ylabel('Score'); axes[0, 2].legend()

        # Panel 4: Learning Rate Schedule
        axes[1, 0].plot(epochs, self.history['lr'], label='Effective LR', color='darkmagenta', linewidth=2)
        axes[1, 0].set_title('Learning Rate Schedule', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].ticklabel_format(style='scientific', scilimits=(0, 0), axis='y'); axes[1, 0].legend()

        # Panel 5: Patience Tracker
        axes[1, 1].plot(epochs, self.history['patience'], label='Patience Count', color='orangered', linewidth=2, marker='o', markersize=4)
        axes[1, 1].axhline(y=patience_limit, color='red', linestyle='--', linewidth=2, label=f'Patience Limit ({patience_limit})')
        axes[1, 1].set_title('Early Stopping Patience Tracker', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('Stagnant Epochs'); axes[1, 1].legend()

        # Panel 6: Summary Text Card
        axes[1, 2].axis('off')
        axes[1, 2].text(
            0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
            fontsize=10.5, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='whitesmoke', alpha=0.9, edgecolor='lightgray')
        )

        plt.tight_layout()
        dashboard_path = os.path.join(self.save_dir, filename)
        plt.savefig(dashboard_path, dpi=300)
        plt.close()
        print(f"📊 [1/7] Dashboard Gabungan 6-in-1 disimpan di: {dashboard_path}")

        # -------------------------------------------------------------
        # 2. EKSPOR 6 GRAFIK TERPISAH (File ke 2 s.d 7)
        # -------------------------------------------------------------
        if save_individual:
            plots_dir = os.path.join(self.save_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)

            # File 2: Loss Curve
            plt.figure(figsize=(7, 5))
            plt.plot(epochs, self.history['train_loss'], label='Train Loss', color='crimson', linewidth=2)
            plt.plot(epochs, self.history['val_loss'], label='Val Loss', color='navy', linestyle='--', linewidth=2)
            plt.axvline(x=best_epoch, color='gold', linestyle=':', linewidth=2, label=f'Best Epoch ({best_epoch})')
            plt.title('Loss Dynamics per Epoch', fontweight='bold')
            plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "loss_curve.png"), dpi=300); plt.close()

            # File 3: Accuracy Curve
            plt.figure(figsize=(7, 5))
            plt.plot(epochs, self.history['train_acc'], label='Train Acc', color='forestgreen', linewidth=2)
            plt.plot(epochs, self.history['val_acc'], label='Val Acc', color='darkorange', linestyle='--', linewidth=2)
            plt.axvline(x=best_epoch, color='gold', linestyle=':', linewidth=2, label=f'Best Epoch ({best_epoch})')
            plt.title('Accuracy Progression per Epoch', fontweight='bold')
            plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "accuracy_curve.png"), dpi=300); plt.close()

            # File 4: Val Metrics (F1 vs MCC)
            plt.figure(figsize=(7, 5))
            plt.plot(epochs, self.history['val_f1'], label='Val F1 (Macro)', color='teal', linewidth=2)
            plt.plot(epochs, self.history['val_mcc'], label='Val MCC', color='purple', linewidth=2)
            plt.scatter([best_epoch], [best_mcc_val], color='gold', s=120, zorder=5, edgecolors='black', label=f'Peak MCC: {best_mcc_val:.4f}')
            plt.axvline(x=best_epoch, color='gold', linestyle=':', linewidth=2)
            plt.title('Validation Quality (F1 vs MCC)', fontweight='bold')
            plt.xlabel('Epoch'); plt.ylabel('Score'); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "val_metrics_mcc_f1.png"), dpi=300); plt.close()

            # File 5: Learning Rate Schedule
            plt.figure(figsize=(7, 5))
            plt.plot(epochs, self.history['lr'], label='Effective LR', color='darkmagenta', linewidth=2)
            plt.title('Learning Rate Schedule per Epoch', fontweight='bold')
            plt.xlabel('Epoch'); plt.ylabel('Learning Rate')
            plt.ticklabel_format(style='scientific', scilimits=(0, 0), axis='y')
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "lr_schedule.png"), dpi=300); plt.close()

            # File 6: Patience Tracker
            plt.figure(figsize=(7, 5))
            plt.plot(epochs, self.history['patience'], label='Patience Count', color='orangered', linewidth=2, marker='o', markersize=4)
            plt.axhline(y=patience_limit, color='red', linestyle='--', linewidth=2, label=f'Patience Limit ({patience_limit})')
            plt.title('Early Stopping Patience Tracker', fontweight='bold')
            plt.xlabel('Epoch'); plt.ylabel('Stagnant Epochs'); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "patience_tracker.png"), dpi=300); plt.close()

            # File 7: Performance & Test Benchmark Summary Card
            plt.figure(figsize=(7, 5))
            plt.axis('off')
            plt.text(
                0.05, 0.95, summary_text, transform=plt.gca().transAxes,
                fontsize=10.5, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='whitesmoke', alpha=0.9, edgecolor='lightgray')
            )
            plt.title('Performance & Benchmark Summary', fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "summary_card.png"), dpi=300); plt.close()

            print(f"📁 [2-7/7] 6 Grafik terpisah berhasil disimpan di folder: {plots_dir}")
        
if __name__ == "__main__":

    code_experiment = "exp-1"
    epochs = 10

    logger = ExperimentLogger(save_dir=f"./{code_experiment}")

    # 1. Di dalam loop epoch (selama training)
    for epoch in range(1, epochs + 1):
        # ... jalankan train & val ...
        # misal dummy
        train_loss = val_loss = train_acc = val_acc = val_f1 = val_mcc = 0
        logger.log_epoch(epoch, train_loss, val_loss, train_acc, val_acc, val_f1, val_mcc)

    # Ekspor CSV history dan plot grafik pelatihan
    logger.export_csv()
    logger.plot_learning_curves()

    np.random.seed(42)

    def generate_dummy_eval_data(n_samples=500):
        """
        Menghasilkan dummy y_true, y_pred, dan y_probs untuk 3 kelas:
        0: Bleeding, 1: Ischemia, 2: Normal
        """
        y_true = np.random.choice([0, 1, 2], size=n_samples, p=[0.2, 0.2, 0.6])

        raw_logits = np.random.randn(n_samples, 3)
        for i in range(n_samples):
            raw_logits[i, y_true[i]] += np.random.uniform(1.5, 3.0)

        exp_logits = np.exp(raw_logits - np.max(raw_logits, axis=1, keepdims=True))
        y_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        y_pred = np.argmax(y_probs, axis=1)

        return y_true, y_pred, y_probs

    y_train_true, y_train_pred, y_train_probs = generate_dummy_eval_data(n_samples=1000)
    y_val_true, y_val_pred, y_val_probs       = generate_dummy_eval_data(n_samples=200)
    y_test_true, y_test_pred, y_test_probs     = generate_dummy_eval_data(n_samples=200)

    train_data = (y_train_true, y_train_pred, y_train_probs)
    val_data   = (y_val_true, y_val_pred, y_val_probs)
    test_data  = (y_test_true, y_test_pred, y_test_probs)

    logger.save_best_results_json(train_data, val_data, test_data)