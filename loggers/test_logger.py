import os
import json
import numpy as np
import pandas as pd

# Wajib untuk server headless / GPU instance tanpa GUI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore

from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, cohen_kappa_score,
    matthews_corrcoef, roc_curve, precision_recall_curve, auc
)
from sklearn.preprocessing import label_binarize


class TestLogger:
    """Logger khusus untuk mode Evaluasi/Testing Murni (tanpa riwayat epoch/training)."""
    def __init__(self, save_dir="./test_results", class_names=None, hparams=None):
        self.save_dir = save_dir
        self.class_names = class_names or ["Bleeding", "Ischemia", "Normal"]
        self.num_classes = len(self.class_names)
        self.hparams = hparams or {}
        os.makedirs(save_dir, exist_ok=True)
        
        # Subfolder khusus gambar
        self.plots_dir = os.path.join(save_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

    def compute_metrics(self, y_true, y_pred, y_probs):
        """Kalkulasi lengkap Confusion Matrix, Global Metrics, dan Per-Class Metrics."""
        labels_idx = list(range(self.num_classes))
        
        cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

        acc = accuracy_score(y_true, y_pred)
        prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

        try:
            auc_macro = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro', labels=labels_idx)
        except Exception:
            auc_macro = 0.0

        kappa = cohen_kappa_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        prec_per_class = precision_score(y_true, y_pred, labels=labels_idx, average=None, zero_division=0)
        rec_per_class = recall_score(y_true, y_pred, labels=labels_idx, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, labels=labels_idx, average=None, zero_division=0)

        metrics = {
            "confusion_matrix_raw": cm.tolist(),
            "confusion_matrix_normalized": np.round(cm_norm, 4).tolist(),
            "global_metrics": {
                "accuracy": round(float(acc), 4),
                "precision_macro": round(float(prec_macro), 4),
                "recall_macro": round(float(rec_macro), 4),
                "f1_score_macro": round(float(f1_macro), 4),
                "auc_roc_ovr_macro": round(float(auc_macro), 4),
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
        return metrics, cm, cm_norm

    def plot_confusion_matrices(self, cm, cm_norm):
        """Visualisasi Confusion Matrix (Raw Counts & Normalized %)."""
        n = self.num_classes
        fig_size = max(7, int(n * 0.85))
        font_size = max(6, 12 - int(n * 0.35))

        fig, axes = plt.subplots(1, 2, figsize=(fig_size * 2, fig_size))

        # 1. Raw Counts
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=self.class_names, yticklabels=self.class_names,
            annot_kws={"size": font_size}
        )
        axes[0].set_title(f'Confusion Matrix (Counts) - {n}x{n}', fontweight='bold', fontsize=12)
        axes[0].set_xlabel('Predicted Class'); axes[0].set_ylabel('True Class')
        axes[0].tick_params(axis='x', rotation=45)

        # 2. Normalized Percentage
        sns.heatmap(
            cm_norm, annot=True, fmt='.2%', cmap='Greens', ax=axes[1],
            xticklabels=self.class_names, yticklabels=self.class_names,
            annot_kws={"size": font_size}
        )
        axes[1].set_title(f'Confusion Matrix (Normalized) - {n}x{n}', fontweight='bold', fontsize=12)
        axes[1].set_xlabel('Predicted Class'); axes[1].set_ylabel('True Class')
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        path = os.path.join(self.plots_dir, "confusion_matrices.png")
        plt.savefig(path, dpi=300)
        plt.close()

    def plot_roc_curves(self, y_true, y_probs):
        """Visualisasi Multi-Class ROC-AUC Curves (One-vs-Rest)."""
        y_true_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
        if self.num_classes == 2 and y_true_bin.shape[1] == 1:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin]) # type: ignore

        plt.figure(figsize=(8, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_classes))

        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i]) # type: ignore
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'{class_name} (AUC = {roc_auc:.4f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance')
        plt.xlim([-0.01, 1.0]) # type: ignore
        plt.ylim([0.0, 1.05]) # type: ignore
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('Multi-Class Receiver Operating Characteristic (ROC)', fontweight='bold')
        plt.legend(loc="lower right", fontsize=9)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()

        path = os.path.join(self.plots_dir, "roc_curves.png")
        plt.savefig(path, dpi=300)
        plt.close()

    def plot_precision_recall_curves(self, y_true, y_probs):
        """Visualisasi Multi-Class Precision-Recall Curves."""
        y_true_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
        if self.num_classes == 2 and y_true_bin.shape[1] == 1:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin]) # type: ignore

        plt.figure(figsize=(8, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_classes))

        for i, class_name in enumerate(self.class_names):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i]) # type: ignore
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, color=colors[i], lw=2, label=f'{class_name} (PR-AUC = {pr_auc:.4f})')

        plt.xlim([0.0, 1.05]) # type: ignore
        plt.ylim([0.0, 1.05]) # type: ignore
        plt.xlabel('Recall (Sensitivity)')
        plt.ylabel('Precision (PPV)')
        plt.title('Multi-Class Precision-Recall Curves', fontweight='bold')
        plt.legend(loc="lower left", fontsize=9)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()

        path = os.path.join(self.plots_dir, "precision_recall_curves.png")
        plt.savefig(path, dpi=300)
        plt.close()

    def plot_per_class_bar(self, per_class_metrics):
        """Grouped Bar Chart untuk Precision, Recall, dan F1-Score per kelas."""
        classes = list(per_class_metrics.keys())
        precisions = [per_class_metrics[c]['precision'] for c in classes]
        recalls = [per_class_metrics[c]['recall'] for c in classes]
        f1s = [per_class_metrics[c]['f1_score'] for c in classes]

        x = np.arange(len(classes))
        width = 0.25

        fig_width = max(8, int(len(classes) * 0.9))
        plt.figure(figsize=(fig_width, 5.5))

        plt.bar(x - width, precisions, width, label='Precision', color='#3498db')
        plt.bar(x, recalls, width, label='Recall', color='#2ecc71')
        plt.bar(x + width, f1s, width, label='F1-Score', color='#e74c3c')

        plt.xlabel('Class Label', fontweight='bold')
        plt.ylabel('Score (0.0 - 1.0)', fontweight='bold')
        plt.title('Per-Class Performance Breakdown', fontweight='bold')
        plt.xticks(x, classes, rotation=35, ha='right')
        plt.ylim(0, 1.15)
        plt.legend(loc='upper right')
        plt.grid(axis='y', linestyle=':', alpha=0.7)

        # Annotate bar values
        for i in range(len(classes)):
            plt.text(x[i] - width, precisions[i] + 0.02, f"{precisions[i]:.2f}", ha='center', fontsize=7, rotation=90)
            plt.text(x[i], recalls[i] + 0.02, f"{recalls[i]:.2f}", ha='center', fontsize=7, rotation=90)
            plt.text(x[i] + width, f1s[i] + 0.02, f"{f1s[i]:.2f}", ha='center', fontsize=7, rotation=90)

        plt.tight_layout()
        path = os.path.join(self.plots_dir, "per_class_metrics_bar.png")
        plt.savefig(path, dpi=300)
        plt.close()

    def plot_test_dashboard(self, cm_norm, metrics, y_true, y_probs):
        """Dashboard Gabungan 4-in-1 untuk laporan visual cepat (Test Set Only)."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        sns.set_theme(style="whitegrid")

        # 1. Normalized Confusion Matrix
        sns.heatmap(
            cm_norm, annot=True, fmt='.1%', cmap='Blues', ax=axes[0, 0],
            xticklabels=self.class_names, yticklabels=self.class_names
        )
        axes[0, 0].set_title('Normalized Confusion Matrix', fontweight='bold')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. Per-Class Bar Chart
        pcm = metrics['per_class_metrics']
        classes = list(pcm.keys())
        x = np.arange(len(classes))
        width = 0.25
        axes[0, 1].bar(x - width, [pcm[c]['precision'] for c in classes], width, label='Precision', color='#3498db')
        axes[0, 1].bar(x, [pcm[c]['recall'] for c in classes], width, label='Recall', color='#2ecc71')
        axes[0, 1].bar(x + width, [pcm[c]['f1_score'] for c in classes], width, label='F1-Score', color='#e74c3c')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(classes, rotation=35, ha='right')
        axes[0, 1].set_title('Per-Class Performance', fontweight='bold')
        axes[0, 1].legend(fontsize=8)

        # 3. Multi-Class ROC Curves
        y_true_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
        if self.num_classes == 2 and y_true_bin.shape[1] == 1:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin]) # type: ignore

        colors = plt.cm.tab10(np.linspace(0, 1, self.num_classes))
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i]) # type: ignore
            roc_auc = auc(fpr, tpr)
            axes[1, 0].plot(fpr, tpr, color=colors[i], label=f'{class_name} ({roc_auc:.3f})')

        axes[1, 0].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[1, 0].set_title('ROC Curves (One-vs-Rest)', fontweight='bold')
        axes[1, 0].legend(fontsize=7, loc='lower right')

        # 4. Summary Text Card
        axes[1, 1].axis('off')
        gm = metrics['global_metrics']
        summary_text = (
            f"  === STANDALONE TEST BENCHMARK REPORT ===\n\n"
            f"  • Total Test Samples : {len(y_true)}\n"
            f"  • Number of Classes  : {self.num_classes}\n\n"
            f"  -----------------------------------------\n"
            f"  • Accuracy           : {gm['accuracy'] * 100:.2f}% ({gm['accuracy']:.4f})\n"
            f"  • Macro F1-Score     : {gm['f1_score_macro']:.4f}\n"
            f"  • Matthews CorrCoef  : {gm['mcc']:.4f}\n"
            f"  • Macro Precision    : {gm['precision_macro']:.4f}\n"
            f"  • Macro Recall       : {gm['recall_macro']:.4f}\n"
            f"  • Macro ROC-AUC      : {gm['auc_roc_ovr_macro']:.4f}\n"
            f"  • Cohen's Kappa      : {gm['cohens_kappa']:.4f}\n"
            f"  -----------------------------------------\n"
        )
        axes[1, 1].text(
            0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1.0', facecolor='whitesmoke', alpha=0.9, edgecolor='lightgray')
        )

        plt.tight_layout()
        dashboard_path = os.path.join(self.save_dir, "test_summary_dashboard.png")
        plt.savefig(dashboard_path, dpi=300)
        plt.close()
        print(f"📊 Summary Dashboard 4-in-1 disimpan di: {dashboard_path}")

    def evaluate_and_log(self, y_true, y_pred, y_probs, filename="test_evaluation_results.json"):
        """Fungsi utama: Menghitung metrik, mengekspor JSON, dan menggambar seluruh grafik visual."""
        metrics, cm, cm_norm = self.compute_metrics(y_true, y_pred, y_probs)

        output_data = {
            "dataset_info": {
                "num_samples": len(y_true),
                "num_classes": self.num_classes,
                "class_labels": self.class_names
            },
            "hyperparameters": self.hparams,
            "test_results": metrics
        }

        # 1. Simpan Laporan JSON
        json_path = os.path.join(self.save_dir, filename)
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=4, default=str)
        print(f"📄 Hasil Evaluasi JSON disimpan di: {json_path}")

        # 2. Render Seluruh Grafik Visual PNG
        print("🎨 Meringkas visualisasi grafik (PNG)...")
        self.plot_confusion_matrices(cm, cm_norm)
        self.plot_roc_curves(y_true, y_probs)
        self.plot_precision_recall_curves(y_true, y_probs)
        self.plot_per_class_bar(metrics['per_class_metrics'])
        self.plot_test_dashboard(cm_norm, metrics, y_true, y_probs)

        print(f"🖼️ Seluruh grafik individual tersimpan rapi di: {self.plots_dir}")
        return metrics