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

    def log_epoch(self, epoch, train_loss, val_loss, train_acc, val_acc, val_f1, val_mcc, lr=0.0, patience=0, class_weights=None):
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

        if class_weights is not None:
            cw_list = (
                class_weights.cpu().tolist()
                if hasattr(class_weights, "cpu")
                else list(class_weights)
            )
            for i, w in enumerate(cw_list):
                cls_name = (
                    self.class_names[i]
                    if i < len(self.class_names)
                    else f"cls_{i}"
                )
                col_key = f"cw_{cls_name}"

                # Proteksi Mismatch: Jika kolom baru muncul di tengah epoch, isi epoch sebelumnya dengan NaN/1.0
                if col_key not in self.history:
                    self.history[col_key] = [np.nan] * (
                        len(self.history["epoch"]) - 1
                    )

                self.history[col_key].append(round(float(w), 4))
        else:
            # Jika ada kolom cw_ sebelumnya tetapi epoch ini class_weights=None, isi dengan NaN agar panjang list tetap sama
            for key in list(self.history.keys()):
                if key.startswith("cw_") and len(self.history[key]) < len( # type: ignore
                    self.history["epoch"]
                ):
                    self.history[key].append(np.nan) # type: ignore

    def export_csv(self, filename="training_history.csv"):
        """Ekspor riwayat epoch ke CSV."""
        if not self.history['epoch']:
            print("⚠️ Belum ada data history untuk diekspor ke CSV.")
            return None
        df = pd.DataFrame(self.history)
        path = os.path.join(self.save_dir, filename)
        df.to_csv(path, index=False)
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
        """Hitung Confusion Matrix NxN dan seluruh metrik evaluasi (termasuk OvR MCC per-kelas)."""
        if hasattr(y_true, "cpu"):
            y_true = y_true.cpu().numpy()
        if hasattr(y_pred, "cpu"):
            y_pred = y_pred.cpu().numpy()
        if hasattr(y_probs, "cpu"):
            y_probs = y_probs.cpu().numpy()
        labels_idx = list(range(len(self.class_names)))

        cm = confusion_matrix(y_true, y_pred, labels=labels_idx)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        try:
            auc = roc_auc_score(
                y_true,
                y_probs,
                multi_class="ovr",
                average="macro",
                labels=labels_idx,
            )
        except Exception:
            auc = 0.0

        kappa = cohen_kappa_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        prec_per_class = precision_score(
            y_true, y_pred, labels=labels_idx, average=None, zero_division=0
        )
        rec_per_class = recall_score(
            y_true, y_pred, labels=labels_idx, average=None, zero_division=0
        )
        f1_per_class = f1_score(
            y_true, y_pred, labels=labels_idx, average=None, zero_division=0
        )

        # 🌟 Kalkulasi One-vs-Rest (OvR) MCC Per-Kelas
        mcc_per_class = []
        for i in range(len(self.class_names)):
            y_true_bin = (y_true == i).astype(int)
            y_pred_bin = (y_pred == i).astype(int)
            try:
                mcc_i = matthews_corrcoef(y_true_bin, y_pred_bin)
            except Exception:
                mcc_i = 0.0
            mcc_per_class.append(mcc_i)

        metrics = {
            "confusion_matrix": cm.tolist(),
            "global_metrics": {
                "accuracy": round(float(acc), 4),
                "precision_macro": round(float(prec), 4),
                "recall_macro": round(float(rec), 4),
                "f1_score_macro": round(float(f1), 4),
                "auc_roc_ovr": round(float(auc), 4),
                "cohens_kappa": round(float(kappa), 4),
                "mcc": round(float(mcc), 4),
            },
            "per_class_metrics": {
                name: {
                    "precision": round(float(prec_per_class[i]), 4),  # type: ignore
                    "recall": round(float(rec_per_class[i]), 4),  # type: ignore
                    "f1_score": round(float(f1_per_class[i]), 4),  # type: ignore
                    "mcc": round(float(mcc_per_class[i]), 4),  # 🌟 Per-Class OvR MCC
                }
                for i, name in enumerate(self.class_names)
            },
        }
        return metrics, cm

    def save_best_results_json(
        self,
        class_counts=None,
        initial_class_weights=None,
        best_mcc_epoch=-1,
        best_val_mcc=-1.0,
        best_loss_epoch=-1,
        best_val_loss=float("inf"),
        mcc_eval_data=None,  # Tuple: (train_eval, val_eval, test_eval)
        loss_eval_data=None, # Tuple: (train_eval, val_eval, test_eval)
        filename="best_model_metrics.json",
    ):
        """Menyimpan snapshot performa terbaik ke format JSON dengan skema yang

        100% identik dengan output evaluate_best_checkpoints.
        """

        def format_eval_res(eval_tuple):
            if eval_tuple is None or eval_tuple[0] is None:
                return None
            metrics, cm = self.compute_metrics(*eval_tuple)
            return {
                "confusion_matrix": cm.tolist() if hasattr(cm, "tolist") else cm,
                "global_metrics": metrics["global_metrics"],
                "per_class_metrics": metrics["per_class_metrics"],
            }

        # 1. Ekstrak Hasil Evaluasi Model Best Val MCC
        tr_mcc_res, va_mcc_res, te_mcc_res = None, None, None
        if mcc_eval_data is not None:
            tr_mcc_res = format_eval_res(mcc_eval_data[0])
            va_mcc_res = format_eval_res(mcc_eval_data[1])
            te_mcc_res = format_eval_res(mcc_eval_data[2])

        # 2. Ekstrak Hasil Evaluasi Model Best Val Loss
        if loss_eval_data is not None:
            tr_loss_res = format_eval_res(loss_eval_data[0])
            va_loss_res = format_eval_res(loss_eval_data[1])
            te_loss_res = format_eval_res(loss_eval_data[2])
        else:
            # Fallback jika same_best_epoch atau loss_eval_data tidak diisi
            tr_loss_res, va_loss_res, te_loss_res = (
                tr_mcc_res,
                va_mcc_res,
                te_mcc_res,
            )

        # 3. Format Array Dataset Info
        cw_list = []
        if initial_class_weights is not None:
            cw_list = (
                initial_class_weights.cpu().tolist()
                if hasattr(initial_class_weights, "cpu")
                else list(initial_class_weights)
            )

        cc_list = []
        if class_counts is not None:
            cc_list = (
                class_counts.tolist()
                if hasattr(class_counts, "tolist")
                else list(class_counts)
            )

        # 4. Susun Struktur JSON
        summary_json = {
            "dataset_info": {
                "num_classes": len(self.class_names),
                "class_labels": self.class_names,
                "train_class_counts": cc_list,
                "initial_class_weights": cw_list,
            },
            "hyperparameters": self.hparams,
            "model_by_val_mcc": {
                "criterion": "best_val_mcc",
                "best_epoch": int(best_mcc_epoch),
                "best_metric_value": round(float(best_val_mcc), 4),
                "train_results": tr_mcc_res,
                "val_results": va_mcc_res,
                "test_results": te_mcc_res,
            },
            "model_by_val_loss": {
                "criterion": "best_val_loss",
                "best_epoch": int(best_loss_epoch),
                "best_metric_value": round(float(best_val_loss), 4),
                "train_results": tr_loss_res,
                "val_results": va_loss_res,
                "test_results": te_loss_res,
            },
        }

        json_path = os.path.join(self.save_dir, filename)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_json, f, indent=4, default=str)

        print(
            f"📄 Snapshot JSON (Identik evaluate_best_checkpoints) disimpan di: {json_path}"
        )
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


    def plot_learning_curves(
        self,
        test_eval=None,
        filename="training_dashboard.png",
        save_individual=True,
    ):
        """Visualisasi grafik pelatihan:

        1 Dashboard Gabungan (6-in-1 / 8-in-1) + File Grafik Terpisah (Total 7/8
        PNG).
        """
        if not self.history["epoch"]:
            print("⚠️ Belum ada riwayat epoch untuk digambar grafiknya.")
            return
        sns.set_theme(style="whitegrid")

        epochs = self.history["epoch"]
        best_idx = (
            int(np.argmax(self.history["val_mcc"]))
            if self.history["val_mcc"]
            else 0
        )
        best_epoch = epochs[best_idx] if epochs else 1
        best_mcc_val = (
            self.history["val_mcc"][best_idx] if self.history["val_mcc"] else 0.0
        )
        patience_limit = self.hparams.get("early_stop_patience", 10)

        # Deteksi keberadaan class weights dinamis (prefix 'cw_')
        cw_keys = [k for k in self.history.keys() if k.startswith("cw_")] # type: ignore
        has_dynamic_cw = (
            len(cw_keys) > 0 and len(self.history[cw_keys[0]]) == len(epochs) # type: ignore
        )

        # String ringkasan performa untuk Panel Summary
        summary_text = (
            f"   === TRAINING STATUS SUMMARY ===\n\n"
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
                f"   === FINAL TEST BENCHMARK ===\n\n"
                f" • Test Accuracy  : {g['accuracy']:.4f}\n"
                f" • Test F1 Macro   : {g['f1_score_macro']:.4f}\n"
                f" • Test MCC        : {g['mcc']:.4f}\n"
                f" • Test ROC-AUC    : {g['auc_roc_ovr']:.4f}\n"
                f" • Cohen's Kappa   : {g['cohens_kappa']:.4f}"
            )

        # -------------------------------------------------------------
        # 1. RENDER DASHBOARD GABUNGAN (File ke-1)
        # -------------------------------------------------------------
        if has_dynamic_cw:
            fig, axes = plt.subplots(2, 4, figsize=(24, 10))
            cw_colors = sns.color_palette("Set2", len(cw_keys))
        else:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Panel 1: Loss Dynamics
        axes[0, 0].plot(
            epochs,
            self.history["train_loss"],
            label="Train Loss",
            color="crimson",
            linewidth=2,
        )
        axes[0, 0].plot(
            epochs,
            self.history["val_loss"],
            label="Val Loss",
            color="navy",
            linestyle="--",
            linewidth=2,
        )
        axes[0, 0].axvline(
            x=best_epoch,
            color="gold",
            linestyle=":",
            linewidth=2,
            label=f"Best Epoch ({best_epoch})",
        )
        axes[0, 0].set_title("Loss Dynamics", fontweight="bold")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()

        # Panel 2: Accuracy Progression
        axes[0, 1].plot(
            epochs,
            self.history["train_acc"],
            label="Train Acc",
            color="forestgreen",
            linewidth=2,
        )
        axes[0, 1].plot(
            epochs,
            self.history["val_acc"],
            label="Val Acc",
            color="darkorange",
            linestyle="--",
            linewidth=2,
        )
        axes[0, 1].axvline(
            x=best_epoch,
            color="gold",
            linestyle=":",
            linewidth=2,
            label=f"Best Epoch ({best_epoch})",
        )
        axes[0, 1].set_title("Accuracy Progression", fontweight="bold")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].legend()

        # Panel 3: Val Metrics (F1 vs MCC)
        axes[0, 2].plot(
            epochs,
            self.history["val_f1"],
            label="Val F1 (Macro)",
            color="teal",
            linewidth=2,
        )
        axes[0, 2].plot(
            epochs,
            self.history["val_mcc"],
            label="Val MCC",
            color="purple",
            linewidth=2,
        )
        axes[0, 2].scatter(
            [best_epoch],
            [best_mcc_val],
            color="gold",
            s=120,
            zorder=5,
            edgecolors="black",
            label=f"Peak MCC: {best_mcc_val:.4f}",
        )
        axes[0, 2].axvline(x=best_epoch, color="gold", linestyle=":", linewidth=2)
        axes[0, 2].set_title("Validation Quality (F1 vs MCC)", fontweight="bold")
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylabel("Score")
        axes[0, 2].legend()

        if has_dynamic_cw:
            # Panel 4 (Mode Dynamic): Dynamic Class Weights Trajectory
            for i, k in enumerate(cw_keys):
                cls_label = k.replace("cw_", "") # type: ignore
                axes[0, 3].plot(
                    epochs,
                    self.history[k], # type: ignore
                    label=f"Weight {cls_label}",
                    color=cw_colors[i],
                    linewidth=2,
                )
            axes[0, 3].axvline(
                x=best_epoch,
                color="gold",
                linestyle=":",
                linewidth=2,
                label=f"Best Epoch ({best_epoch})",
            )
            axes[0, 3].set_title(
                "Dynamic Class Weights Trajectory", fontweight="bold"
            )
            axes[0, 3].set_xlabel("Epoch")
            axes[0, 3].set_ylabel("Weight Value")
            axes[0, 3].legend()

            ax_lr, ax_pat, ax_sum = axes[1, 0], axes[1, 1], axes[1, 2]
            axes[1, 3].axis("off")
        else:
            ax_lr, ax_pat, ax_sum = axes[1, 0], axes[1, 1], axes[1, 2]

        # Panel LR Schedule
        ax_lr.plot(
            epochs,
            self.history["lr"],
            label="Effective LR",
            color="darkmagenta",
            linewidth=2,
        )
        ax_lr.set_title("Learning Rate Schedule", fontweight="bold")
        ax_lr.set_xlabel("Epoch")
        ax_lr.set_ylabel("Learning Rate")
        ax_lr.ticklabel_format(style="scientific", scilimits=(0, 0), axis="y")
        ax_lr.legend()

        # Panel Patience Tracker
        ax_pat.plot(
            epochs,
            self.history["patience"],
            label="Patience Count",
            color="orangered",
            linewidth=2,
            marker="o",
            markersize=4,
        )
        ax_pat.axhline(
            y=patience_limit,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Patience Limit ({patience_limit})",
        )
        ax_pat.set_title(
            "Early Stopping Patience Tracker", fontweight="bold"
        )  # Restored Title
        ax_pat.set_xlabel("Epoch")
        ax_pat.set_ylabel("Stagnant Epochs")
        ax_pat.legend()

        # Panel Summary Text Card
        ax_sum.axis("off")
        ax_sum.text(
            0.05,
            0.95,
            summary_text,
            transform=ax_sum.transAxes,
            fontsize=10.5,  # Restored font size 10.5
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.8",
                facecolor="whitesmoke",
                alpha=0.9,
                edgecolor="lightgray",
            ),
        )

        plt.tight_layout()
        dashboard_path = os.path.join(self.save_dir, filename)
        plt.savefig(dashboard_path, dpi=300)
        plt.close()

        dash_mode_str = "8-in-1" if has_dynamic_cw else "6-in-1"
        dash_count_str = "1/8" if has_dynamic_cw else "1/7"
        print(
            f"📊 [{dash_count_str}] Dashboard Gabungan {dash_mode_str} disimpan di: {dashboard_path}"
        )

        # -------------------------------------------------------------
        # 2. EKSPOR GRAFIK TERPISAH (File ke-2 dst)
        # -------------------------------------------------------------
        if save_individual:
            plots_dir = os.path.join(self.save_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)

            # File 2: Loss Curve
            plt.figure(figsize=(7, 5))
            plt.plot(
                epochs,
                self.history["train_loss"],
                label="Train Loss",
                color="crimson",
                linewidth=2,
            )
            plt.plot(
                epochs,
                self.history["val_loss"],
                label="Val Loss",
                color="navy",
                linestyle="--",
                linewidth=2,
            )
            plt.axvline(
                x=best_epoch,
                color="gold",
                linestyle=":",
                linewidth=2,
                label=f"Best Epoch ({best_epoch})",
            )
            plt.title("Loss Dynamics per Epoch", fontweight="bold")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "loss_curve.png"), dpi=300)
            plt.close()

            # File 3: Accuracy Curve
            plt.figure(figsize=(7, 5))
            plt.plot(
                epochs,
                self.history["train_acc"],
                label="Train Acc",
                color="forestgreen",
                linewidth=2,
            )
            plt.plot(
                epochs,
                self.history["val_acc"],
                label="Val Acc",
                color="darkorange",
                linestyle="--",
                linewidth=2,
            )
            plt.axvline(
                x=best_epoch,
                color="gold",
                linestyle=":",
                linewidth=2,
                label=f"Best Epoch ({best_epoch})",
            )
            plt.title("Accuracy Progression per Epoch", fontweight="bold")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "accuracy_curve.png"), dpi=300)
            plt.close()

            # File 4: Val Metrics (F1 vs MCC)
            plt.figure(figsize=(7, 5))
            plt.plot(
                epochs,
                self.history["val_f1"],
                label="Val F1 (Macro)",
                color="teal",
                linewidth=2,
            )
            plt.plot(
                epochs,
                self.history["val_mcc"],
                label="Val MCC",
                color="purple",
                linewidth=2,
            )
            plt.scatter(
                [best_epoch],
                [best_mcc_val],
                color="gold",
                s=120,
                zorder=5,
                edgecolors="black",
                label=f"Peak MCC: {best_mcc_val:.4f}",
            )
            plt.axvline(x=best_epoch, color="gold", linestyle=":", linewidth=2)
            plt.title("Validation Quality (F1 vs MCC)", fontweight="bold")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(plots_dir, "val_metrics_mcc_f1.png"), dpi=300
            )
            plt.close()

            # File 5: Learning Rate Schedule
            plt.figure(figsize=(7, 5))
            plt.plot(
                epochs,
                self.history["lr"],
                label="Effective LR",
                color="darkmagenta",
                linewidth=2,
            )
            plt.title("Learning Rate Schedule per Epoch", fontweight="bold")
            plt.xlabel("Epoch")
            plt.ylabel("Learning Rate")
            plt.ticklabel_format(style="scientific", scilimits=(0, 0), axis="y")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "lr_schedule.png"), dpi=300)
            plt.close()

            # File 6: Patience Tracker
            plt.figure(figsize=(7, 5))
            plt.plot(
                epochs,
                self.history["patience"],
                label="Patience Count",
                color="orangered",
                linewidth=2,
                marker="o",
                markersize=4,
            )
            plt.axhline(
                y=patience_limit,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Patience Limit ({patience_limit})",
            )
            plt.title("Early Stopping Patience Tracker", fontweight="bold")
            plt.xlabel("Epoch")
            plt.ylabel("Stagnant Epochs")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "patience_tracker.png"), dpi=300)
            plt.close()

            # File 7: Performance & Test Benchmark Summary Card
            plt.figure(figsize=(7, 5))
            plt.axis("off")
            plt.text(
                0.05,
                0.95,
                summary_text,
                transform=plt.gca().transAxes,
                fontsize=10.5,  # Restored font size 10.5
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(
                    boxstyle="round,pad=0.8",
                    facecolor="whitesmoke",
                    alpha=0.9,
                    edgecolor="lightgray",
                ),
            )
            plt.title("Performance & Benchmark Summary", fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "summary_card.png"), dpi=300)
            plt.close()

            # File 8 (Opsional, Hanya jika Dynamic MCC Aktif): Dynamic Class Weights Curve
            if has_dynamic_cw:
                plt.figure(figsize=(7, 5))
                for i, k in enumerate(cw_keys):
                    cls_label = k.replace("cw_", "") # type: ignore
                    plt.plot(
                        epochs,
                        self.history[k], # type: ignore
                        label=f"Weight {cls_label}",
                        color=cw_colors[i],
                        linewidth=2,
                    )
                plt.axvline(
                    x=best_epoch,
                    color="gold",
                    linestyle=":",
                    linewidth=2,
                    label=f"Best Epoch ({best_epoch})",
                )
                plt.title(
                    "Dynamic Class Weights Trajectory per Epoch", fontweight="bold"
                )
                plt.xlabel("Epoch")
                plt.ylabel("Weight Value")
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    os.path.join(plots_dir, "class_weights_curve.png"), dpi=300
                )
                plt.close()

            files_range_str = "2-8/8" if has_dynamic_cw else "2-7/7"
            files_count_str = (
                "7 Grafik terpisah" if has_dynamic_cw else "6 Grafik terpisah"
            )
            print(
                f"📁 [{files_range_str}] {files_count_str} berhasil disimpan di folder: {plots_dir}"
            )
        
if __name__ == "__main__":
    import numpy as np

    print("🧪 Memulai Internal Sanity Check untuk ExperimentLogger...")

    code_experiment = "exp_sanity_check"
    epochs = 10
    class_names = ["Bleeding", "Ischemia", "Normal"]
    dummy_hparams = {
        "max_epochs": epochs,
        "early_stop_patience": 5,
        "learning_rate": 1e-4,
        "use_class_weights": True,
        "weight_mode": "d-b-mcc",
    }

    # Inisialisasi Logger
    logger = ExperimentLogger(
        save_dir=f"./{code_experiment}",
        class_names=class_names,
        hparams=dummy_hparams,
    )

    # 1. Simulasi Logging Training Loop Per-Epoch
    np.random.seed(42)
    for epoch in range(1, epochs + 1):
        tr_loss = 1.0 / epoch + np.random.uniform(0.01, 0.05)
        va_loss = 1.2 / epoch + np.random.uniform(0.02, 0.08)
        tr_acc = 0.5 + 0.04 * epoch
        va_acc = 0.48 + 0.038 * epoch
        val_f1 = 0.45 + 0.035 * epoch
        val_mcc = 0.40 + 0.035 * epoch
        lr = 1e-4 * (0.95**epoch)
        patience = 0 if epoch in [7, 8] else (epoch % 3)

        # Simulasi Vektor Class Weights Dinamis per Epoch
        dummy_cw = np.array(
            [1.2 - 0.02 * epoch, 1.5 - 0.03 * epoch, 0.7 + 0.01 * epoch]
        )

        logger.log_epoch(
            epoch,
            tr_loss,
            va_loss,
            tr_acc,
            va_acc,
            val_f1,
            val_mcc,
            lr=lr,
            patience=patience,
            class_weights=dummy_cw,
        )

    # Ekspor Riwayat ke CSV
    logger.export_csv()

    # 2. Generator Data Evaluasi Mentah Dummy (y_true, y_pred, y_probs)
    def generate_dummy_eval_data(n_samples=500):
        y_true = np.random.choice([0, 1, 2], size=n_samples, p=[0.2, 0.2, 0.6])
        raw_logits = np.random.randn(n_samples, 3)
        for i in range(n_samples):
            raw_logits[i, y_true[i]] += np.random.uniform(1.5, 3.0)

        exp_logits = np.exp(
            raw_logits - np.max(raw_logits, axis=1, keepdims=True)
        )
        y_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        y_pred = np.argmax(y_probs, axis=1)

        return y_true, y_pred, y_probs

    # Model 1 (Best MCC - misal Epoch 8)
    mcc_eval_data = (
        generate_dummy_eval_data(1000),  # Train
        generate_dummy_eval_data(200),  # Val
        generate_dummy_eval_data(200),  # Test
    )

    # Model 2 (Best Loss - misal Epoch 7)
    loss_eval_data = (
        generate_dummy_eval_data(1000),  # Train
        generate_dummy_eval_data(200),  # Val
        generate_dummy_eval_data(200),  # Test
    )

    dummy_class_counts = np.array([200, 200, 600])
    dummy_initial_weights = np.array([1.5, 1.5, 0.5])

    # Plot Dashboard Utama dengan Test Set
    logger.plot_learning_curves(test_eval=mcc_eval_data[2])

    # 3. Simpan JSON Performa Terbaik (Skema Identik dengan evaluate_best_checkpoints)
    summary_json = logger.save_best_results_json(
        class_counts=dummy_class_counts,
        initial_class_weights=dummy_initial_weights,
        best_mcc_epoch=8,
        best_val_mcc=0.7250,
        best_loss_epoch=7,
        best_val_loss=0.1850,
        mcc_eval_data=mcc_eval_data,
        loss_eval_data=loss_eval_data,
        filename="best_model_metrics.json",
    )

    # Plot Confusion Matrix Masing-masing Checkpoint
    _, cm_mcc = logger.compute_metrics(*mcc_eval_data[2])
    _, cm_loss = logger.compute_metrics(*loss_eval_data[2])

    logger.plot_confusion_matrix(
        cm_mcc, filename="test_confusion_matrix_by_mcc.png"
    )
    logger.plot_confusion_matrix(
        cm_loss, filename="test_confusion_matrix_by_loss.png"
    )

    print("\n✨ Sanity Check Selesai! Seluruh struktur file & JSON valid.")