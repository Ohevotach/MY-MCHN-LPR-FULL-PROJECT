import torch
import numpy as np
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

class MetricVisualizer:
    def __init__(self, img_size=(32, 64), save_dir="./results"):
        self.img_w, self.img_h = img_size
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        font_path = '/kaggle/working/SimHei.ttf'
        self.zh_font = FontProperties(fname=font_path) if os.path.exists(font_path) else FontProperties()

    def _tensor_to_img(self, tensor):
        arr = tensor.detach().cpu().numpy()
        return arr.reshape((self.img_h, self.img_w))

    def plot_reconstruction_grid(self, clean_qs, polluted_qs, reconstructed_zs, labels=None, filename="mchn_reconstruction_demo.png"):
        num_samples = min(clean_qs.shape[0], 5) 
        fig, axes = plt.subplots(nrows=num_samples, ncols=3, figsize=(8, 2 * num_samples))
        plt.suptitle("MCHN Single-shot Reconstruction", fontsize=16, y=0.98)
        
        for i in range(num_samples):
            imgs = [self._tensor_to_img(clean_qs[i]), self._tensor_to_img(polluted_qs[i]), self._tensor_to_img(reconstructed_zs[i])]
            titles = ["Ground Truth", "Polluted Input", "Reconstructed"]
            for j, img in enumerate(imgs):
                ax = axes[i, j] if num_samples > 1 else axes[j]
                ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                ax.axis('off')
                if i == 0: ax.set_title(titles[j])
                if j == 0 and labels:
                    ax.text(-12, 32, str(labels[i]), fontsize=14, color='blue', va='center', fontproperties=self.zh_font)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300)
        plt.close()

    def plot_robustness_curve(self, severities, accuracies, baseline_acc=None, pollution_type="Masking", filename="robustness_curve.png"):
        plt.figure(figsize=(8, 6))
        plt.plot(severities, accuracies, marker='o', color='#e74c3c', label='MCHN (Ours)')
        if baseline_acc:
            plt.plot(severities, baseline_acc, marker='s', color='#95a5a6', linestyle='--', label='Baseline')
        
        plt.title(f"Accuracy vs. {pollution_type}", fontsize=14)
        plt.xlabel("Severity Ratio"), plt.ylabel("Accuracy (%)")
        plt.legend(), plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300)
        plt.close()

    def plot_multi_robustness_curve(self, severities, model_results, pollution_type="mixed", filename=None):
        filename = filename or f"robustness_{pollution_type}_multi_curve.png"
        plt.figure(figsize=(9, 6))
        styles = ["o-", "s--", "^--", "d--", "x--", "v--", "p--"]
        colors = ["#e74c3c", "#2c7fb8", "#7f8c8d", "#27ae60", "#8e44ad", "#d35400", "#16a085"]

        for i, (name, values) in enumerate(model_results.items()):
            style = styles[i % len(styles)]
            color = colors[i % len(colors)]
            linewidth = 2.4 if i == 0 else 1.7
            plt.plot(severities, values, style, color=color, linewidth=linewidth, label=name)

        plt.title(f"Accuracy vs. {pollution_type}", fontsize=14)
        plt.xlabel("Severity Ratio")
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 105)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300)
        plt.close()

    def plot_summary_heatmap(self, matrix, row_labels, col_labels, title, filename):
        data = np.asarray(matrix, dtype=float)
        fig_w = max(8, 1.1 * len(col_labels))
        fig_h = max(5, 0.55 * len(row_labels))
        plt.figure(figsize=(fig_w, fig_h))
        plt.imshow(data, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
        plt.colorbar(label="Accuracy (%)")
        plt.xticks(range(len(col_labels)), col_labels, rotation=35, ha="right")
        plt.yticks(range(len(row_labels)), row_labels)
        plt.title(title)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                color = "white" if data[i, j] > 65 else "black"
                plt.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=8, color=color)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300)
        plt.close()

    def plot_method_pollution_curves(self, severities, all_results, method_name, filename):
        plt.figure(figsize=(10, 6))
        styles = ["o-", "s-", "^-", "d-", "x-", "v-", "p-", "h-"]
        colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#7f7f7f"]

        plotted = 0
        for i, (pollution, method_results) in enumerate(all_results.items()):
            if method_name not in method_results:
                continue
            plt.plot(
                severities,
                method_results[method_name],
                styles[i % len(styles)],
                color=colors[i % len(colors)],
                linewidth=2.0,
                markersize=5,
                label=pollution,
            )
            plotted += 1

        plt.title(f"{method_name} accuracy under different pollutions", fontsize=14)
        plt.xlabel("Pollution severity")
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 105)
        plt.grid(True, alpha=0.3)
        if plotted:
            plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300)
        plt.close()

    def plot_final_severity_bar(self, final_scores, pollution_type, severity, filename):
        names = list(final_scores.keys())
        values = [final_scores[name] for name in names]
        colors = ["#e74c3c"] + ["#7f8c8d"] * max(0, len(names) - 1)

        plt.figure(figsize=(9, 5))
        bars = plt.bar(names, values, color=colors)
        plt.ylim(0, 105)
        plt.ylabel("Accuracy (%)")
        plt.title(f"{pollution_type} accuracy at severity={severity}")
        plt.xticks(rotation=25, ha="right")
        plt.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}", ha="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300)
        plt.close()
