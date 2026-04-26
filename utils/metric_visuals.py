import matplotlib.pyplot as plt
import torch
import numpy as np
import os
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
        plt.show() # 🌟 新增：在 Kaggle 单元格直接显示
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
        plt.show() # 🌟 新增：在 Kaggle 单元格直接显示
        plt.close()
