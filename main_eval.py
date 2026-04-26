import argparse
import os

import cv2
import torch
from torch.utils.data import DataLoader

from dataset.lp_dataset import PollutedCharDataset, TemplateLoader
from models.mchn import ModernHopfieldNetwork
from utils.image_processing import LPRPipeline
from utils.metric_visuals import MetricVisualizer


def template_indices_to_classes(template_indices, template_labels, device):
    classes = template_labels[template_indices.detach().cpu()].to(device)
    return classes


def build_position_masks(loader, num_templates, device):
    chinese_mask = torch.zeros(num_templates, dtype=torch.bool, device=device)
    if loader.chinese_indices:
        chinese_mask[loader.chinese_indices] = True

    alnum_mask = torch.zeros(num_templates, dtype=torch.bool, device=device)
    if loader.alnum_indices:
        alnum_mask[loader.alnum_indices] = True

    return chinese_mask, alnum_mask


def run_reconstruction_demo(mchn, dataset, visualizer, device, batch_size=5):
    print("\n" + "=" * 50)
    print("Task 1: generating MCHN reconstruction demo...")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    polluted_q, clean_m, labels_idx = next(iter(dataloader))
    polluted_q = polluted_q.to(device)

    with torch.no_grad():
        reconstructed_z, _ = mchn(polluted_q)

    labels_str = [dataset.idx_to_label[int(idx)] for idx in labels_idx]

    visualizer.plot_reconstruction_grid(
        clean_qs=clean_m,
        polluted_qs=polluted_q,
        reconstructed_zs=reconstructed_z,
        labels=labels_str,
        filename="mchn_reconstruction_demo.png",
    )


def evaluate_one_setting(model, loader, dataloader, device):
    correct = 0
    total = 0
    template_labels = loader.labels.to(device)

    with torch.no_grad():
        for q, _, labels in dataloader:
            q = q.to(device)
            labels = labels.to(device)
            _, pred_template_indices = model(q)
            pred_classes = template_indices_to_classes(pred_template_indices, template_labels, device)
            correct += (pred_classes == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / max(total, 1)


def run_robustness_evaluation(loader, visualizer, device, pollution_type, samples_per_level, batch_size):
    print("\n" + "=" * 50)
    print(f"Task 2: robustness evaluation, pollution={pollution_type}...")

    severities = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]
    acc_mchn = []
    acc_baseline = []

    memory = loader.memory_matrix.to(device)
    mchn = ModernHopfieldNetwork(memory, beta=25.0, metric="dot", normalize=True).to(device)
    baseline = ModernHopfieldNetwork(memory, beta=10.0, metric="euclidean", normalize=False).to(device)

    for sev in severities:
        print(f"Testing severity={sev:.1f} ...")
        test_dataset = PollutedCharDataset(
            loader,
            virtual_size=samples_per_level,
            pollution_type=pollution_type,
            severity=sev,
            seed=1234,
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        acc_mchn.append(evaluate_one_setting(mchn, loader, test_loader, device))
        acc_baseline.append(evaluate_one_setting(baseline, loader, test_loader, device))
        print(f"  MCHN-dot: {acc_mchn[-1]:.2f}% | Euclidean baseline: {acc_baseline[-1]:.2f}%")

    visualizer.plot_robustness_curve(
        severities,
        acc_mchn,
        baseline_acc=acc_baseline,
        pollution_type=pollution_type,
        filename=f"robustness_{pollution_type}_curve.png",
    )


def run_end_to_end_system(loader, device, test_dir="./data/full_cars/ccpd_weather", max_images=3):
    print("\n" + "=" * 50)
    print("Task 3: end-to-end plate demo...")

    if not os.path.exists(test_dir) or len(os.listdir(test_dir)) == 0:
        print(f"Warning: test image directory not found or empty: {test_dir}")
        return

    pipeline = LPRPipeline(use_synthetic_pollution=False)
    mchn = ModernHopfieldNetwork(loader.memory_matrix.to(device), beta=25.0, metric="dot", normalize=True).to(device)
    chinese_mask, alnum_mask = build_position_masks(loader, mchn.num_templates, device)

    test_files = [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ][:max_images]

    for img_path in test_files:
        print(f"\nProcessing image: {os.path.basename(img_path)}")
        plate_img, chars_img_list = pipeline.process_image(img_path)

        if plate_img is None or len(chars_img_list) == 0:
            print("  OpenCV plate location or character segmentation failed.")
            continue

        plate_result = ""
        confidences = []
        for i, char_img in enumerate(chars_img_list):
            char_resized = cv2.resize(char_img, (32, 64))
            char_tensor = torch.tensor(char_resized, dtype=torch.float32, device=device) / 255.0
            char_tensor = char_tensor.view(1, -1)

            current_mask = chinese_mask if i == 0 else alnum_mask
            with torch.no_grad():
                _, pred_idx, weights = mchn(char_tensor, template_mask=current_mask, return_attention=True)

            template_idx = pred_idx.item()
            class_idx = int(loader.labels[template_idx])
            plate_result += loader.idx_to_label[class_idx]
            confidences.append(weights[0, template_idx].item())

        conf_text = ", ".join(f"{c:.3f}" for c in confidences)
        print(f"  MCHN result: {plate_result} | chars={len(chars_img_list)} | attention={conf_text}")


def parse_args():
    parser = argparse.ArgumentParser(description="Modern Hopfield polluted license-plate character demo")
    parser.add_argument("--pollution", default="mixed", choices=["mixed", "mask", "noise", "salt_pepper", "blur", "fog", "dirt", "affine", "none"])
    parser.add_argument("--samples-per-level", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--skip-e2e", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Modern Hopfield polluted LPR evaluation, device={device}")

    os.makedirs("./results", exist_ok=True)
    visualizer = MetricVisualizer(save_dir="./results")

    loader = TemplateLoader(data_roots=["./data/chars2", "./data/charsChinese"], img_size=(32, 64))
    if loader.memory_matrix.shape[0] == 0:
        raise RuntimeError("Template memory is empty. Please check ./data/chars2 and ./data/charsChinese.")

    demo_dataset = PollutedCharDataset(
        loader,
        virtual_size=10,
        pollution_type=args.pollution,
        severity=0.6,
        seed=2026,
    )
    demo_model = ModernHopfieldNetwork(
        loader.memory_matrix.to(device),
        beta=25.0,
        metric="dot",
        normalize=True,
    ).to(device)

    run_reconstruction_demo(demo_model, demo_dataset, visualizer, device)
    run_robustness_evaluation(
        loader,
        visualizer,
        device,
        pollution_type=args.pollution,
        samples_per_level=args.samples_per_level,
        batch_size=args.batch_size,
    )

    if not args.skip_e2e:
        run_end_to_end_system(loader, device)

    print("\nAll tasks finished. Figures are saved in ./results.")
