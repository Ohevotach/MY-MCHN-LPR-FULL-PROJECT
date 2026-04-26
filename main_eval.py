import argparse
import os

import cv2
import torch
from torch.utils.data import DataLoader

from dataset.lp_dataset import PollutedCharDataset, TemplateLoader, build_class_memory
from models.mchn import ModernHopfieldNetwork
from utils.image_processing import LPRPipeline
from utils.metric_visuals import MetricVisualizer


def memory_indices_to_classes(memory_indices, memory_labels, device):
    classes = memory_labels[memory_indices.detach().cpu()].to(device)
    return classes


def aggregate_attention_by_class(attention_weights, template_labels, num_classes):
    batch_size = attention_weights.shape[0]
    class_scores = attention_weights.new_zeros((batch_size, num_classes))
    label_index = template_labels.to(attention_weights.device).view(1, -1).expand(batch_size, -1)
    class_scores.scatter_add_(1, label_index, attention_weights)
    return class_scores


def build_position_masks(loader, memory_labels, device):
    chinese_mask = torch.zeros(len(memory_labels), dtype=torch.bool, device=device)
    alnum_mask = torch.zeros(len(memory_labels), dtype=torch.bool, device=device)
    for i, class_idx in enumerate(memory_labels.tolist()):
        label = loader.idx_to_label[int(class_idx)]
        if len(label) == 1 and "\u4e00" <= label <= "\u9fff":
            chinese_mask[i] = True
        else:
            alnum_mask[i] = True

    return chinese_mask, alnum_mask


def run_reconstruction_demo(mchn, dataset, visualizer, device, class_memory, batch_size=5):
    print("\n" + "=" * 50)
    print("Task 1: generating MCHN reconstruction demo...")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    polluted_q, clean_m, labels_idx = next(iter(dataloader))
    polluted_q = polluted_q.to(device)

    with torch.no_grad():
        _, _, weights = mchn(polluted_q, return_attention=True)
        template_labels = dataset.L.to(device)
        class_scores = aggregate_attention_by_class(weights, template_labels, len(dataset.idx_to_label))
        pred_classes = torch.argmax(class_scores, dim=-1).detach().cpu()
        reconstructed_z = class_memory[pred_classes].detach().cpu()

    labels_str = [dataset.idx_to_label[int(idx)] for idx in labels_idx]

    visualizer.plot_reconstruction_grid(
        clean_qs=clean_m,
        polluted_qs=polluted_q,
        reconstructed_zs=reconstructed_z,
        labels=labels_str,
        filename="mchn_reconstruction_demo.png",
    )


def evaluate_one_setting(model, template_labels, num_classes, dataloader, device):
    correct = 0
    total = 0
    template_labels = template_labels.to(device)

    with torch.no_grad():
        for q, _, labels in dataloader:
            q = q.to(device)
            labels = labels.to(device)
            _, _, weights = model(q, return_attention=True)
            class_scores = aggregate_attention_by_class(weights, template_labels, num_classes)
            pred_classes = torch.argmax(class_scores, dim=-1)
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
    template_labels = loader.labels
    num_classes = len(loader.idx_to_label)
    mchn = ModernHopfieldNetwork(memory, beta=80.0, metric="dot", normalize=True, feature_mode="centered").to(device)
    baseline = ModernHopfieldNetwork(memory, beta=10.0, metric="euclidean", normalize=False, feature_mode="raw").to(device)

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

        acc_mchn.append(evaluate_one_setting(mchn, template_labels, num_classes, test_loader, device))
        acc_baseline.append(evaluate_one_setting(baseline, template_labels, num_classes, test_loader, device))
        print(f"  MCHN centered-dot: {acc_mchn[-1]:.2f}% | Euclidean baseline: {acc_baseline[-1]:.2f}%")

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
    memory, memory_labels = build_class_memory(loader, reduce="medoid")
    mchn = ModernHopfieldNetwork(memory.to(device), beta=80.0, metric="dot", normalize=True, feature_mode="centered").to(device)
    chinese_mask, alnum_mask = build_position_masks(loader, memory_labels, device)

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

            memory_idx = pred_idx.item()
            class_idx = int(memory_labels[memory_idx])
            plate_result += loader.idx_to_label[class_idx]
            confidences.append(weights[0, memory_idx].item())

        conf_text = ", ".join(f"{c:.3f}" for c in confidences)
        print(f"  MCHN result: {plate_result} | chars={len(chars_img_list)} | attention={conf_text}")


def parse_args():
    parser = argparse.ArgumentParser(description="Modern Hopfield polluted license-plate character demo")
    parser.add_argument("--pollution", default="mixed", choices=["mixed", "mask", "noise", "salt_pepper", "blur", "fog", "dirt", "affine", "none"])
    parser.add_argument("--samples-per-level", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--skip-e2e", action="store_true")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Modern Hopfield polluted LPR evaluation, device={device}")

    os.makedirs(args.output_dir, exist_ok=True)
    visualizer = MetricVisualizer(save_dir=args.output_dir)

    loader = TemplateLoader(
        data_roots=[os.path.join(args.data_dir, "chars2"), os.path.join(args.data_dir, "charsChinese")],
        img_size=(32, 64),
        cache_path=os.path.join(args.data_dir, "template_cache_32x64.pt"),
    )
    if loader.memory_matrix.shape[0] == 0:
        raise RuntimeError("Template memory is empty. Please check ./data/chars2 and ./data/charsChinese.")

    demo_dataset = PollutedCharDataset(
        loader,
        virtual_size=10,
        pollution_type=args.pollution,
        severity=0.6,
        seed=2026,
    )
    class_memory, _ = build_class_memory(loader, reduce="medoid")
    demo_model = ModernHopfieldNetwork(
        loader.memory_matrix.to(device),
        beta=80.0,
        metric="dot",
        normalize=True,
        feature_mode="centered",
    ).to(device)

    run_reconstruction_demo(demo_model, demo_dataset, visualizer, device, class_memory)
    run_robustness_evaluation(
        loader,
        visualizer,
        device,
        pollution_type=args.pollution,
        samples_per_level=args.samples_per_level,
        batch_size=args.batch_size,
    )

    if not args.skip_e2e:
        run_end_to_end_system(loader, device, test_dir=os.path.join(args.data_dir, "full_cars", "ccpd_weather"))

    print(f"\nAll tasks finished. Figures are saved in {args.output_dir}.")
