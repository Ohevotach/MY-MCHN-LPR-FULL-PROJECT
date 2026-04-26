import argparse
import csv
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


def class_free_energy_scores(sim_scores, template_labels, beta, num_classes, template_mask=None):
    labels = template_labels.to(sim_scores.device)
    scaled = beta * sim_scores
    if template_mask is not None:
        mask = template_mask.to(device=sim_scores.device, dtype=torch.bool)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        scaled = scaled.masked_fill(~mask, -1e9)

    scores = []
    for class_idx in range(num_classes):
        class_mask = labels == class_idx
        if template_mask is not None and template_mask.dim() == 1:
            class_mask = class_mask & template_mask.to(labels.device)
        count = int(class_mask.sum().item())
        if count == 0:
            scores.append(scaled.new_full((scaled.shape[0],), -1e9))
            continue
        class_score = torch.logsumexp(scaled[:, class_mask], dim=-1) - torch.log(
            scaled.new_tensor(float(count))
        )
        scores.append(class_score)
    return torch.stack(scores, dim=-1)


def select_best_template_in_class(sim_scores, template_labels, pred_classes):
    labels = template_labels.to(sim_scores.device)
    selected = []
    for row, class_idx in zip(sim_scores, pred_classes):
        mask = labels == int(class_idx)
        masked = row.masked_fill(~mask, -1e9)
        selected.append(torch.argmax(masked))
    return torch.stack(selected)


def ensemble_class_scores(models, q, template_labels, num_classes, template_mask=None):
    fused = None
    primary_sim = None
    for model in models:
        _, _, sim_scores = model(q, template_mask=template_mask, return_similarity=True)
        scores = class_free_energy_scores(
            sim_scores,
            template_labels,
            beta=model.beta,
            num_classes=num_classes,
            template_mask=template_mask,
        )
        log_probs = torch.log_softmax(scores, dim=-1)
        fused = log_probs if fused is None else fused + log_probs
        if primary_sim is None:
            primary_sim = sim_scores
    return fused, primary_sim


def build_template_masks(loader, num_templates, device):
    chinese_mask = torch.zeros(num_templates, dtype=torch.bool, device=device)
    if loader.chinese_indices:
        chinese_mask[loader.chinese_indices] = True

    alnum_mask = torch.zeros(num_templates, dtype=torch.bool, device=device)
    if loader.alnum_indices:
        alnum_mask[loader.alnum_indices] = True

    return chinese_mask, alnum_mask


def run_reconstruction_demo(models, dataset, visualizer, device, class_memory, batch_size=5):
    print("\n" + "=" * 50)
    print("Task 1: generating MCHN reconstruction demo...")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    polluted_q, clean_m, labels_idx = next(iter(dataloader))
    polluted_q = polluted_q.to(device)

    with torch.no_grad():
        template_labels = dataset.L.to(device)
        class_scores, sim_scores = ensemble_class_scores(
            models,
            polluted_q,
            template_labels,
            num_classes=len(dataset.idx_to_label),
        )
        pred_classes = torch.argmax(class_scores, dim=-1).detach().cpu()
        best_template_indices = select_best_template_in_class(sim_scores, template_labels, pred_classes.to(device))
        reconstructed_z = models[0].M[best_template_indices].detach().cpu()

    labels_str = [dataset.idx_to_label[int(idx)] for idx in labels_idx]

    visualizer.plot_reconstruction_grid(
        clean_qs=clean_m,
        polluted_qs=polluted_q,
        reconstructed_zs=reconstructed_z,
        labels=labels_str,
        filename="mchn_reconstruction_demo.png",
    )


def evaluate_one_setting(models, template_labels, num_classes, dataloader, device):
    correct = 0
    total = 0
    template_labels = template_labels.to(device)
    if not isinstance(models, (list, tuple)):
        models = [models]

    with torch.no_grad():
        for q, _, labels in dataloader:
            q = q.to(device)
            labels = labels.to(device)
            class_scores, _ = ensemble_class_scores(models, q, template_labels, num_classes)
            pred_classes = torch.argmax(class_scores, dim=-1)
            correct += (pred_classes == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / max(total, 1)


def build_model_suite(memory, device):
    return {
        "MCHN multi-view": [
            ModernHopfieldNetwork(memory, beta=60.0, metric="dot", normalize=True, feature_mode="binary").to(device),
            ModernHopfieldNetwork(memory, beta=80.0, metric="dot", normalize=True, feature_mode="centered").to(device),
        ],
        "MCHN binary": ModernHopfieldNetwork(
            memory, beta=60.0, metric="dot", normalize=True, feature_mode="binary"
        ).to(device),
        "MCHN centered": ModernHopfieldNetwork(
            memory, beta=80.0, metric="dot", normalize=True, feature_mode="centered"
        ).to(device),
        "Dot raw": ModernHopfieldNetwork(memory, beta=25.0, metric="dot", normalize=True, feature_mode="raw").to(device),
        "Euclidean": ModernHopfieldNetwork(
            memory, beta=10.0, metric="euclidean", normalize=False, feature_mode="raw"
        ).to(device),
        "Manhattan": ModernHopfieldNetwork(
            memory, beta=0.05, metric="manhattan", normalize=False, feature_mode="raw"
        ).to(device),
    }


def run_robustness_evaluation(loader, visualizer, device, pollution_type, samples_per_level, batch_size):
    print("\n" + "=" * 50)
    print(f"Task 2: robustness evaluation, pollution={pollution_type}...")

    severities = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]
    memory = loader.memory_matrix.to(device)
    template_labels = loader.labels
    num_classes = len(loader.idx_to_label)
    model_suite = build_model_suite(memory, device)
    results = {name: [] for name in model_suite}

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

        for name, model in model_suite.items():
            acc = evaluate_one_setting(model, template_labels, num_classes, test_loader, device)
            results[name].append(acc)
        print("  " + " | ".join(f"{name}: {values[-1]:.2f}%" for name, values in results.items()))

    visualizer.plot_multi_robustness_curve(
        severities,
        results,
        pollution_type=pollution_type,
        filename=f"robustness_{pollution_type}_multi_curve.png",
    )
    visualizer.plot_final_severity_bar(
        {name: values[-1] for name, values in results.items()},
        pollution_type=pollution_type,
        severity=severities[-1],
        filename=f"robustness_{pollution_type}_final_bar.png",
    )
    return severities, results


def save_results_csv(output_dir, all_results, severities):
    csv_path = os.path.join(output_dir, "robustness_all_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pollution", "model", *[f"severity_{s}" for s in severities]])
        for pollution, model_results in all_results.items():
            for model_name, values in model_results.items():
                writer.writerow([pollution, model_name, *[f"{v:.4f}" for v in values]])
    print(f"Saved CSV: {csv_path}")


def plot_all_pollution_summary(visualizer, all_results, severities):
    if not all_results:
        return
    model_names = list(next(iter(all_results.values())).keys())
    pollution_names = list(all_results.keys())
    final_matrix = []
    avg_matrix = []
    for model_name in model_names:
        final_matrix.append([all_results[p][model_name][-1] for p in pollution_names])
        avg_matrix.append([sum(all_results[p][model_name]) / len(severities) for p in pollution_names])

    visualizer.plot_summary_heatmap(
        final_matrix,
        row_labels=model_names,
        col_labels=pollution_names,
        title=f"Accuracy at severity={severities[-1]}",
        filename="summary_final_severity_heatmap.png",
    )
    visualizer.plot_summary_heatmap(
        avg_matrix,
        row_labels=model_names,
        col_labels=pollution_names,
        title="Mean accuracy across severities",
        filename="summary_mean_accuracy_heatmap.png",
    )


def run_end_to_end_system(loader, device, test_dir="./data/full_cars/ccpd_weather", max_images=3):
    print("\n" + "=" * 50)
    print("Task 3: end-to-end plate demo...")

    if not os.path.exists(test_dir) or len(os.listdir(test_dir)) == 0:
        print(f"Warning: test image directory not found or empty: {test_dir}")
        return

    pipeline = LPRPipeline(use_synthetic_pollution=False)
    mchn = ModernHopfieldNetwork(
        loader.memory_matrix.to(device),
        beta=60.0,
        metric="dot",
        normalize=True,
        feature_mode="binary",
    ).to(device)
    template_labels = loader.labels.to(device)
    chinese_mask, alnum_mask = build_template_masks(loader, mchn.num_templates, device)

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
                _, _, sim_scores = mchn(char_tensor, template_mask=current_mask, return_similarity=True)
                class_scores = class_free_energy_scores(
                    sim_scores,
                    template_labels,
                    beta=mchn.beta,
                    num_classes=len(loader.idx_to_label),
                    template_mask=current_mask,
                )

            class_idx = int(torch.argmax(class_scores, dim=-1).item())
            plate_result += loader.idx_to_label[class_idx]
            confidences.append(torch.softmax(class_scores, dim=-1)[0, class_idx].item())

        conf_text = ", ".join(f"{c:.3f}" for c in confidences)
        print(f"  MCHN result: {plate_result} | chars={len(chars_img_list)} | attention={conf_text}")


def parse_args():
    parser = argparse.ArgumentParser(description="Modern Hopfield polluted license-plate character demo")
    parser.add_argument(
        "--pollution",
        default="mixed",
        choices=["all", "mixed", "mask", "noise", "salt_pepper", "blur", "fog", "dirt", "affine", "none"],
    )
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
        cache_path=os.path.join(args.output_dir, "template_cache_32x64.pt"),
    )
    if loader.memory_matrix.shape[0] == 0:
        raise RuntimeError("Template memory is empty. Please check ./data/chars2 and ./data/charsChinese.")

    demo_dataset = PollutedCharDataset(
        loader,
        virtual_size=10,
        pollution_type=args.pollution,
        severity=0.4,
        seed=2026,
    )
    class_memory, _ = build_class_memory(loader, reduce="medoid")
    demo_models = [
        ModernHopfieldNetwork(
            loader.memory_matrix.to(device),
            beta=60.0,
            metric="dot",
            normalize=True,
            feature_mode="binary",
        ).to(device),
        ModernHopfieldNetwork(
            loader.memory_matrix.to(device),
            beta=80.0,
            metric="dot",
            normalize=True,
            feature_mode="centered",
        ).to(device),
    ]

    run_reconstruction_demo(demo_models, demo_dataset, visualizer, device, class_memory)
    pollution_types = ["mask", "noise", "salt_pepper", "blur", "fog", "dirt", "affine", "mixed"]
    if args.pollution != "all":
        pollution_types = [args.pollution]

    all_results = {}
    severities = None
    for pollution_type in pollution_types:
        severities, results = run_robustness_evaluation(
            loader,
            visualizer,
            device,
            pollution_type=pollution_type,
            samples_per_level=args.samples_per_level,
            batch_size=args.batch_size,
        )
        all_results[pollution_type] = results

    save_results_csv(args.output_dir, all_results, severities)
    if args.pollution == "all":
        plot_all_pollution_summary(visualizer, all_results, severities)

    if not args.skip_e2e:
        run_end_to_end_system(loader, device, test_dir=os.path.join(args.data_dir, "full_cars", "ccpd_weather"))

    print(f"\nAll tasks finished. Figures are saved in {args.output_dir}.")
