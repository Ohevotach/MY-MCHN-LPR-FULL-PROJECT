import argparse
import csv
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.lp_dataset import PollutedCharDataset, TemplateLoader, build_class_memory, normalize_char_tensor
from models.mchn import ModernHopfieldNetwork
from utils.image_processing import LPRPipeline
from utils.metric_visuals import MetricVisualizer


SEVERITIES = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 1, 64, 32)
        return self.classifier(self.features(x))


def build_hopfield_ensemble(memory, device):
    return [
        ModernHopfieldNetwork(memory, beta=60.0, metric="dot", normalize=True, feature_mode="binary").to(device),
        ModernHopfieldNetwork(memory, beta=80.0, metric="dot", normalize=True, feature_mode="centered").to(device),
        ModernHopfieldNetwork(memory, beta=55.0, metric="dot", normalize=True, feature_mode="hybrid_shape").to(device),
        ModernHopfieldNetwork(memory, beta=70.0, metric="dot", normalize=True, feature_mode="profile").to(device),
        ModernHopfieldNetwork(memory, beta=7.0, metric="euclidean", normalize=False, feature_mode="profile").to(device),
    ]


def build_stratified_split(labels, train_ratio=0.7, seed=2026):
    rng = random.Random(seed)
    train_indices = []
    test_indices = []
    labels_list = labels.tolist()
    for class_idx in sorted(set(labels_list)):
        indices = [i for i, label in enumerate(labels_list) if label == class_idx]
        rng.shuffle(indices)
        if len(indices) <= 1:
            train_indices.extend(indices)
            continue
        test_count = max(1, int(round(len(indices) * (1.0 - train_ratio))))
        test_count = min(test_count, len(indices) - 1)
        test_indices.extend(indices[:test_count])
        train_indices.extend(indices[test_count:])
    return train_indices, test_indices


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
        score = torch.logsumexp(scaled[:, class_mask], dim=-1) - torch.log(scaled.new_tensor(float(count)))
        scores.append(score)
    return torch.stack(scores, dim=-1)


def ensemble_hopfield_scores(models, q, template_labels, num_classes, template_mask=None):
    fused = None
    primary_sim = None
    for model in models:
        _, _, sim_scores = model(q, template_mask=template_mask, return_similarity=True)
        scores = class_free_energy_scores(sim_scores, template_labels, beta=model.beta, num_classes=num_classes, template_mask=template_mask)
        log_probs = torch.log_softmax(scores, dim=-1)
        fused = log_probs if fused is None else fused + log_probs
        if primary_sim is None:
            primary_sim = sim_scores
    return fused, primary_sim


def predict_nearest_neighbor(q, train_memory, train_labels, metric):
    if metric == "cosine":
        qn = F.normalize(q, dim=-1)
        mn = F.normalize(train_memory, dim=-1)
        indices = torch.argmax(qn @ mn.t(), dim=-1)
    elif metric == "euclidean":
        indices = torch.argmin(torch.cdist(q, train_memory, p=2.0), dim=-1)
    else:
        raise ValueError(metric)
    return train_labels[indices]


def predict_prototype(q, prototypes, prototype_labels):
    qn = F.normalize(q - q.mean(dim=-1, keepdim=True), dim=-1)
    pn = F.normalize(prototypes - prototypes.mean(dim=-1, keepdim=True), dim=-1)
    indices = torch.argmax(qn @ pn.t(), dim=-1)
    return prototype_labels[indices]


def normalize_char_array(arr):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape != (64, 32):
        arr = cv2.resize(arr, (32, 64), interpolation=cv2.INTER_NEAREST)
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary > 0) > 0.55:
        binary = cv2.bitwise_not(binary)
    ys, xs = np.where(binary > 0)
    if len(xs) == 0 or len(ys) == 0:
        return binary
    crop = binary[max(0, ys.min() - 1) : min(64, ys.max() + 2), max(0, xs.min() - 1) : min(32, xs.max() + 2)]
    h, w = crop.shape[:2]
    canvas = np.zeros((64, 32), dtype=np.uint8)
    scale = min(26.0 / max(1, w), 56.0 / max(1, h))
    new_w = max(1, min(30, int(round(w * scale))))
    new_h = max(1, min(62, int(round(h * scale))))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    y = (64 - new_h) // 2
    x = (32 - new_w) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def build_affine_query_variants(q):
    tensors = []
    for row in q.detach().cpu().view(-1, 64, 32).numpy():
        base = normalize_char_array(row * 255.0)
        variants = [base]
        for angle in (-14, -8, 8, 14):
            matrix = cv2.getRotationMatrix2D((16, 32), angle, 1.0)
            variants.append(normalize_char_array(cv2.warpAffine(base, matrix, (32, 64), flags=cv2.INTER_NEAREST, borderValue=0)))
        for shear in (-0.16, -0.08, 0.08, 0.16):
            matrix = np.array([[1.0, shear, -shear * 32.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            variants.append(normalize_char_array(cv2.warpAffine(base, matrix, (32, 64), flags=cv2.INTER_NEAREST, borderValue=0)))
        tensors.extend(torch.tensor(item, dtype=torch.float32).view(1, -1) / 255.0 for item in variants)
    return torch.cat(tensors, dim=0), 9


def predict_affine_robust_hopfield(models, q, template_labels, num_classes):
    base_scores, _ = ensemble_hopfield_scores(models, q, template_labels, num_classes)
    base_probs = torch.softmax(base_scores, dim=-1)
    base_conf, base_pred = torch.max(base_probs, dim=-1)

    q_variants, variants_per_sample = build_affine_query_variants(q)
    q_variants = q_variants.to(q.device)
    scores, _ = ensemble_hopfield_scores(models, q_variants, template_labels, num_classes)
    scores = scores.view(q.shape[0], variants_per_sample, num_classes)
    pooled_scores = torch.logsumexp(scores, dim=1) - torch.log(scores.new_tensor(float(variants_per_sample)))
    pooled_pred = torch.argmax(pooled_scores, dim=-1)
    return torch.where(base_conf >= 0.72, base_pred, pooled_pred)


def train_cnn(loader, train_indices, num_classes, device, epochs, train_samples, batch_size, seed):
    model = SimpleCNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    train_dataset = PollutedCharDataset(
        loader,
        virtual_size=train_samples,
        pollution_type="mixed",
        severity=(0.0, 0.6),
        seed=seed,
        sample_indices=train_indices,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total = 0
        correct = 0
        for q, _, labels in train_loader:
            q = q.to(device)
            labels = labels.to(device)
            logits = model(q)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (torch.argmax(logits, dim=-1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        print(f"  CNN epoch {epoch + 1}/{epochs}: loss={total_loss / max(total, 1):.4f}, acc={100 * correct / max(total, 1):.2f}%")
    return model.eval()


def evaluate_methods(methods, test_loader, device):
    correct = {name: 0 for name in methods}
    total = 0
    with torch.no_grad():
        for q, _, labels in test_loader:
            q = q.to(device)
            labels = labels.to(device)
            for name, predictor in methods.items():
                pred = predictor(q)
                correct[name] += (pred == labels).sum().item()
            total += labels.size(0)
    return {name: 100.0 * value / max(total, 1) for name, value in correct.items()}


def select_best_template_in_class(sim_scores, template_labels, pred_classes):
    labels = template_labels.to(sim_scores.device)
    selected = []
    for row, class_idx in zip(sim_scores, pred_classes):
        mask = labels == int(class_idx)
        selected.append(torch.argmax(row.masked_fill(~mask, -1e9)))
    return torch.stack(selected)


def run_reconstruction_demo(models, loader, test_indices, visualizer, device):
    print("\n" + "=" * 50)
    print("Task 1: generating held-out reconstruction demo...")
    dataset = PollutedCharDataset(loader, virtual_size=10, pollution_type="mixed", severity=0.4, seed=2026, sample_indices=test_indices)
    q, clean, labels = next(iter(DataLoader(dataset, batch_size=5, shuffle=False)))
    q_device = q.to(device)
    train_labels = models["train_labels"].to(device)
    with torch.no_grad():
        class_scores, sim_scores = ensemble_hopfield_scores(
            models["hopfield"], q_device, train_labels, len(loader.idx_to_label)
        )
        pred_classes = torch.argmax(class_scores, dim=-1)
        best_indices = select_best_template_in_class(sim_scores, train_labels, pred_classes)
        reconstructed = models["train_memory"][best_indices].detach().cpu()

    visualizer.plot_reconstruction_grid(
        clean_qs=clean,
        polluted_qs=q,
        reconstructed_zs=reconstructed,
        labels=[loader.idx_to_label[int(label)] for label in labels],
        filename="mchn_reconstruction_demo.png",
    )


def run_robustness_evaluation(
    loader,
    visualizer,
    device,
    pollution_type,
    samples_per_level,
    batch_size,
    train_indices,
    test_indices,
    trained_cnn,
    seed,
):
    print("\n" + "=" * 50)
    print(f"Task 2: held-out evaluation, pollution={pollution_type}...")
    train_memory = loader.memory_matrix[train_indices].to(device)
    train_labels = loader.labels[train_indices].to(device)
    prototypes, prototype_labels = build_class_memory_from_tensors(train_memory, train_labels)
    num_classes = len(loader.idx_to_label)

    hopfield_models = build_hopfield_ensemble(train_memory, device)

    def make_methods():
        return {
            "Modern Hopfield": lambda q: torch.argmax(
                ensemble_hopfield_scores(hopfield_models, q, train_labels, num_classes)[0], dim=-1
            ),
            "Affine-robust Hopfield": lambda q: predict_affine_robust_hopfield(hopfield_models, q, train_labels, num_classes),
            "CNN": lambda q: torch.argmax(trained_cnn(q), dim=-1),
            "Nearest Neighbor": lambda q: predict_nearest_neighbor(q, train_memory, train_labels, metric="cosine"),
            "Euclidean NN": lambda q: predict_nearest_neighbor(q, train_memory, train_labels, metric="euclidean"),
            "Class Prototype": lambda q: predict_prototype(q, prototypes, prototype_labels),
        }

    results = {name: [] for name in make_methods()}
    for severity in SEVERITIES:
        test_dataset = PollutedCharDataset(
            loader,
            virtual_size=samples_per_level,
            pollution_type=pollution_type,
            severity=severity,
            seed=seed,
            sample_indices=test_indices,
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        accs = evaluate_methods(make_methods(), test_loader, device)
        for name, acc in accs.items():
            results[name].append(acc)
        print("  severity=" + f"{severity:.1f}: " + " | ".join(f"{k}: {v:.2f}%" for k, v in accs.items()))

    visualizer.plot_multi_robustness_curve(
        SEVERITIES,
        results,
        pollution_type=pollution_type,
        filename=f"robustness_{pollution_type}_methods_curve.png",
    )
    visualizer.plot_final_severity_bar(
        {name: values[-1] for name, values in results.items()},
        pollution_type=pollution_type,
        severity=SEVERITIES[-1],
        filename=f"robustness_{pollution_type}_methods_final_bar.png",
    )
    return results


def build_class_memory_from_tensors(memory, labels):
    vectors = []
    class_labels = []
    for class_idx in sorted(labels.unique().tolist()):
        samples = memory[labels == class_idx]
        vectors.append(samples.mean(dim=0))
        class_labels.append(int(class_idx))
    return torch.stack(vectors), torch.tensor(class_labels, dtype=torch.long, device=memory.device)


def save_results_csv(output_dir, all_results):
    csv_path = os.path.join(output_dir, "robustness_all_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pollution", "method", *[f"severity_{s}" for s in SEVERITIES]])
        for pollution, method_results in all_results.items():
            for method, values in method_results.items():
                writer.writerow([pollution, method, *[f"{value:.4f}" for value in values]])
    print(f"Saved CSV: {csv_path}")


def plot_all_pollution_summary(visualizer, all_results):
    method_names = list(next(iter(all_results.values())).keys())
    pollution_names = list(all_results.keys())
    final_matrix = []
    mean_matrix = []
    for method in method_names:
        final_matrix.append([all_results[p][method][-1] for p in pollution_names])
        mean_matrix.append([sum(all_results[p][method]) / len(SEVERITIES) for p in pollution_names])

    visualizer.plot_summary_heatmap(
        final_matrix,
        row_labels=method_names,
        col_labels=pollution_names,
        title=f"Held-out accuracy at severity={SEVERITIES[-1]}",
        filename="summary_final_severity_heatmap.png",
    )
    visualizer.plot_summary_heatmap(
        mean_matrix,
        row_labels=method_names,
        col_labels=pollution_names,
        title="Held-out mean accuracy across severities",
        filename="summary_mean_accuracy_heatmap.png",
    )
    visualizer.plot_method_pollution_curves(
        SEVERITIES,
        all_results,
        method_name="Modern Hopfield",
        filename="mchn_pollution_severity_curves.png",
    )
    visualizer.plot_method_pollution_curves(
        SEVERITIES,
        all_results,
        method_name="Affine-robust Hopfield",
        filename="affine_robust_mchn_pollution_severity_curves.png",
    )


def run_end_to_end_system(loader, device, test_dir="./data/full_cars/ccpd_weather", max_images=3):
    print("\n" + "=" * 50)
    print("Task 3: end-to-end plate demo...")
    if not os.path.exists(test_dir) or len(os.listdir(test_dir)) == 0:
        print(f"Warning: test image directory not found or empty: {test_dir}")
        return

    pipeline = LPRPipeline(use_synthetic_pollution=False)
    mchn_models = build_hopfield_ensemble(loader.memory_matrix.to(device), device)
    template_labels = loader.labels.to(device)
    chinese_mask = torch.zeros(loader.memory_matrix.shape[0], dtype=torch.bool, device=device)
    alnum_mask = torch.zeros(loader.memory_matrix.shape[0], dtype=torch.bool, device=device)
    letter_mask = torch.zeros(loader.memory_matrix.shape[0], dtype=torch.bool, device=device)
    if loader.chinese_indices:
        chinese_mask[loader.chinese_indices] = True
    if loader.alnum_indices:
        alnum_mask[loader.alnum_indices] = True
    for idx, label_idx in enumerate(loader.labels.tolist()):
        label = loader.idx_to_label[int(label_idx)]
        if len(label) == 1 and "A" <= label <= "Z":
            letter_mask[idx] = True

    for img_name in [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))][:max_images]:
        img_path = os.path.join(test_dir, img_name)
        plate_img, chars_img_list = pipeline.process_image(img_path)
        if plate_img is None or not chars_img_list:
            print(f"{img_name}: segmentation failed")
            continue
        result = ""
        for char_img in chars_img_list:
            char_tensor = torch.tensor(cv2.resize(char_img, (32, 64)), dtype=torch.float32).view(1, 64, 32) / 255.0
            char_tensor = normalize_char_tensor(char_tensor, img_size=(32, 64)).view(1, -1).to(device)
            position = len(result)
            template_mask = chinese_mask if position == 0 else letter_mask if position == 1 and bool(letter_mask.any().item()) else alnum_mask
            with torch.no_grad():
                scores, _ = ensemble_hopfield_scores(
                    mchn_models,
                    char_tensor,
                    template_labels,
                    len(loader.idx_to_label),
                    template_mask=template_mask,
                )
            result += loader.idx_to_label[int(torch.argmax(scores, dim=-1))]
        print(f"{img_name}: {result}")


def parse_args():
    parser = argparse.ArgumentParser(description="Modern Hopfield polluted license-plate character evaluation")
    parser.add_argument("--pollution", default="all", choices=["all", "mixed", "mask", "noise", "salt_pepper", "blur", "fog", "dirt", "affine", "none"])
    parser.add_argument("--samples-per-level", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cnn-epochs", type=int, default=5)
    parser.add_argument("--cnn-train-samples", type=int, default=20000)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-e2e", action="store_true", default=True)
    parser.add_argument("--run-e2e", action="store_false", dest="skip_e2e")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Modern Hopfield held-out evaluation, device={device}")
    os.makedirs(args.output_dir, exist_ok=True)
    visualizer = MetricVisualizer(save_dir=args.output_dir)

    loader = TemplateLoader(
        data_roots=[os.path.join(args.data_dir, "chars2"), os.path.join(args.data_dir, "charsChinese")],
        img_size=(32, 64),
        cache_path=os.path.join(args.output_dir, "template_cache_32x64.pt"),
    )
    if loader.memory_matrix.shape[0] == 0:
        raise RuntimeError("Template memory is empty.")

    train_indices, test_indices = build_stratified_split(loader.labels, train_ratio=args.train_ratio, seed=args.seed)
    print(f"Held-out split: train templates={len(train_indices)}, test templates={len(test_indices)}")

    print("\nTraining CNN baseline on polluted training templates...")
    cnn = train_cnn(
        loader,
        train_indices=train_indices,
        num_classes=len(loader.idx_to_label),
        device=device,
        epochs=args.cnn_epochs,
        train_samples=args.cnn_train_samples,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    train_memory = loader.memory_matrix[train_indices].to(device)
    train_labels = loader.labels[train_indices].to(device)
    demo_models = build_hopfield_ensemble(train_memory, device)
    run_reconstruction_demo(
        {"hopfield": demo_models, "train_memory": train_memory, "train_labels": train_labels},
        loader,
        test_indices,
        visualizer,
        device,
    )

    pollution_types = ["mask", "noise", "salt_pepper", "blur", "fog", "dirt", "affine"]
    if args.pollution != "all":
        pollution_types = [args.pollution]

    all_results = {}
    for pollution_type in pollution_types:
        all_results[pollution_type] = run_robustness_evaluation(
            loader,
            visualizer,
            device,
            pollution_type=pollution_type,
            samples_per_level=args.samples_per_level,
            batch_size=args.batch_size,
            train_indices=train_indices,
            test_indices=test_indices,
            trained_cnn=cnn,
            seed=args.seed,
        )

    save_results_csv(args.output_dir, all_results)
    if args.pollution == "all":
        plot_all_pollution_summary(visualizer, all_results)

    if not args.skip_e2e:
        run_end_to_end_system(loader, device, test_dir=os.path.join(args.data_dir, "full_cars", "ccpd_weather"))

    print(f"\nAll tasks finished. Figures and CSV are saved in {args.output_dir}.")
