import argparse
import csv
import os
import random

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.lp_dataset import PollutedCharDataset, TemplateLoader, build_class_memory
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


def class_free_energy_scores(sim_scores, template_labels, beta, num_classes):
    labels = template_labels.to(sim_scores.device)
    scaled = beta * sim_scores
    scores = []
    for class_idx in range(num_classes):
        class_mask = labels == class_idx
        count = int(class_mask.sum().item())
        if count == 0:
            scores.append(scaled.new_full((scaled.shape[0],), -1e9))
            continue
        score = torch.logsumexp(scaled[:, class_mask], dim=-1) - torch.log(scaled.new_tensor(float(count)))
        scores.append(score)
    return torch.stack(scores, dim=-1)


def ensemble_hopfield_scores(models, q, template_labels, num_classes):
    fused = None
    primary_sim = None
    for model in models:
        _, _, sim_scores = model(q, return_similarity=True)
        scores = class_free_energy_scores(sim_scores, template_labels, beta=model.beta, num_classes=num_classes)
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

    hopfield_models = [
        ModernHopfieldNetwork(train_memory, beta=60.0, metric="dot", normalize=True, feature_mode="binary").to(device),
        ModernHopfieldNetwork(train_memory, beta=80.0, metric="dot", normalize=True, feature_mode="centered").to(device),
    ]

    def make_methods():
        return {
            "Modern Hopfield": lambda q: torch.argmax(
                ensemble_hopfield_scores(hopfield_models, q, train_labels, num_classes)[0], dim=-1
            ),
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


def run_end_to_end_system(loader, device, test_dir="./data/full_cars/ccpd_weather", max_images=3):
    print("\n" + "=" * 50)
    print("Task 3: end-to-end plate demo...")
    if not os.path.exists(test_dir) or len(os.listdir(test_dir)) == 0:
        print(f"Warning: test image directory not found or empty: {test_dir}")
        return

    pipeline = LPRPipeline(use_synthetic_pollution=False)
    mchn = ModernHopfieldNetwork(loader.memory_matrix.to(device), beta=60.0, metric="dot", normalize=True, feature_mode="binary").to(device)
    template_labels = loader.labels.to(device)

    for img_name in [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))][:max_images]:
        img_path = os.path.join(test_dir, img_name)
        plate_img, chars_img_list = pipeline.process_image(img_path)
        if plate_img is None or not chars_img_list:
            print(f"{img_name}: segmentation failed")
            continue
        result = ""
        for char_img in chars_img_list:
            char_tensor = torch.tensor(cv2.resize(char_img, (32, 64)), dtype=torch.float32, device=device).view(1, -1) / 255.0
            with torch.no_grad():
                _, _, sim_scores = mchn(char_tensor, return_similarity=True)
                scores = class_free_energy_scores(sim_scores, template_labels, mchn.beta, len(loader.idx_to_label))
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
    parser.add_argument("--skip-e2e", action="store_true")
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
    demo_models = [
        ModernHopfieldNetwork(train_memory, beta=60.0, metric="dot", normalize=True, feature_mode="binary").to(device),
        ModernHopfieldNetwork(train_memory, beta=80.0, metric="dot", normalize=True, feature_mode="centered").to(device),
    ]
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
