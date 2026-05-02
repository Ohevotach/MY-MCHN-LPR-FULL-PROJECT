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
from models.traditional_hopfield import TraditionalHopfieldNetwork
from utils.image_processing import LPRPipeline
from utils.metric_visuals import MetricVisualizer


SEVERITIES = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]
POLLUTIONS = ["mask", "noise", "salt_pepper", "blur", "fog", "dirt", "affine"]
CORE_POLLUTIONS = ["noise", "salt_pepper", "blur", "mask", "dirt", "fog"]
METHOD_ORDER = [
    "Modern Hopfield",
    "Affine-robust Hopfield",
    "Balanced Traditional Hopfield",
    "CNN",
    "Nearest Neighbor",
    "Euclidean NN",
    "Class Prototype",
]
CHINESE_LABEL_TO_PINYIN = {
    "京": "zh_jing",
    "津": "zh_jin",
    "冀": "zh_ji",
    "晋": "zh_jin1",
    "蒙": "zh_meng",
    "辽": "zh_liao",
    "吉": "zh_ji1",
    "黑": "zh_hei",
    "沪": "zh_hu",
    "苏": "zh_su",
    "浙": "zh_zhe",
    "皖": "zh_wan",
    "闽": "zh_min",
    "赣": "zh_gan",
    "鲁": "zh_lu",
    "豫": "zh_yu",
    "鄂": "zh_e",
    "湘": "zh_xiang",
    "粤": "zh_yue",
    "桂": "zh_gui1",
    "琼": "zh_qiong",
    "渝": "zh_yu1",
    "川": "zh_chuan",
    "贵": "zh_gui",
    "云": "zh_yun",
    "藏": "zh_zang",
    "陕": "zh_shan",
    "甘": "zh_gan1",
    "青": "zh_qing",
    "宁": "zh_ning",
    "新": "zh_xin",
}


def order_method_results(results):
    return {name: results[name] for name in METHOD_ORDER if name in results}


def display_label(label):
    return CHINESE_LABEL_TO_PINYIN.get(str(label), str(label))


def resolve_pollution_types(pollution_arg):
    value = str(pollution_arg).strip()
    if value == "all":
        return POLLUTIONS
    if value in {"core", "main", "plate_core"}:
        return CORE_POLLUTIONS
    if "," in value:
        pollution_types = [item.strip() for item in value.split(",") if item.strip()]
    else:
        pollution_types = [value]

    allowed = set(POLLUTIONS + ["mixed", "none"])
    unknown = [item for item in pollution_types if item not in allowed]
    if unknown:
        raise ValueError(
            f"Unsupported pollution(s): {', '.join(unknown)}. "
            f"Use one of: all, core, mixed, none, {', '.join(POLLUTIONS)}; "
            "or a comma-separated list such as noise,salt_pepper,blur,mask,dirt,fog."
        )
    return pollution_types


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
        ModernHopfieldNetwork(memory, beta=28.0, metric="dot", normalize=True, feature_mode="binary").to(device),
        ModernHopfieldNetwork(memory, beta=32.0, metric="dot", normalize=True, feature_mode="centered").to(device),
        ModernHopfieldNetwork(memory, beta=30.0, metric="dot", normalize=True, feature_mode="binary_centered").to(device),
        ModernHopfieldNetwork(memory, beta=26.0, metric="dot", normalize=True, feature_mode="hybrid_shape").to(device),
        ModernHopfieldNetwork(memory, beta=30.0, metric="dot", normalize=True, feature_mode="profile").to(device),
        ModernHopfieldNetwork(memory, beta=2.5, metric="euclidean", normalize=False, feature_mode="profile").to(device),
    ]


def augment_hopfield_memory(memory, labels, img_h=64, img_w=32):
    if memory.numel() == 0:
        return memory, labels
    imgs = memory.float().view(-1, 1, img_h, img_w)
    variants = [
        imgs,
        torch.roll(imgs, shifts=1, dims=2),
        torch.roll(imgs, shifts=-1, dims=2),
        torch.roll(imgs, shifts=1, dims=3),
        torch.roll(imgs, shifts=-1, dims=3),
    ]
    dilated = F.max_pool2d(imgs, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-imgs, kernel_size=3, stride=1, padding=1)
    blurred = F.avg_pool2d(F.pad(imgs, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)
    variants.extend([dilated, eroded, 0.65 * imgs + 0.35 * blurred])
    variants.extend(affine_memory_variants(imgs))
    stacked = torch.cat([v.clamp(0.0, 1.0).view(memory.shape[0], -1) for v in variants], dim=0)
    return stacked.contiguous(), labels.repeat(len(variants)).contiguous()


def affine_memory_variants(imgs):
    device = imgs.device
    dtype = imgs.dtype
    transforms_ = []
    for angle in (-7.0, 7.0):
        rad = np.deg2rad(angle)
        transforms_.append([[np.cos(rad), -np.sin(rad), 0.0], [np.sin(rad), np.cos(rad), 0.0]])
    for shear in (-0.12, 0.12):
        transforms_.append([[1.0, shear, 0.0], [0.0, 1.0, 0.0]])
    for scale_x in (0.88, 1.12):
        transforms_.append([[scale_x, 0.0, 0.0], [0.0, 1.0, 0.0]])

    out = []
    for matrix in transforms_:
        theta = torch.tensor(matrix, dtype=dtype, device=device).unsqueeze(0).repeat(imgs.shape[0], 1, 1)
        grid = F.affine_grid(theta, imgs.shape, align_corners=False)
        out.append(F.grid_sample(imgs, grid, mode="bilinear", padding_mode="zeros", align_corners=False))
    return out


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
        score = torch.max(scaled[:, class_mask], dim=-1).values
        scores.append(score)
    return torch.stack(scores, dim=-1)


def ensemble_hopfield_scores(models, q, template_labels, num_classes, template_mask=None):
    log_prob_parts = []
    primary_sim = None
    for model in models:
        _, _, sim_scores = model(q, template_mask=template_mask, return_similarity=True)
        scores = class_free_energy_scores(sim_scores, template_labels, beta=model.beta, num_classes=num_classes, template_mask=template_mask)
        log_probs = torch.log_softmax(scores, dim=-1)
        log_prob_parts.append(log_probs)
        if primary_sim is None:
            primary_sim = sim_scores
    fused = torch.logsumexp(torch.stack(log_prob_parts, dim=0), dim=0) - torch.log(
        primary_sim.new_tensor(float(len(log_prob_parts)))
    )
    return fused, primary_sim


def predict_modern_hopfield_scores(models, q, template_labels, num_classes, template_mask=None):
    return ensemble_hopfield_scores(models, q, template_labels, num_classes, template_mask=template_mask)[0]


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


def build_affine_query_variants(q, variant_level="light"):
    tensors = []
    for row in q.detach().cpu().view(-1, 64, 32).numpy():
        base = normalize_char_array(row * 255.0)
        variants = [base]
        if variant_level == "none":
            pass
        elif variant_level == "light":
            for angle in (-8, 8):
                matrix = cv2.getRotationMatrix2D((16, 32), angle, 1.0)
                variants.append(normalize_char_array(cv2.warpAffine(base, matrix, (32, 64), flags=cv2.INTER_NEAREST, borderValue=0)))
        elif variant_level == "medium":
            for angle in (-10, 10):
                matrix = cv2.getRotationMatrix2D((16, 32), angle, 1.0)
                variants.append(normalize_char_array(cv2.warpAffine(base, matrix, (32, 64), flags=cv2.INTER_NEAREST, borderValue=0)))
            for shear in (-0.10, 0.10):
                matrix = np.array([[1.0, shear, -shear * 32.0], [0.0, 1.0, 0.0]], dtype=np.float32)
                variants.append(normalize_char_array(cv2.warpAffine(base, matrix, (32, 64), flags=cv2.INTER_NEAREST, borderValue=0)))
        elif variant_level == "full":
            for angle in (-14, -8, 8, 14):
                matrix = cv2.getRotationMatrix2D((16, 32), angle, 1.0)
                variants.append(normalize_char_array(cv2.warpAffine(base, matrix, (32, 64), flags=cv2.INTER_NEAREST, borderValue=0)))
            for shear in (-0.16, -0.08, 0.08, 0.16):
                matrix = np.array([[1.0, shear, -shear * 32.0], [0.0, 1.0, 0.0]], dtype=np.float32)
                variants.append(normalize_char_array(cv2.warpAffine(base, matrix, (32, 64), flags=cv2.INTER_NEAREST, borderValue=0)))
        else:
            raise ValueError(f"Unsupported affine variant level: {variant_level}")
        tensors.extend(torch.tensor(item, dtype=torch.float32).view(1, -1) / 255.0 for item in variants)
    variants_per_sample = {
        "none": 1,
        "light": 3,
        "medium": 5,
        "full": 9,
    }[variant_level]
    return torch.cat(tensors, dim=0), variants_per_sample


def predict_affine_robust_hopfield(models, q, template_labels, num_classes, variant_level="light"):
    base_scores, _ = ensemble_hopfield_scores(models, q, template_labels, num_classes)
    base_probs = torch.softmax(base_scores, dim=-1)
    base_conf, base_pred = torch.max(base_probs, dim=-1)

    if variant_level == "none":
        return base_pred

    q_variants, variants_per_sample = build_affine_query_variants(q, variant_level=variant_level)
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=device.type == "cuda")
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


def evaluate_topk_score_methods(score_methods, test_loader, device, topk=(1, 3)):
    correct = {name: {k: 0 for k in topk} for name in score_methods}
    total = 0
    with torch.no_grad():
        for q, _, labels in test_loader:
            q = q.to(device)
            labels = labels.to(device)
            max_k = max(topk)
            for name, scorer in score_methods.items():
                scores = scorer(q)
                _, pred = torch.topk(scores, k=min(max_k, scores.shape[-1]), dim=-1)
                for k in topk:
                    kk = min(k, pred.shape[-1])
                    correct[name][k] += pred[:, :kk].eq(labels.unsqueeze(1)).any(dim=1).sum().item()
            total += labels.size(0)
    return {
        name: {k: 100.0 * value / max(total, 1) for k, value in values.items()}
        for name, values in correct.items()
    }


def evaluate_methods_with_topk(methods, score_methods, test_loader, device, topk=(3,)):
    correct = {name: 0 for name in methods}
    topk_correct = {name: {k: 0 for k in topk} for name in score_methods}
    total = 0
    with torch.no_grad():
        for q, _, labels in test_loader:
            q = q.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            score_cache = {}
            for name, scorer in score_methods.items():
                score_cache[name] = scorer(q)

            for name, predictor in methods.items():
                if name in score_cache:
                    pred = torch.argmax(score_cache[name], dim=-1)
                else:
                    pred = predictor(q)
                correct[name] += (pred == labels).sum().item()

            if topk:
                max_k = max(topk)
                for name, scores in score_cache.items():
                    _, pred = torch.topk(scores, k=min(max_k, scores.shape[-1]), dim=-1)
                    for k in topk:
                        kk = min(k, pred.shape[-1])
                        topk_correct[name][k] += pred[:, :kk].eq(labels.unsqueeze(1)).any(dim=1).sum().item()

            total += labels.size(0)

    accs = {name: 100.0 * value / max(total, 1) for name, value in correct.items()}
    topk_accs = {
        name: {k: 100.0 * value / max(total, 1) for k, value in values.items()}
        for name, values in topk_correct.items()
    }
    return accs, topk_accs


def collect_prediction_outputs(methods, test_loader, device):
    labels_all = []
    preds_all = {name: [] for name in methods}
    with torch.no_grad():
        for q, _, labels in test_loader:
            q = q.to(device)
            labels = labels.to(device)
            labels_all.append(labels.detach().cpu())
            for name, predictor in methods.items():
                preds_all[name].append(predictor(q).detach().cpu())
    labels_all = torch.cat(labels_all, dim=0)
    preds_all = {name: torch.cat(parts, dim=0) for name, parts in preds_all.items()}
    return labels_all, preds_all


def build_confusion_matrix(labels, preds, num_classes):
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for target, pred in zip(labels.view(-1), preds.view(-1)):
        matrix[int(target), int(pred)] += 1
    return matrix.numpy()


def build_class_balanced_indices(labels, base_indices, samples_per_class, seed=2026):
    rng = random.Random(seed)
    by_class = {}
    labels_list = labels.tolist()
    for idx in base_indices:
        by_class.setdefault(int(labels_list[idx]), []).append(int(idx))

    balanced = []
    for class_idx in sorted(by_class):
        indices = by_class[class_idx]
        if not indices:
            continue
        for _ in range(max(1, int(samples_per_class))):
            balanced.append(rng.choice(indices))
    rng.shuffle(balanced)
    return balanced


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
    include_affine_robust=False,
    affine_variant_level="light",
    save_confusion=False,
    num_workers=0,
):
    print("\n" + "=" * 50)
    print(f"Task 2: held-out evaluation, pollution={pollution_type}...")
    train_memory = loader.memory_matrix[train_indices].to(device)
    train_labels = loader.labels[train_indices].to(device)
    hopfield_memory, hopfield_labels = augment_hopfield_memory(train_memory, train_labels)
    prototypes, prototype_labels = build_class_memory_from_tensors(train_memory, train_labels)
    num_classes = len(loader.idx_to_label)

    hopfield_models = build_hopfield_ensemble(hopfield_memory, device)
    # Classical Hopfield is kept as a fair thesis baseline. Character images are
    # sparse, so the implementation balances patterns before Hebbian storage and
    # stores one prototype per class to avoid background-dominated cross-talk.
    traditional_hopfield = TraditionalHopfieldNetwork(
        prototypes,
        prototype_labels,
        steps=6,
        center_patterns=True,
        retrieval_weight=0.35,
    ).to(device)

    def make_methods():
        methods = {
            "Balanced Traditional Hopfield": lambda q: traditional_hopfield.predict(q),
            "Nearest Neighbor": lambda q: predict_nearest_neighbor(q, train_memory, train_labels, metric="cosine"),
            "Euclidean NN": lambda q: predict_nearest_neighbor(q, train_memory, train_labels, metric="euclidean"),
            "Class Prototype": lambda q: predict_prototype(q, prototypes, prototype_labels),
        }
        methods["Modern Hopfield"] = lambda q: torch.argmax(
            predict_modern_hopfield_scores(hopfield_models, q, hopfield_labels, num_classes), dim=-1
        )
        methods["CNN"] = lambda q: torch.argmax(trained_cnn(q), dim=-1)
        if include_affine_robust:
            methods["Affine-robust Hopfield"] = lambda q: predict_affine_robust_hopfield(
                hopfield_models,
                q,
                hopfield_labels,
                num_classes,
                variant_level=affine_variant_level,
            )
        return order_method_results(methods)

    def make_score_methods():
        return {
            "Modern Hopfield": lambda q: predict_modern_hopfield_scores(hopfield_models, q, hopfield_labels, num_classes),
            "CNN": lambda q: trained_cnn(q),
        }

    results = {name: [] for name in make_methods()}
    top3_results = {f"{name} Top-3": [] for name in make_score_methods()}
    for severity in SEVERITIES:
        test_dataset = PollutedCharDataset(
            loader,
            virtual_size=samples_per_level,
            pollution_type=pollution_type,
            severity=severity,
            seed=seed,
            sample_indices=test_indices,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(0, int(num_workers)),
            pin_memory=device.type == "cuda",
        )
        accs, topk_accs = evaluate_methods_with_topk(make_methods(), make_score_methods(), test_loader, device, topk=(3,))
        for name, acc in accs.items():
            results[name].append(acc)
        for name, values in topk_accs.items():
            top3_results[f"{name} Top-3"].append(values[3])
        top3_text = " | ".join(f"{k}: {v[-1]:.2f}%" for k, v in top3_results.items())
        print(
            "  severity="
            + f"{severity:.1f}: "
            + " | ".join(f"{k}: {v:.2f}%" for k, v in accs.items())
            + " | "
            + top3_text
        )

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
    save_topk_results_csv(visualizer.save_dir, pollution_type, top3_results)
    if save_confusion:
        save_confusion_reports(
            visualizer,
            pollution_type,
            loader.idx_to_label,
            make_methods(),
            loader,
            test_indices,
            batch_size,
            device,
            severity=SEVERITIES[-1],
            method_names=("Modern Hopfield", "Balanced Traditional Hopfield", "CNN"),
            num_workers=num_workers,
        )
    return results


def run_class_balanced_evaluation(
    loader,
    visualizer,
    device,
    pollution_type,
    samples_per_class,
    batch_size,
    train_indices,
    test_indices,
    trained_cnn,
    seed,
    include_affine_robust=False,
    affine_variant_level="light",
    num_workers=0,
):
    print("\n" + "=" * 50)
    print(f"Task 2c: class-balanced evaluation, pollution={pollution_type}...")
    balanced_indices = build_class_balanced_indices(
        loader.labels,
        test_indices,
        samples_per_class=samples_per_class,
        seed=seed + 1009,
    )
    if not balanced_indices:
        print("Warning: class-balanced evaluation skipped because no balanced indices were built.")
        return {}

    train_memory = loader.memory_matrix[train_indices].to(device)
    train_labels = loader.labels[train_indices].to(device)
    hopfield_memory, hopfield_labels = augment_hopfield_memory(train_memory, train_labels)
    prototypes, prototype_labels = build_class_memory_from_tensors(train_memory, train_labels)
    num_classes = len(loader.idx_to_label)

    hopfield_models = build_hopfield_ensemble(hopfield_memory, device)
    traditional_hopfield = TraditionalHopfieldNetwork(
        prototypes,
        prototype_labels,
        steps=6,
        center_patterns=True,
        retrieval_weight=0.35,
    ).to(device)

    def make_methods():
        methods = {
            "Balanced Traditional Hopfield": lambda q: traditional_hopfield.predict(q),
            "Nearest Neighbor": lambda q: predict_nearest_neighbor(q, train_memory, train_labels, metric="cosine"),
            "Euclidean NN": lambda q: predict_nearest_neighbor(q, train_memory, train_labels, metric="euclidean"),
            "Class Prototype": lambda q: predict_prototype(q, prototypes, prototype_labels),
            "Modern Hopfield": lambda q: torch.argmax(
                predict_modern_hopfield_scores(hopfield_models, q, hopfield_labels, num_classes), dim=-1
            ),
            "CNN": lambda q: torch.argmax(trained_cnn(q), dim=-1),
        }
        if include_affine_robust:
            methods["Affine-robust Hopfield"] = lambda q: predict_affine_robust_hopfield(
                hopfield_models,
                q,
                hopfield_labels,
                num_classes,
                variant_level=affine_variant_level,
            )
        return order_method_results(methods)

    results = {name: [] for name in make_methods()}
    for severity in SEVERITIES:
        dataset = PollutedCharDataset(
            loader,
            virtual_size=len(balanced_indices),
            pollution_type=pollution_type,
            severity=severity,
            seed=seed + 2003,
            sample_indices=balanced_indices,
        )
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(0, int(num_workers)),
            pin_memory=device.type == "cuda",
        )
        accs = evaluate_methods(make_methods(), data_loader, device)
        for name, acc in accs.items():
            results[name].append(acc)
        print("  balanced severity=" + f"{severity:.1f}: " + " | ".join(f"{k}: {v:.2f}%" for k, v in accs.items()))

    visualizer.plot_multi_robustness_curve(
        SEVERITIES,
        results,
        pollution_type=f"class-balanced {pollution_type}",
        filename=f"balanced_robustness_{pollution_type}_methods_curve.png",
    )
    visualizer.plot_final_severity_bar(
        {name: values[-1] for name, values in results.items()},
        pollution_type=f"class-balanced {pollution_type}",
        severity=SEVERITIES[-1],
        filename=f"balanced_robustness_{pollution_type}_methods_final_bar.png",
    )
    return results


def run_ablation_evaluation(
    loader,
    visualizer,
    device,
    train_indices,
    test_indices,
    samples,
    batch_size,
    pollution_type="mixed",
    severity=0.6,
    seed=2026,
):
    print("\n" + "=" * 50)
    print(f"Task 2b: ablation study, pollution={pollution_type}, severity={severity}...")
    train_memory = loader.memory_matrix[train_indices].to(device)
    train_labels = loader.labels[train_indices].to(device)
    aug_memory, aug_labels = augment_hopfield_memory(train_memory, train_labels)
    num_classes = len(loader.idx_to_label)

    ablations = {
        "MCHN-Raw": (
            [ModernHopfieldNetwork(train_memory, beta=30.0, metric="dot", normalize=True, feature_mode="raw").to(device)],
            train_labels,
        ),
        "MCHN-Binary": (
            [ModernHopfieldNetwork(train_memory, beta=30.0, metric="dot", normalize=True, feature_mode="binary").to(device)],
            train_labels,
        ),
        "MCHN-Shape": (
            [ModernHopfieldNetwork(train_memory, beta=28.0, metric="dot", normalize=True, feature_mode="hybrid_shape").to(device)],
            train_labels,
        ),
        "MCHN-Profile": (
            [ModernHopfieldNetwork(train_memory, beta=30.0, metric="dot", normalize=True, feature_mode="profile").to(device)],
            train_labels,
        ),
        "MCHN-Ensemble-NoAug": (build_hopfield_ensemble(train_memory, device), train_labels),
        "MCHN-Ensemble-Aug": (build_hopfield_ensemble(aug_memory, device), aug_labels),
    }

    dataset = PollutedCharDataset(
        loader,
        virtual_size=samples,
        pollution_type=pollution_type,
        severity=severity,
        seed=seed + 17,
        sample_indices=test_indices,
    )
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    methods = {
        name: (lambda q, models=models, labels=labels: torch.argmax(
            predict_modern_hopfield_scores(models, q, labels, num_classes),
            dim=-1,
        ))
        for name, (models, labels) in ablations.items()
    }
    scores = evaluate_methods(methods, test_loader, device)
    for name, value in scores.items():
        print(f"  {name}: {value:.2f}%")

    csv_path = os.path.join(visualizer.save_dir, "ablation_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pollution", "severity", "method", "accuracy"])
        for name, value in scores.items():
            writer.writerow([pollution_type, severity, name, f"{value:.4f}"])
    visualizer.plot_final_severity_bar(
        scores,
        pollution_type=f"ablation_{pollution_type}",
        severity=severity,
        filename="ablation_final_bar.png",
    )
    print(f"Saved ablation CSV: {csv_path}")
    return scores


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


def save_named_results_csv(output_dir, all_results, filename):
    csv_path = os.path.join(output_dir, filename)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pollution", "method", *[f"severity_{s}" for s in SEVERITIES]])
        for pollution, method_results in all_results.items():
            for method, values in method_results.items():
                writer.writerow([pollution, method, *[f"{value:.4f}" for value in values]])
    print(f"Saved CSV: {csv_path}")


def save_topk_results_csv(output_dir, pollution_type, topk_results):
    csv_path = os.path.join(output_dir, f"topk_{pollution_type}_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pollution", "metric", *[f"severity_{s}" for s in SEVERITIES]])
        for metric, values in topk_results.items():
            writer.writerow([pollution_type, metric, *[f"{value:.4f}" for value in values]])
    print(f"Saved Top-k CSV: {csv_path}")


def save_confusion_reports(
    visualizer,
    pollution_type,
    idx_to_label,
    methods,
    loader,
    test_indices,
    batch_size,
    device,
    severity,
    method_names=("Modern Hopfield",),
    num_workers=0,
):
    selected_methods = {name: methods[name] for name in method_names if name in methods}
    if not selected_methods:
        return

    dataset = PollutedCharDataset(
        loader,
        virtual_size=max(1000, len(test_indices) * 20),
        pollution_type=pollution_type,
        severity=severity,
        seed=3026,
        sample_indices=test_indices,
    )
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=device.type == "cuda",
    )
    true_labels, predictions = collect_prediction_outputs(selected_methods, test_loader, device)
    label_names = [display_label(idx_to_label[i]) for i in range(len(idx_to_label))]

    for method_name, pred_labels in predictions.items():
        safe_name = method_name.lower().replace(" ", "_").replace("-", "_")
        matrix = build_confusion_matrix(true_labels, pred_labels, len(idx_to_label))
        matrix_path = os.path.join(visualizer.save_dir, f"confusion_{pollution_type}_{safe_name}.csv")
        with open(matrix_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["true\\pred", *label_names])
            for label_name, row in zip(label_names, matrix):
                writer.writerow([label_name, *row.tolist()])
        visualizer.plot_confusion_matrix(
            matrix,
            label_names,
            title=f"{method_name} confusion matrix ({pollution_type}, severity={severity})",
            filename=f"confusion_{pollution_type}_{safe_name}.png",
        )
        save_top_confusions_csv(visualizer.save_dir, pollution_type, safe_name, matrix, label_names)
        save_group_accuracy_csv(visualizer.save_dir, pollution_type, safe_name, matrix, label_names)
        print(f"Saved confusion report: {matrix_path}")


def save_top_confusions_csv(output_dir, pollution_type, method_name, matrix, label_names, top_n=30):
    pairs = []
    for i in range(matrix.shape[0]):
        row_total = int(matrix[i].sum())
        if row_total == 0:
            continue
        for j in range(matrix.shape[1]):
            if i == j or matrix[i, j] == 0:
                continue
            pairs.append((int(matrix[i, j]), row_total, label_names[i], label_names[j]))
    pairs.sort(key=lambda item: item[0], reverse=True)

    csv_path = os.path.join(output_dir, f"top_confusions_{pollution_type}_{method_name}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["wrong_count", "true_total", "true_label", "pred_label", "error_rate"])
        for wrong_count, true_total, true_label, pred_label in pairs[:top_n]:
            writer.writerow([wrong_count, true_total, true_label, pred_label, f"{wrong_count / max(true_total, 1):.4f}"])
    print(f"Saved top confusions CSV: {csv_path}")


def label_group(label):
    text = str(label)
    if len(text) == 1 and text.isdigit():
        return "digit"
    if len(text) == 1 and "A" <= text <= "Z":
        return "letter"
    return "chinese"


def save_group_accuracy_csv(output_dir, pollution_type, method_name, matrix, label_names):
    groups = {}
    for idx, label in enumerate(label_names):
        group = label_group(label)
        if group not in groups:
            groups[group] = {"correct": 0, "total": 0}
        groups[group]["correct"] += int(matrix[idx, idx])
        groups[group]["total"] += int(matrix[idx].sum())

    csv_path = os.path.join(output_dir, f"group_accuracy_{pollution_type}_{method_name}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pollution", "method", "group", "correct", "total", "accuracy"])
        for group in ("chinese", "letter", "digit"):
            values = groups.get(group, {"correct": 0, "total": 0})
            accuracy = 100.0 * values["correct"] / max(values["total"], 1)
            writer.writerow([pollution_type, method_name, group, values["correct"], values["total"], f"{accuracy:.4f}"])
    print(f"Saved group accuracy CSV: {csv_path}")


def save_mchn_memory_artifacts(loader, train_indices, test_indices, hopfield_memory, hopfield_labels, output_dir="./saved_weights"):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "mchn_eval_memory_32x64.pt")
    torch.save(
        {
            "image_shape": (64, 32),
            "original_memory_matrix": loader.memory_matrix.cpu(),
            "original_labels": loader.labels.cpu(),
            "train_indices": torch.tensor(train_indices, dtype=torch.long),
            "test_indices": torch.tensor(test_indices, dtype=torch.long),
            "hopfield_memory_matrix": hopfield_memory.detach().cpu(),
            "hopfield_labels": hopfield_labels.detach().cpu(),
            "idx_to_label": loader.idx_to_label,
            "label_to_idx": loader.label_to_idx,
            "description": "Held-out Modern Hopfield associative memory used by main_eval.py.",
        },
        save_path,
    )
    print(f"Saved MCHN evaluation memory matrix: {save_path}")


def plot_all_pollution_summary(visualizer, all_results, prefix=""):
    available_methods = list(next(iter(all_results.values())).keys())
    method_names = [name for name in METHOD_ORDER if name in available_methods]
    method_names.extend(name for name in available_methods if name not in method_names)
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
        filename=f"{prefix}summary_final_severity_heatmap.png",
    )
    visualizer.plot_summary_heatmap(
        mean_matrix,
        row_labels=method_names,
        col_labels=pollution_names,
        title="Held-out mean accuracy across severities",
        filename=f"{prefix}summary_mean_accuracy_heatmap.png",
    )
    for method_name, filename in (
        ("Modern Hopfield", "mchn_pollution_severity_curves.png"),
        ("Affine-robust Hopfield", "affine_robust_mchn_pollution_severity_curves.png"),
    ):
        if any(method_name in method_results for method_results in all_results.values()):
            visualizer.plot_method_pollution_curves(
                SEVERITIES,
                all_results,
                method_name=method_name,
                filename=f"{prefix}{filename}",
            )


def save_summary_ranking_csv(output_dir, all_results, filename="summary_method_ranking.csv"):
    csv_path = os.path.join(output_dir, filename)
    rows = []
    for pollution, method_results in all_results.items():
        for method, values in method_results.items():
            rows.append(
                {
                    "pollution": pollution,
                    "method": method,
                    "final_accuracy": values[-1],
                    "mean_accuracy": sum(values) / max(len(values), 1),
                    "min_accuracy": min(values),
                    "max_accuracy": max(values),
                }
            )
    rows.sort(key=lambda row: (row["pollution"], -row["mean_accuracy"]))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pollution", "method", "final_accuracy", "mean_accuracy", "min_accuracy", "max_accuracy"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{value:.4f}" if isinstance(value, float) else value for key, value in row.items()})
    print(f"Saved summary ranking CSV: {csv_path}")


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
    parser.add_argument(
        "--pollution",
        default="all",
        help=(
            "Pollution to evaluate. Use 'core' for noise,salt_pepper,blur,mask,dirt,fog; "
            "use 'all' for mask,noise,salt_pepper,blur,fog,dirt,affine; "
            "or pass a comma-separated list."
        ),
    )
    parser.add_argument("--samples-per-level", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cnn-epochs", type=int, default=5)
    parser.add_argument("--cnn-train-samples", type=int, default=20000)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--include-affine-robust", action="store_true", help="Also evaluate the slow affine-query Hopfield variant.")
    parser.add_argument("--affine-variant-level", default="light", choices=["none", "light", "medium", "full"])
    parser.add_argument("--save-confusion", action="store_true", default=True, help="Save confusion matrices for each pollution.")
    parser.add_argument("--skip-confusion", action="store_false", dest="save_confusion", help="Skip confusion matrices for a faster exploratory run.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers. Use 2 on Kaggle if CPU preprocessing is the bottleneck.")
    parser.add_argument("--skip-balanced-eval", action="store_true", help="Skip class-balanced robustness evaluation.")
    parser.add_argument("--balanced-samples-per-class", type=int, default=8)
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--ablation-samples", type=int, default=1000)
    parser.add_argument("--ablation-pollution", default="mixed", choices=["mixed", "mask", "noise", "salt_pepper", "blur", "fog", "dirt", "affine", "none"])
    parser.add_argument("--ablation-severity", type=float, default=0.6)
    parser.add_argument("--skip-e2e", action="store_true", default=True)
    parser.add_argument("--run-e2e", action="store_false", dest="skip_e2e")
    parser.add_argument(
        "--save-mchn-memory",
        action="store_true",
        default=True,
        help="Save the large augmented MCHN evaluation memory artifact. Enabled by default for reproducible experiment artifacts.",
    )
    parser.add_argument(
        "--skip-save-mchn-memory",
        action="store_false",
        dest="save_mchn_memory",
        help="Skip the large MCHN memory artifact when storage or Kaggle upload size is limited.",
    )
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--saved-weights-dir", default="./saved_weights")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Modern Hopfield held-out evaluation, device={device}")
    if torch.cuda.is_available():
        print(f"CUDA devices visible: {torch.cuda.device_count()} (single-process evaluation uses cuda:0).")
    if args.save_confusion:
        print("Confusion reports are enabled; this adds an extra evaluation pass per pollution.")
    if args.include_affine_robust:
        print(f"Affine-robust Hopfield enabled with variant level: {args.affine_variant_level}.")
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
    demo_memory, demo_labels = augment_hopfield_memory(train_memory, train_labels)
    if args.save_mchn_memory:
        save_mchn_memory_artifacts(loader, train_indices, test_indices, demo_memory, demo_labels, args.saved_weights_dir)
    else:
        print("Skipping large MCHN evaluation memory artifact. Use --save-mchn-memory if you need it.")
    demo_models = build_hopfield_ensemble(demo_memory.to(device), device)
    run_reconstruction_demo(
        {"hopfield": demo_models, "train_memory": demo_memory.to(device), "train_labels": demo_labels.to(device)},
        loader,
        test_indices,
        visualizer,
        device,
    )

    pollution_types = resolve_pollution_types(args.pollution)
    print(f"Pollutions: {', '.join(pollution_types)}")

    all_results = {}
    balanced_results = {}
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
            include_affine_robust=args.include_affine_robust,
            affine_variant_level=args.affine_variant_level,
            save_confusion=args.save_confusion,
            num_workers=args.num_workers,
        )
        if not args.skip_balanced_eval:
            balanced_results[pollution_type] = run_class_balanced_evaluation(
                loader,
                visualizer,
                device,
                pollution_type=pollution_type,
                samples_per_class=args.balanced_samples_per_class,
                batch_size=args.batch_size,
                train_indices=train_indices,
                test_indices=test_indices,
                trained_cnn=cnn,
                seed=args.seed,
                include_affine_robust=args.include_affine_robust,
                affine_variant_level=args.affine_variant_level,
                num_workers=args.num_workers,
            )

    save_results_csv(args.output_dir, all_results)
    save_summary_ranking_csv(args.output_dir, all_results)
    if balanced_results:
        save_named_results_csv(args.output_dir, balanced_results, "balanced_robustness_all_results.csv")
        save_summary_ranking_csv(args.output_dir, balanced_results, filename="balanced_summary_method_ranking.csv")
    if len(pollution_types) > 1:
        plot_all_pollution_summary(visualizer, all_results)
        if balanced_results:
            plot_all_pollution_summary(visualizer, balanced_results, prefix="balanced_")

    if not args.skip_ablation:
        run_ablation_evaluation(
            loader,
            visualizer,
            device,
            train_indices=train_indices,
            test_indices=test_indices,
            samples=args.ablation_samples,
            batch_size=args.batch_size,
            pollution_type=args.ablation_pollution,
            severity=args.ablation_severity,
            seed=args.seed,
        )

    if not args.skip_e2e:
        run_end_to_end_system(loader, device, test_dir=os.path.join(args.data_dir, "full_cars", "ccpd_weather"))

    print(f"\nAll tasks finished. Figures and CSV are saved in {args.output_dir}.")
