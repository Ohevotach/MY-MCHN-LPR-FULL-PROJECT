import argparse
import csv
import os
import random
import sys
from collections import defaultdict

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dataset.lp_dataset import TemplateLoader, normalize_char_tensor
from main_eval import augment_hopfield_memory, build_hopfield_ensemble, fast_classic_hopfield_scores
from utils.image_processing import PlateSegmenter

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


POLLUTIONS = ["none", "mask", "noise", "salt_pepper", "blur", "fog", "dirt", "affine"]
CORE_PLATE_POLLUTIONS = ["noise", "salt_pepper", "blur", "mask", "dirt", "fog"]
DEFAULT_SEVERITIES = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]


def is_chinese_label(label):
    text = str(label)
    return text.startswith("zh_") or (len(text) == 1 and "\u4e00" <= text <= "\u9fff")


def build_template_masks(loader, labels, device):
    chinese_mask = torch.zeros(labels.numel(), dtype=torch.bool, device=device)
    letter_mask = torch.zeros_like(chinese_mask)
    digit_mask = torch.zeros_like(chinese_mask)
    for idx, label_idx in enumerate(labels.detach().cpu().tolist()):
        label = str(loader.idx_to_label[int(label_idx)])
        if is_chinese_label(label):
            chinese_mask[idx] = True
        elif len(label) == 1 and "A" <= label <= "Z":
            letter_mask[idx] = True
        elif len(label) == 1 and label.isdigit():
            digit_mask[idx] = True
    alnum_mask = letter_mask | digit_mask
    plate_tail_mask = alnum_mask.clone()
    for idx, label_idx in enumerate(labels.detach().cpu().tolist()):
        if str(loader.idx_to_label[int(label_idx)]) in {"I", "O"}:
            plate_tail_mask[idx] = False
    return chinese_mask, letter_mask, plate_tail_mask, alnum_mask


def plate_position_mask(position, masks):
    chinese_mask, letter_mask, plate_tail_mask, alnum_mask = masks
    if position == 0:
        return chinese_mask
    if position == 1 and bool(letter_mask.any().item()):
        return letter_mask
    return plate_tail_mask if bool(plate_tail_mask.any().item()) else alnum_mask


def plate_position_class_mask(position, loader, device):
    mask = torch.zeros(len(loader.idx_to_label), dtype=torch.bool, device=device)
    for class_idx, label in loader.idx_to_label.items():
        text = str(label)
        if position == 0:
            allowed = is_chinese_label(text)
        elif position == 1:
            allowed = len(text) == 1 and "A" <= text <= "Z"
        else:
            allowed = (len(text) == 1 and ("A" <= text <= "Z" or text.isdigit())) and text not in {"I", "O"}
        mask[int(class_idx)] = bool(allowed)
    return mask


def normalize_char_image(arr):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape != (64, 32):
        arr = cv2.resize(arr, (32, 64), interpolation=cv2.INTER_NEAREST)
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary > 0) > 0.55:
        binary = cv2.bitwise_not(binary)
    binary = strip_character_frame_lines(binary)
    ys, xs = np.where(binary > 0)
    if len(xs) == 0 or len(ys) == 0:
        return binary
    crop = binary[max(0, ys.min() - 1) : min(64, ys.max() + 2), max(0, xs.min() - 1) : min(32, xs.max() + 2)]
    h, w = crop.shape[:2]
    scale = min(26.0 / max(1, w), 56.0 / max(1, h))
    new_w = max(1, min(30, int(round(w * scale))))
    new_h = max(1, min(62, int(round(h * scale))))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((64, 32), dtype=np.uint8)
    y = (64 - new_h) // 2
    x = (32 - new_w) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def strip_character_frame_lines(binary):
    work = (binary > 0).astype(np.uint8) * 255
    if work.shape != (64, 32):
        work = cv2.resize(work, (32, 64), interpolation=cv2.INTER_NEAREST)
    cleaned = work.copy()
    h_img, w_img = cleaned.shape[:2]
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        near_side = x <= 2 or x + w >= w_img - 2
        near_top_bottom = y <= 2 or y + h >= h_img - 2
        if near_side and h >= 0.62 * h_img and w <= 0.12 * w_img:
            cv2.drawContours(cleaned, [cnt], -1, 0, thickness=-1)
        elif near_top_bottom and w >= 0.55 * w_img and h <= 0.12 * h_img:
            cv2.drawContours(cleaned, [cnt], -1, 0, thickness=-1)
        elif (near_side or near_top_bottom) and area < 0.008 * h_img * w_img:
            cv2.drawContours(cleaned, [cnt], -1, 0, thickness=-1)
    return cleaned


def keep_likely_character_components(binary):
    work = (binary > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return work
    h_img, w_img = work.shape[:2]
    center_x = w_img / 2.0
    scored = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area < max(3, 0.004 * h_img * w_img):
            continue
        center_score = 1.0 - min(abs((x + w / 2.0) - center_x) / max(center_x, 1.0), 1.0)
        height_score = min(h / max(1.0, 0.55 * h_img), 1.0)
        scored.append((area + 20.0 * center_score + 25.0 * height_score, cnt))
    if not scored:
        return work
    scored.sort(key=lambda item: item[0], reverse=True)
    kept = np.zeros_like(work)
    for _, cnt in scored[:4]:
        cv2.drawContours(kept, [cnt], -1, 255, thickness=-1)
    return kept


def resize_char_candidate(arr):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape != (64, 32):
        arr = cv2.resize(arr, (32, 64), interpolation=cv2.INTER_NEAREST)
    return arr


def query_variant_quality(arr):
    work = resize_char_candidate(arr)
    blurred = cv2.GaussianBlur(work, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary > 0) > 0.55:
        binary = cv2.bitwise_not(binary)
    binary = strip_character_frame_lines(binary)
    ink = float(np.mean(binary > 0))
    if ink <= 0.0:
        return {"ink": ink, "largest_area": 0.0, "largest_bbox": 0.0, "largest_extent": 0.0}

    count, _, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)
    largest_area = 0.0
    largest_bbox = 0.0
    largest_extent = 0.0
    image_area = float(binary.shape[0] * binary.shape[1])
    for idx in range(1, count):
        w = float(stats[idx, cv2.CC_STAT_WIDTH])
        h = float(stats[idx, cv2.CC_STAT_HEIGHT])
        area = float(stats[idx, cv2.CC_STAT_AREA])
        bbox_area = max(1.0, w * h)
        if area > largest_area:
            largest_area = area
            largest_bbox = bbox_area
            largest_extent = area / bbox_area

    return {
        "ink": ink,
        "largest_area": largest_area / image_area,
        "largest_bbox": largest_bbox / image_area,
        "largest_extent": largest_extent,
    }


def is_query_variant_usable(arr):
    stats = query_variant_quality(arr)
    ink = stats["ink"]
    if ink < 0.01 or ink > 0.70:
        return False
    if stats["largest_area"] >= 0.30 and stats["largest_extent"] >= 0.70:
        return False
    if stats["largest_bbox"] >= 0.32 and stats["largest_extent"] >= 0.68:
        return False
    if stats["largest_bbox"] >= 0.20 and stats["largest_extent"] >= 0.86:
        return False
    return True


def remove_tiny_foreground_components(binary, min_area=4):
    work = (binary > 0).astype(np.uint8) * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
    if count <= 1:
        return work
    cleaned = np.zeros_like(work)
    for idx in range(1, count):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        height = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if area >= min_area or height >= 8:
            cleaned[labels == idx] = 255
    return cleaned


def despeckle_char_image(arr):
    work = resize_char_candidate(arr)
    median = cv2.medianBlur(work, 3)
    _, binary = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary > 0) > 0.55:
        binary = cv2.bitwise_not(binary)
    binary = strip_character_frame_lines(binary)
    binary = remove_tiny_foreground_components(binary, min_area=5)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
    return normalize_char_image(binary)


def robust_char_query_variants(char_img):
    source = resize_char_candidate(char_img)
    source_median = cv2.medianBlur(source, 3)
    tensor_img = torch.tensor(cv2.resize(char_img, (32, 64)), dtype=torch.float32).view(1, 64, 32) / 255.0
    normalized = normalize_char_tensor(tensor_img, img_size=(32, 64))
    base = np.clip(normalized.detach().cpu().view(64, 32).numpy() * 255, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(source_median, (3, 3), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(otsu > 0) > 0.55:
        otsu = cv2.bitwise_not(otsu)

    candidates = [
        source,
        source_median,
        normalize_char_image(source),
        despeckle_char_image(source),
        base,
        normalize_char_image(base),
        normalize_char_image(otsu),
        despeckle_char_image(otsu),
        normalize_char_image(cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))),
        normalize_char_image(cv2.medianBlur(otsu, 3)),
        normalize_char_image(keep_likely_character_components(otsu)),
    ]
    unique = []
    seen = set()
    for item in candidates:
        item = resize_char_candidate(item)
        if not is_query_variant_usable(item):
            continue
        key = item.tobytes()
        if key in seen:
            continue
        seen.add(key)
        unique.append(torch.tensor(item, dtype=torch.float32).view(1, -1) / 255.0)
    if not unique:
        unique.append(normalized.view(1, -1))
    return torch.cat(unique, dim=0)


def ensemble_scores(models, q, template_labels, num_classes, template_mask=None):
    parts = []
    for model in models:
        sim_scores = compute_cached_similarity(model, q)
        if template_mask is not None:
            mask = template_mask.to(device=sim_scores.device, dtype=torch.bool)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            sim_scores = sim_scores.masked_fill(~mask, -1e9)
        scores = fast_class_free_energy_scores(sim_scores, template_labels, model.beta, num_classes, template_mask)
        parts.append(torch.log_softmax(scores, dim=-1))
    return torch.logsumexp(torch.stack(parts, dim=0), dim=0) - torch.log(q.new_tensor(float(len(parts))))


def cache_model_memories(models):
    for model in models:
        with torch.no_grad():
            model._cached_memory_for_similarity = model._memory_for_similarity()


def compute_cached_similarity(model, q):
    q_sim = model._query_for_similarity(q)
    m_sim = getattr(model, "_cached_memory_for_similarity", None)
    if m_sim is None:
        m_sim = model._memory_for_similarity()
    if model.metric == "dot":
        return torch.matmul(q_sim, m_sim.t())
    if model.metric == "manhattan":
        return -torch.cdist(q_sim, m_sim, p=1.0)
    if model.metric == "euclidean":
        return -torch.cdist(q_sim, m_sim, p=2.0)
    raise ValueError(f"Unsupported metric: {model.metric}")


def fast_class_free_energy_scores(sim_scores, template_labels, beta, num_classes, template_mask=None):
    scaled = beta * sim_scores
    labels = template_labels.to(device=sim_scores.device, dtype=torch.long)
    if template_mask is not None:
        mask = template_mask.to(device=sim_scores.device, dtype=torch.bool)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        scaled = scaled.masked_fill(~mask, -1e9)

    try:
        out = scaled.new_full((scaled.shape[0], num_classes), -1e9)
        expanded_labels = labels.view(1, -1).expand(scaled.shape[0], -1)
        return out.scatter_reduce(1, expanded_labels, scaled, reduce="amax", include_self=True)
    except Exception:
        scores = []
        for class_idx in range(num_classes):
            class_mask = labels == class_idx
            if int(class_mask.sum().item()) == 0:
                scores.append(scaled.new_full((scaled.shape[0],), -1e9))
            else:
                scores.append(torch.max(scaled[:, class_mask], dim=-1).values)
        return torch.stack(scores, dim=-1)


def max_template_scores_to_class(template_scores, template_labels, num_classes, template_mask=None):
    scores = template_scores
    labels = template_labels.to(device=scores.device, dtype=torch.long)
    if template_mask is not None:
        mask = template_mask.to(device=scores.device, dtype=torch.bool)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        scores = scores.masked_fill(~mask, -1e9)

    try:
        out = scores.new_full((scores.shape[0], num_classes), -1e9)
        expanded_labels = labels.view(1, -1).expand(scores.shape[0], -1)
        return out.scatter_reduce(1, expanded_labels, scores, reduce="amax", include_self=True)
    except Exception:
        parts = []
        for class_idx in range(num_classes):
            class_mask = labels == class_idx
            if int(class_mask.sum().item()) == 0:
                parts.append(scores.new_full((scores.shape[0],), -1e9))
            else:
                parts.append(torch.max(scores[:, class_mask], dim=-1).values)
        return torch.stack(parts, dim=-1)


def recognize_char(char_img, models, loader, template_labels, masks, device, position):
    q = robust_char_query_variants(char_img).to(device)
    template_mask = plate_position_mask(position, masks)
    with torch.no_grad():
        scores = ensemble_scores(models, q, template_labels, len(loader.idx_to_label), template_mask=template_mask)
        mean_scores = torch.logsumexp(scores, dim=0) - torch.log(scores.new_tensor(float(scores.shape[0])))
        max_scores = torch.max(scores, dim=0).values
        pooled_scores = 0.72 * mean_scores + 0.28 * max_scores
        pooled_scores = apply_position_prior(pooled_scores, loader, position=position)
        top_values, top_indices = torch.topk(pooled_scores, k=min(3, pooled_scores.numel()))
    top_labels = [loader.idx_to_label[int(idx)] for idx in top_indices.detach().cpu().tolist()]
    return top_labels[0], top_labels


def build_class_prototypes(memory, labels, num_classes, device):
    prototypes = torch.zeros((num_classes, memory.shape[1]), dtype=torch.float32, device=device)
    counts = torch.zeros(num_classes, dtype=torch.float32, device=device)
    labels = labels.to(device=device, dtype=torch.long)
    for class_idx in range(num_classes):
        mask = labels == class_idx
        if bool(mask.any().item()):
            prototypes[class_idx] = memory[mask].float().mean(dim=0)
            counts[class_idx] = float(mask.sum().item())
    prototypes = torch.nn.functional.normalize(prototypes, p=2, dim=-1)
    return prototypes, counts


def prototype_shape_scores(q, class_prototypes):
    q_norm = torch.nn.functional.normalize(q.float(), p=2, dim=-1)
    return torch.matmul(q_norm, class_prototypes.t())


def recognize_plate_chars(chars, truth_len, models, loader, template_labels, masks, device, class_prototypes=None):
    queries = []
    spans = []
    mask_rows = []
    start = 0
    for pos, char_img in enumerate(chars[:truth_len]):
        q = robust_char_query_variants(char_img)
        end = start + q.shape[0]
        queries.append(q)
        spans.append((start, end))
        position_mask = plate_position_mask(pos, masks).detach().cpu()
        mask_rows.extend([position_mask] * q.shape[0])
        start = end

    if not queries:
        return [], []

    q_batch = torch.cat(queries, dim=0).to(device)
    template_mask = torch.stack(mask_rows, dim=0).to(device)
    with torch.no_grad():
        all_scores = ensemble_scores(models, q_batch, template_labels, len(loader.idx_to_label), template_mask=template_mask)

    pred_parts = []
    top3_parts = []
    for start, end in spans:
        scores = all_scores[start:end]
        mean_scores = torch.logsumexp(scores, dim=0) - torch.log(scores.new_tensor(float(scores.shape[0])))
        max_scores = torch.max(scores, dim=0).values
        pooled_scores = 0.72 * mean_scores + 0.28 * max_scores
        if class_prototypes is not None:
            proto_scores = prototype_shape_scores(q_batch[start:end], class_prototypes)
            proto_scores = torch.max(proto_scores, dim=0).values
            proto_scores = proto_scores.masked_fill(torch.isinf(pooled_scores), -1e9)
            weight = 0.32 if len(pred_parts) == 0 else 0.22
            pooled_scores = (1.0 - weight) * pooled_scores + weight * proto_scores
        pooled_scores = apply_position_prior(pooled_scores, loader, position=len(pred_parts))
        _, top_indices = torch.topk(pooled_scores, k=min(3, pooled_scores.numel()))
        top_labels = [loader.idx_to_label[int(idx)] for idx in top_indices.detach().cpu().tolist()]
        pred_parts.append(top_labels[0])
        top3_parts.append(top_labels)
    return pred_parts, top3_parts


def build_class_mean_memory(memory, labels, num_classes, device):
    class_vectors = []
    class_labels = []
    labels = labels.to(dtype=torch.long)
    for class_idx in range(num_classes):
        mask = labels == class_idx
        if bool(mask.any().item()):
            class_vectors.append(memory[mask].float().mean(dim=0))
            class_labels.append(class_idx)
    return torch.stack(class_vectors).to(device), torch.tensor(class_labels, dtype=torch.long, device=device)


def build_method_contexts(loader, mchn_memory, template_labels, models, masks, class_prototypes, device, classic_steps=4):
    base_memory = loader.memory_matrix.float().to(device)
    base_labels = loader.labels.long().to(device)
    base_memory_norm = F.normalize(base_memory, p=2, dim=-1)
    base_masks = build_template_masks(loader, base_labels, device)
    classic_memory, classic_labels = build_class_mean_memory(
        loader.memory_matrix,
        loader.labels,
        len(loader.idx_to_label),
        device,
    )
    return {
        "MCHN": {
            "kind": "mchn",
            "models": models,
            "template_labels": template_labels,
            "masks": masks,
            "class_prototypes": class_prototypes,
            "memory": mchn_memory,
        },
        "Classic Hopfield": {
            "kind": "classic",
            "memory": classic_memory,
            "template_labels": classic_labels,
            "classic_steps": int(classic_steps),
        },
        "Nearest Neighbor": {
            "kind": "nn",
            "memory": base_memory,
            "memory_norm": base_memory_norm,
            "template_labels": base_labels,
            "masks": base_masks,
        },
    }


def score_query_variants_by_method(q_batch, position, method_ctx, loader, device):
    kind = method_ctx["kind"]
    num_classes = len(loader.idx_to_label)
    if kind == "mchn":
        template_mask = torch.stack(
            [plate_position_mask(position, method_ctx["masks"]).detach().cpu()] * q_batch.shape[0],
            dim=0,
        ).to(device)
        scores = ensemble_scores(
            method_ctx["models"],
            q_batch,
            method_ctx["template_labels"],
            num_classes,
            template_mask=template_mask,
        )
        if method_ctx.get("class_prototypes") is not None:
            mean_scores = torch.logsumexp(scores, dim=0) - torch.log(scores.new_tensor(float(scores.shape[0])))
            max_scores = torch.max(scores, dim=0).values
            pooled = 0.72 * mean_scores + 0.28 * max_scores
            proto_scores = prototype_shape_scores(q_batch, method_ctx["class_prototypes"])
            proto_scores = torch.max(proto_scores, dim=0).values
            proto_scores = proto_scores.masked_fill(torch.isinf(pooled), -1e9)
            weight = 0.32 if position == 0 else 0.22
            pooled = (1.0 - weight) * pooled + weight * proto_scores
            pooled = apply_position_prior(pooled, loader, position=position)
            return pooled
        return torch.max(scores, dim=0).values

    if kind == "nn":
        template_mask = torch.stack(
            [plate_position_mask(position, method_ctx["masks"]).detach().cpu()] * q_batch.shape[0],
            dim=0,
        ).to(device)
        q_norm = F.normalize(q_batch.float(), p=2, dim=-1)
        template_scores = torch.matmul(q_norm, method_ctx["memory_norm"].t())
        class_scores = max_template_scores_to_class(
            template_scores,
            method_ctx["template_labels"],
            num_classes,
            template_mask=template_mask,
        )
        pooled = torch.max(class_scores, dim=0).values
        return apply_position_prior(pooled, loader, position=position)

    if kind == "classic":
        template_scores = fast_classic_hopfield_scores(
            q_batch,
            method_ctx["memory"],
            steps=method_ctx.get("classic_steps", 4),
        )
        class_scores = max_template_scores_to_class(
            template_scores,
            method_ctx["template_labels"],
            num_classes,
        )
        pooled = torch.max(class_scores, dim=0).values
        class_mask = plate_position_class_mask(position, loader, device)
        pooled = pooled.masked_fill(~class_mask, -1e9)
        return apply_position_prior(pooled, loader, position=position)

    raise ValueError(f"Unsupported comparison method: {kind}")


def recognize_plate_chars_with_method(chars, truth_len, method_ctx, loader, device):
    pred_parts = []
    top3_parts = []
    for position, char_img in enumerate(chars[:truth_len]):
        q = robust_char_query_variants(char_img).to(device)
        with torch.no_grad():
            pooled_scores = score_query_variants_by_method(q, position, method_ctx, loader, device)
            top_values, top_indices = torch.topk(pooled_scores, k=min(3, pooled_scores.numel()))
        top_labels = [loader.idx_to_label[int(idx)] for idx in top_indices.detach().cpu().tolist()]
        pred_parts.append(top_labels[0])
        top3_parts.append(top_labels)
    return pred_parts, top3_parts


def prepare_plate_for_segmentation(img_bgr, pollution="none", severity=0.0):
    if img_bgr is None:
        return None
    severity = float(max(0.0, min(1.0, severity)))
    if severity <= 0.0 or pollution == "none":
        return img_bgr.copy()
    if pollution == "salt_pepper":
        kernel = 5 if severity >= 0.70 else 3
        return cv2.medianBlur(img_bgr, kernel)
    if pollution == "noise":
        denoised = cv2.medianBlur(img_bgr, 3)
        return cv2.bilateralFilter(denoised, 5, 32, 32)
    if pollution == "mixed":
        return cv2.medianBlur(img_bgr, 3)
    if pollution == "dirt" and severity >= 0.65:
        return cv2.medianBlur(img_bgr, 3)
    return img_bgr.copy()


def apply_position_prior(scores, loader, position):
    """Light plate-layout prior for Chinese blue plates.

    Position constraints already mask impossible groups. This prior only nudges
    visually ambiguous tail positions where real plates are digit-heavy; it is
    intentionally weak so MCHN similarity remains the main decision signal.
    """
    if position < 2:
        return scores
    digit_bonus_by_pos = {2: 0.18, 3: 0.12, 4: 0.08, 5: 0.14, 6: 0.08}
    bonus = digit_bonus_by_pos.get(position, 0.0)
    if bonus <= 0:
        return scores
    adjusted = scores.clone()
    for class_idx, label in loader.idx_to_label.items():
        if str(label).isdigit():
            adjusted[int(class_idx)] += bonus
    return adjusted


def build_plate_calibration_memory(rows, segmenter, loader):
    tensors = []
    labels = []
    used_plates = 0
    for item in rows:
        image_path = item["image_path"]
        if not os.path.isabs(image_path):
            image_path = os.path.join(ROOT_DIR, image_path)
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        plate = cv2.resize(img, (PlateSegmenter.PLATE_W, PlateSegmenter.PLATE_H))
        chars = segmenter.segment_characters(plate)
        truth = item["plate_text"]
        if len(chars) < len(truth):
            continue
        added = 0
        for char_img, label in zip(chars[: len(truth)], truth):
            if label not in loader.label_to_idx:
                continue
            tensor = torch.tensor(cv2.resize(char_img, (32, 64)), dtype=torch.float32).view(1, 64, 32) / 255.0
            tensor = normalize_char_tensor(tensor, img_size=(32, 64)).view(-1)
            tensors.append(tensor)
            labels.append(int(loader.label_to_idx[label]))
            added += 1
        if added == len(truth):
            used_plates += 1
    if not tensors:
        return None, None, 0
    return torch.stack(tensors, dim=0), torch.tensor(labels, dtype=torch.long), used_plates


def iter_with_progress(items, enabled=True, desc=None):
    if enabled and tqdm is not None:
        return tqdm(items, desc=desc, leave=False)
    return items


def evaluation_grid(pollutions, severities):
    grid = []
    for pollution in pollutions:
        current_severities = [0.0] if pollution == "none" else severities
        for severity in current_severities:
            grid.append((pollution, float(severity)))
    return grid


def resolve_pollutions(pollution_arg):
    value = str(pollution_arg).strip()
    if value == "all":
        return POLLUTIONS
    if value in {"core", "main", "plate_core"}:
        return CORE_PLATE_POLLUTIONS
    if "," in value:
        pollutions = [item.strip() for item in value.split(",") if item.strip()]
    else:
        pollutions = [value]

    allowed = set(POLLUTIONS + ["mixed"])
    unknown = [item for item in pollutions if item not in allowed]
    if unknown:
        raise ValueError(
            f"Unsupported pollution(s): {', '.join(unknown)}. "
            f"Use one of: all, core, mixed, {', '.join(POLLUTIONS)}; "
            "or a comma-separated list such as noise,salt_pepper,blur,mask,dirt,fog."
        )
    return pollutions


def rng_randint(rng, low, high):
    low = int(low)
    high = int(high)
    if high < low:
        high = low
    if hasattr(rng, "integers"):
        return int(rng.integers(low, high + 1))
    return int(rng.randint(low, high))


def rng_sample(rng, items, count):
    count = int(max(0, min(count, len(items))))
    if count <= 0:
        return []
    if hasattr(rng, "choice"):
        return [items[int(idx)] for idx in rng.choice(len(items), size=count, replace=False)]
    return rng.sample(items, count)


def stable_text_seed(text):
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(text)))


def apply_plate_pollution(img, pollution, severity, rng):
    severity = float(max(0.0, min(1.0, severity)))
    if pollution == "none" or severity <= 0.0:
        return img.copy()
    if pollution == "mixed":
        choices = ["mask", "noise", "salt_pepper", "blur", "fog", "dirt", "affine"]
        count = 1 if severity < 0.35 else 2 if severity < 0.7 else 3
        out = img.copy()
        for name in rng_sample(rng, choices, count):
            out = apply_plate_pollution(out, name, severity, rng)
        return out

    out = img.copy()
    h, w = out.shape[:2]
    if pollution == "mask":
        for _ in range(1 + int(3 * severity)):
            bw = rng_randint(rng, max(6, int(w * 0.06)), max(8, int(w * (0.12 + 0.18 * severity))))
            bh = rng_randint(rng, max(5, int(h * 0.10)), max(6, int(h * (0.18 + 0.22 * severity))))
            x = rng_randint(rng, 0, max(0, w - bw))
            y = rng_randint(rng, 0, max(0, h - bh))
            color = (0, 0, 0) if rng.random() < 0.65 else (255, 255, 255)
            cv2.rectangle(out, (x, y), (x + bw, y + bh), color, thickness=-1)
        return out
    if pollution == "noise":
        sigma = 8 + 55 * severity
        noise = rng.normal(0, sigma, out.shape).astype(np.float32)
        return np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if pollution == "salt_pepper":
        prob = 0.005 + 0.12 * severity
        mask = rng.random(out.shape[:2])
        out[mask < prob / 2] = 0
        out[(mask >= prob / 2) & (mask < prob)] = 255
        return out
    if pollution == "blur":
        kernel = int(3 + 10 * severity)
        kernel = kernel + 1 if kernel % 2 == 0 else kernel
        return cv2.GaussianBlur(out, (kernel, kernel), 0)
    if pollution == "fog":
        fog = np.full_like(out, 255)
        return cv2.addWeighted(out, 1.0 - (0.12 + 0.55 * severity), fog, 0.12 + 0.55 * severity, 0)
    if pollution == "dirt":
        for _ in range(1 + int(5 * severity)):
            radius = rng_randint(rng, max(3, int(h * 0.04)), max(4, int(h * (0.08 + 0.15 * severity))))
            x = rng_randint(rng, 0, max(0, w - 1))
            y = rng_randint(rng, 0, max(0, h - 1))
            color = tuple(rng_randint(rng, 20, 89) for _ in range(3))
            cv2.circle(out, (x, y), radius, color, thickness=-1)
        return out
    if pollution == "affine":
        angle = rng.uniform(-8, 8) * severity
        shear = rng.uniform(-0.12, 0.12) * severity
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        matrix[0, 1] += shear
        return cv2.warpAffine(out, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    raise ValueError(f"Unsupported pollution: {pollution}")


def read_labels(labels_csv):
    rows = []
    with open(labels_csv, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({"image_path": row["image_path"], "plate_text": row["plate_text"].strip()})
    return rows


def edit_distance(a, b):
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, start=1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (0 if ca == cb else 1))
            prev = cur
    return dp[-1]


def save_debug_case(output_dir, debug_index, pollution, severity, plate, chars):
    debug_dir = os.path.join(output_dir, "cropped_plate_debug")
    os.makedirs(debug_dir, exist_ok=True)
    plate_vis = cv2.resize(plate, (400, 120))
    if plate_vis.ndim == 2:
        plate_vis = cv2.cvtColor(plate_vis, cv2.COLOR_GRAY2BGR)
    char_tiles = []
    for char_img in chars[:7]:
        tile = cv2.resize(char_img, (48, 96), interpolation=cv2.INTER_NEAREST)
        if tile.ndim == 2:
            tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        char_tiles.append(tile)
    while len(char_tiles) < 7:
        char_tiles.append(np.zeros((96, 48, 3), dtype=np.uint8))
    separator = np.full((96, 4, 3), 230, dtype=np.uint8)
    char_row = char_tiles[0]
    for tile in char_tiles[1:]:
        char_row = np.hstack([char_row, separator, tile])
    char_row = cv2.copyMakeBorder(char_row, 8, 0, 0, max(0, plate_vis.shape[1] - char_row.shape[1]), cv2.BORDER_CONSTANT, value=(255, 255, 255))
    montage = np.vstack([plate_vis, char_row[:, : plate_vis.shape[1]]])
    filename = f"debug_{debug_index:03d}_{pollution}_{severity:.2f}.jpg"
    cv2.imwrite(os.path.join(debug_dir, filename), montage)
    return os.path.join("cropped_plate_debug", filename)


def save_method_comparison_outputs(output_dir, method_stats, method_detail_rows):
    summary_path = os.path.join(output_dir, "cropped_plate_method_comparison.csv")
    detail_path = os.path.join(output_dir, "cropped_plate_method_char_details.csv")
    summary_rows = []
    for (pollution, severity, method), stat in sorted(method_stats.items()):
        images = max(int(stat["images"]), 1)
        char_total = max(int(stat["char_total"]), 1)
        summary_rows.append(
            {
                "pollution": pollution,
                "severity": severity,
                "method": method,
                "image_count": int(stat["images"]),
                "char_accuracy": stat["char_ok"] / char_total,
                "top3_accuracy": stat["top3"] / char_total,
                "plate_accuracy": stat["plate"] / images,
                "mean_edit_distance": stat["edit"] / images,
            }
        )

    with open(summary_path, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "pollution",
            "severity",
            "method",
            "image_count",
            "char_accuracy",
            "top3_accuracy",
            "plate_accuracy",
            "mean_edit_distance",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    key: (
                        f"{value:.4f}"
                        if isinstance(value, float) and np.isfinite(value)
                        else "nan"
                        if isinstance(value, float)
                        else value
                    )
                    for key, value in row.items()
                }
            )

    if method_detail_rows:
        with open(detail_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(method_detail_rows[0].keys()))
            writer.writeheader()
            writer.writerows(method_detail_rows)

    plot_method_comparison(output_dir, summary_rows)
    return summary_path, detail_path if method_detail_rows else None


def plot_method_comparison(output_dir, summary_rows):
    if not summary_rows:
        return
    methods = []
    for row in summary_rows:
        if row["method"] not in methods:
            methods.append(row["method"])
    pollutions = []
    for row in summary_rows:
        label = f"{row['pollution']} {row['severity']}"
        if label not in pollutions:
            pollutions.append(label)

    for metric, ylabel, filename in (
        ("char_accuracy", "Character accuracy (%)", "cropped_plate_method_char_accuracy.png"),
        ("plate_accuracy", "Plate accuracy (%)", "cropped_plate_method_plate_accuracy.png"),
    ):
        x = np.arange(len(pollutions), dtype=float)
        width = 0.78 / max(1, len(methods))
        plt.figure(figsize=(max(9, 1.2 * len(pollutions)), 5.5))
        for idx, method in enumerate(methods):
            values = []
            for pollution_label in pollutions:
                match = next(
                    (
                        row
                        for row in summary_rows
                        if row["method"] == method and f"{row['pollution']} {row['severity']}" == pollution_label
                    ),
                    None,
                )
                value = float("nan") if match is None else float(match[metric]) * 100.0
                values.append(value)
            offsets = x - 0.39 + width / 2 + idx * width
            plt.bar(offsets, values, width=width, label=method)
        plt.ylim(0, 105)
        plt.ylabel(ylabel)
        plt.xticks(x, pollutions, rotation=25, ha="right")
        plt.grid(axis="y", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()


def evaluate(args):
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Cropped-plate MCHN evaluation, device={device}")

    loader = TemplateLoader(
        [os.path.join(args.data_dir, "chars2"), os.path.join(args.data_dir, "charsChinese")],
        img_size=(32, 64),
        cache_path=os.path.join(args.output_dir, "cropped_plate_template_cache_32x64.pt"),
    )
    segmenter = PlateSegmenter()
    rows = read_labels(args.labels_csv)
    if args.calibration_count > 0:
        calibration_rows = rows[: args.calibration_count]
        eval_rows = rows[args.calibration_count :]
        plate_memory, plate_labels, used_plates = build_plate_calibration_memory(calibration_rows, segmenter, loader)
        if plate_memory is not None:
            loader.memory_matrix = torch.cat([loader.memory_matrix, plate_memory], dim=0)
            loader.labels = torch.cat([loader.labels, plate_labels], dim=0)
            print(
                f"Added cropped-plate calibration memory: {plate_memory.shape[0]} characters "
                f"from {used_plates}/{len(calibration_rows)} plates. Evaluation excludes calibration plates."
            )
        else:
            print("Warning: no cropped-plate calibration characters were added.")
        rows = eval_rows
    if args.max_images:
        rows = rows[: args.max_images]
    if args.memory_augmentation == "full":
        memory, template_labels = augment_hopfield_memory(loader.memory_matrix, loader.labels)
    else:
        memory, template_labels = loader.memory_matrix, loader.labels
    memory = memory.to(device)
    template_labels = template_labels.to(device)
    models = build_hopfield_ensemble(memory, device)
    cache_model_memories(models)
    class_prototypes, _ = build_class_prototypes(memory, template_labels, len(loader.idx_to_label), device)
    masks = build_template_masks(loader, template_labels, device)
    method_contexts = build_method_contexts(
        loader,
        memory,
        template_labels,
        models,
        masks,
        class_prototypes,
        device,
        classic_steps=args.classic_steps,
    )
    comparison_methods = [item.strip() for item in args.compare_methods.split(",") if item.strip()]
    unknown_methods = [name for name in comparison_methods if name not in method_contexts]
    if unknown_methods:
        raise ValueError(f"Unsupported compare method(s): {', '.join(unknown_methods)}")
    if not rows:
        raise RuntimeError("No evaluation plates left after calibration/max-images filtering.")

    pollutions = resolve_pollutions(args.pollution)
    severities = [float(item) for item in args.severities.split(",")]
    grid = evaluation_grid(pollutions, severities)
    detail_rows = []
    char_rows = []
    method_compare_detail_rows = []
    debug_detail_rows = []
    position_stats = defaultdict(lambda: {"correct": 0, "total": 0, "top3": 0})
    method_stats = defaultdict(
        lambda: {
            "images": 0,
            "char_ok": 0,
            "char_total": 0,
            "top3": 0,
            "plate": 0,
            "edit": 0,
        }
    )
    debug_saved = 0

    print(f"Evaluation plan: {len(rows)} plates x {len(grid)} pollution/severity settings.")
    print(f"Pollutions: {', '.join(pollutions)}")
    print(f"Comparison methods: {', '.join(comparison_methods)}")
    if args.pollution == "all":
        print("Note: pollution=none is evaluated once at severity=0.00; other pollution types keep all severity levels.")

    for pollution, severity in grid:
        total_chars = correct_chars = top3_chars = plate_correct = edit_sum = 0
        desc = f"{pollution} {severity:.2f}"
        for image_idx, item in enumerate(iter_with_progress(rows, enabled=not args.no_progress, desc=desc), start=1):
            image_path = item["image_path"]
            if not os.path.isabs(image_path):
                image_path = os.path.join(ROOT_DIR, image_path)
            truth = item["plate_text"]
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(image_path)
            pollution_rng = np.random.default_rng(args.seed + stable_text_seed(pollution) * 1009 + image_idx * 1000003)
            polluted = apply_plate_pollution(img, pollution, severity, pollution_rng)
            seg_input = prepare_plate_for_segmentation(polluted, pollution, severity)
            plate = cv2.resize(seg_input, (PlateSegmenter.PLATE_W, PlateSegmenter.PLATE_H))
            chars = segmenter.segment_characters(plate)
            pred_parts = []
            top3_parts = []
            correct_flags = []
            top3_flags = []
            top3_ok = 0
            char_ok = 0
            comparison_outputs = {}
            for method_name in comparison_methods:
                method_preds, method_top3 = recognize_plate_chars_with_method(
                    chars,
                    len(truth),
                    method_contexts[method_name],
                    loader,
                    device,
                )
                comparison_outputs[method_name] = (method_preds, method_top3)

                method_char_ok = 0
                method_top3_ok = 0
                for pos, (pred, top3) in enumerate(zip(method_preds, method_top3)):
                    correct = pred == truth[pos]
                    in_top3 = truth[pos] in top3
                    method_char_ok += int(correct)
                    method_top3_ok += int(in_top3)
                    method_compare_detail_rows.append(
                        {
                            "pollution": pollution,
                            "severity": f"{severity:.2f}",
                            "method": method_name,
                            "image_path": os.path.relpath(image_path, ROOT_DIR),
                            "position": pos + 1,
                            "truth": truth[pos],
                            "pred": pred,
                            "top3": "/".join(str(item) for item in top3),
                            "correct": int(correct),
                            "top3_hit": int(in_top3),
                            "segmented_count": len(chars),
                        }
                    )
                method_pred_text = "".join(method_preds)
                method_key = (pollution, f"{severity:.2f}", method_name)
                method_stats[method_key]["images"] += 1
                method_stats[method_key]["char_ok"] += method_char_ok
                method_stats[method_key]["char_total"] += len(truth)
                method_stats[method_key]["top3"] += method_top3_ok
                method_stats[method_key]["plate"] += int(method_pred_text == truth)
                method_stats[method_key]["edit"] += edit_distance(method_pred_text, truth)

            batch_preds, batch_top3 = comparison_outputs.get("MCHN", ([], []))
            for pos, (pred, top3) in enumerate(zip(batch_preds, batch_top3)):
                pred_parts.append(pred)
                top3_parts.append("/".join(str(item) for item in top3))
                correct = pred == truth[pos]
                in_top3 = truth[pos] in top3
                correct_flags.append("1" if correct else "0")
                top3_flags.append("1" if in_top3 else "0")
                char_ok += int(correct)
                top3_ok += int(in_top3)
                key = (pollution, severity, pos + 1)
                position_stats[key]["correct"] += int(correct)
                position_stats[key]["top3"] += int(in_top3)
                position_stats[key]["total"] += 1
                char_rows.append(
                    {
                        "pollution": pollution,
                        "severity": f"{severity:.2f}",
                        "image_path": os.path.relpath(image_path, ROOT_DIR),
                        "position": pos + 1,
                        "truth": truth[pos],
                        "pred": pred,
                        "top3": "/".join(str(item) for item in top3),
                        "correct": int(correct),
                        "top3_hit": int(in_top3),
                        "segmented_count": len(chars),
                    }
                )
            pred_text = "".join(pred_parts)
            total_chars += len(truth)
            correct_chars += char_ok
            top3_chars += top3_ok
            plate_correct += int(pred_text == truth)
            dist = edit_distance(pred_text, truth)
            edit_sum += dist
            if args.debug_samples > 0 and debug_saved < args.debug_samples and (pred_text != truth or len(chars) != len(truth)):
                debug_image = save_debug_case(args.output_dir, debug_saved + 1, pollution, severity, plate, chars)
                debug_saved += 1
                for pos, (truth_char, pred_char, top3_text, correct, top3_hit) in enumerate(
                    zip(truth, pred_parts, top3_parts, correct_flags, top3_flags),
                    start=1,
                ):
                    debug_detail_rows.append(
                        {
                            "debug_index": debug_saved,
                            "debug_image": debug_image,
                            "pollution": pollution,
                            "severity": f"{severity:.2f}",
                            "image_path": os.path.relpath(image_path, ROOT_DIR),
                            "plate_truth": truth,
                            "plate_pred": pred_text,
                            "position": pos,
                            "truth": truth_char,
                            "pred": pred_char,
                            "top3": top3_text,
                            "correct": int(correct),
                            "top3_hit": int(top3_hit),
                            "segmented_count": len(chars),
                        }
                    )
            detail_rows.append(
                {
                    "pollution": pollution,
                    "severity": f"{severity:.2f}",
                    "image_path": os.path.relpath(image_path, ROOT_DIR),
                    "truth": truth,
                    "pred": pred_text,
                    "pred_chars": " ".join(pred_parts),
                    "top3_by_position": " | ".join(top3_parts),
                    "correct_by_position": "".join(correct_flags),
                    "top3_by_position_hit": "".join(top3_flags),
                    "segmented_count": len(chars),
                    "char_correct": char_ok,
                    "char_total": len(truth),
                    "top3_correct": top3_ok,
                    "plate_correct": int(pred_text == truth),
                    "edit_distance": dist,
                }
            )
        image_count = len(rows)
        print(
            f"{pollution:12s} severity={severity:.2f} | "
            f"char={correct_chars / total_chars * 100:.2f}% | "
            f"plate={plate_correct / image_count * 100:.2f}% | "
            f"top3={top3_chars / total_chars * 100:.2f}%"
        )

    summary_path = os.path.join(args.output_dir, "cropped_plate_summary.csv")
    details_path = os.path.join(args.output_dir, "cropped_plate_details.csv")
    char_details_path = os.path.join(args.output_dir, "cropped_plate_char_details.csv")
    debug_details_path = os.path.join(args.output_dir, "cropped_plate_debug_details.csv")
    position_path = os.path.join(args.output_dir, "cropped_plate_position_accuracy.csv")
    with open(details_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)
    with open(char_details_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(char_rows[0].keys()))
        writer.writeheader()
        writer.writerows(char_rows)
    if debug_detail_rows:
        with open(debug_details_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(debug_detail_rows[0].keys()))
            writer.writeheader()
            writer.writerows(debug_detail_rows)

    summary = defaultdict(lambda: {"images": 0, "char_ok": 0, "char_total": 0, "top3": 0, "plate": 0, "edit": 0})
    for row in detail_rows:
        key = (row["pollution"], row["severity"])
        summary[key]["images"] += 1
        summary[key]["char_ok"] += int(row["char_correct"])
        summary[key]["char_total"] += int(row["char_total"])
        summary[key]["top3"] += int(row["top3_correct"])
        summary[key]["plate"] += int(row["plate_correct"])
        summary[key]["edit"] += int(row["edit_distance"])
    with open(summary_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pollution", "severity", "image_count", "char_accuracy", "top3_accuracy", "plate_accuracy", "mean_edit_distance"])
        for (pollution, severity), stat in sorted(summary.items()):
            writer.writerow(
                [
                    pollution,
                    severity,
                    stat["images"],
                    f"{stat['char_ok'] / stat['char_total']:.4f}",
                    f"{stat['top3'] / stat['char_total']:.4f}",
                    f"{stat['plate'] / stat['images']:.4f}",
                    f"{stat['edit'] / stat['images']:.4f}",
                ]
            )
    with open(position_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pollution", "severity", "position", "correct", "top3_correct", "total", "accuracy", "top3_accuracy"])
        for (pollution, severity, position), stat in sorted(position_stats.items()):
            writer.writerow(
                [
                    pollution,
                    f"{severity:.2f}",
                    position,
                    stat["correct"],
                    stat["top3"],
                    stat["total"],
                    f"{stat['correct'] / stat['total']:.4f}",
                    f"{stat['top3'] / stat['total']:.4f}",
                ]
            )
    print(f"Saved summary: {summary_path}")
    print(f"Saved details: {details_path}")
    print(f"Saved char details: {char_details_path}")
    if debug_detail_rows:
        print(f"Saved debug details: {debug_details_path}")
    print(f"Saved position accuracy: {position_path}")
    method_summary_path, method_detail_path = save_method_comparison_outputs(
        args.output_dir,
        method_stats,
        method_compare_detail_rows,
    )
    print(f"Saved method comparison: {method_summary_path}")
    if method_detail_path:
        print(f"Saved method char details: {method_detail_path}")
    if args.debug_samples > 0:
        print(f"Saved debug montages: {os.path.join(args.output_dir, 'cropped_plate_debug')}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MCHN on clean cropped plates with synthetic pollution.")
    parser.add_argument("--labels-csv", default="./plate_eval/labels.csv")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--run-name", default="cropped_plate_eval", help="Subfolder under output-dir for cropped-plate evaluation artifacts.")
    parser.add_argument(
        "--pollution",
        default="core",
        help=(
            "Pollution to evaluate. Use 'core' for noise,salt_pepper,blur,mask,dirt,fog; "
            "use 'all' for every built-in single pollution including affine; or pass a comma-separated list."
        ),
    )
    parser.add_argument("--severities", default="0.6")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument(
        "--calibration-count",
        type=int,
        default=0,
        help="Use the first N clean cropped plates as MCHN real-domain calibration memory and exclude them from evaluation.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--debug-samples", type=int, default=10, help="Save this many failed plate segmentation/recognition montages.")
    parser.add_argument(
        "--compare-methods",
        default="MCHN,Classic Hopfield,Nearest Neighbor",
        help="Comma-separated recognition methods evaluated on the same segmented cropped plates.",
    )
    parser.add_argument("--classic-steps", type=int, default=4, help="Update steps for the classic Hopfield comparison baseline.")
    parser.add_argument(
        "--memory-augmentation",
        default="none",
        choices=["none", "full"],
        help="Use full MCHN template augmentation. Default none keeps cropped-plate comparison fast and memory-light.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
