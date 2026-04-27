import os
from collections import defaultdict

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

from dataset.lp_dataset import CharPolluter, TemplateLoader, normalize_char_tensor
from models.mchn import ModernHopfieldNetwork
from utils.image_processing import LPRPipeline


def zh(text):
    texts = {
        "title": "\u73b0\u4ee3 Hopfield \u7f51\u7edc\u6c61\u67d3\u8f66\u724c\u5b57\u7b26\u8bc6\u522b",
        "plate_tab": "\u6574\u8f66\u56fe\u7247\u8bc6\u522b",
        "char_tab": "\u5355\u5b57\u7b26\u6c61\u67d3\u6d4b\u8bd5",
        "upload_plate": "\u4e0a\u4f20\u8f66\u724c\u6216\u6574\u8f66\u56fe\u7247",
        "run_lpr": "\u63d0\u4ea4\u8f66\u724c\u8bc6\u522b",
        "overall": "\u603b\u4f53\u8bc6\u522b\u7ed3\u679c",
        "plate_img": "\u8f66\u724c\u5b9a\u4f4d\u4e0e\u900f\u89c6\u77eb\u6b63",
        "char_gallery": "\u5b57\u7b26\u5207\u5272\u7ed3\u679c",
        "debug_gallery": "MCHN \u5b9e\u9645\u8f93\u5165\u4e0e\u5339\u914d\u6a21\u677f",
        "char_table": "\u9010\u5b57\u7b26\u8bc6\u522b\u5206\u6790",
        "choose_class": "\u9009\u62e9\u5b57\u7b26\u7c7b\u522b",
        "choose_sample": "\u4ece\u6587\u4ef6\u5939\u9009\u62e9\u7070\u5ea6\u5b57\u7b26\u56fe",
        "pollution": "\u6c61\u67d3\u7c7b\u578b",
        "severity": "\u6c61\u67d3\u5f3a\u5ea6",
        "seed": "\u968f\u673a\u79cd\u5b50",
        "run_char": "\u8fd0\u884c\u5355\u5b57\u7b26 MCHN \u6d4b\u8bd5",
        "original": "\u539f\u59cb\u5b57\u7b26",
        "polluted": "\u6c61\u67d3\u540e\u5b57\u7b26",
        "matched": "\u5339\u914d\u8bb0\u5fc6\u6a21\u677f",
        "char_result": "\u5355\u5b57\u7b26\u8bc6\u522b\u7ed3\u679c",
        "detector_hint": "\u4e3b\u8981\u4f7f\u7528 YOLO/ONNX \u4e24\u9636\u6bb5\u6d41\u7a0b\uff1aPLATE_DETECTOR_WEIGHTS \u8d1f\u8d23\u8f66\u724c\u5b9a\u4f4d\uff0cCHAR_DETECTOR_WEIGHTS \u8d1f\u8d23\u5b57\u7b26\u6846\u68c0\u6d4b\uff0c\u7136\u540e\u7531 MCHN \u8bc6\u522b\u5b57\u7b26\u7c7b\u522b\u3002OpenCV \u5206\u5272\u4ec5\u4f5c\u4e3a\u5907\u9009\u3002",
    }
    return texts[text]


def is_kaggle_runtime():
    return os.path.exists("/kaggle/working") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def default_cache_path():
    cache_dir = "/kaggle/working/mchn_cache" if is_kaggle_runtime() else os.path.join(os.getcwd(), "results", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, "template_cache_32x64.pt")


def build_template_masks(loader, labels, device):
    num_templates = int(labels.shape[0])
    chinese_mask = torch.zeros(num_templates, dtype=torch.bool, device=device)
    alnum_mask = torch.zeros(num_templates, dtype=torch.bool, device=device)
    plate_tail_mask = torch.zeros(num_templates, dtype=torch.bool, device=device)
    letter_mask = torch.zeros(num_templates, dtype=torch.bool, device=device)
    digit_mask = torch.zeros(num_templates, dtype=torch.bool, device=device)
    for idx, label_idx in enumerate(labels.tolist()):
        label = loader.idx_to_label[int(label_idx)]
        if is_chinese_label(label):
            chinese_mask[idx] = True
            alnum_mask[idx] = False
        elif len(label) == 1 and "A" <= label <= "Z":
            letter_mask[idx] = True
        elif label.isdigit() and len(label) == 1:
            digit_mask[idx] = True
    alnum_mask = letter_mask | digit_mask
    plate_tail_mask = alnum_mask.clone()
    for idx, label_idx in enumerate(labels.tolist()):
        label = loader.idx_to_label[int(label_idx)]
        if label in {"I", "O"}:
            plate_tail_mask[idx] = False
    return chinese_mask, alnum_mask, plate_tail_mask, letter_mask, digit_mask


def is_chinese_label(label):
    return (len(label) == 1 and "\u4e00" <= label <= "\u9fff") or str(label).startswith("zh_")


def augment_hopfield_memory(memory, labels, img_h=64, img_w=32):
    """Add label-preserving template variants to the associative memory.

    This keeps recognition as modern Hopfield retrieval, but gives each class
    memories with stroke thickness, small shifts and blur closer to real plates.
    """
    if memory.numel() == 0:
        return memory, labels
    imgs = memory.float().view(-1, 1, img_h, img_w)
    variants = [imgs]

    variants.append(torch.roll(imgs, shifts=1, dims=2))
    variants.append(torch.roll(imgs, shifts=-1, dims=2))
    variants.append(torch.roll(imgs, shifts=1, dims=3))
    variants.append(torch.roll(imgs, shifts=-1, dims=3))

    dilated = F.max_pool2d(imgs, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-imgs, kernel_size=3, stride=1, padding=1)
    blurred = F.avg_pool2d(F.pad(imgs, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)
    variants.extend([dilated, eroded, 0.65 * imgs + 0.35 * blurred])
    variants.extend(affine_memory_variants(imgs))

    stacked = torch.cat([v.clamp(0.0, 1.0).view(memory.shape[0], -1) for v in variants], dim=0)
    expanded_labels = labels.repeat(len(variants))
    return stacked.contiguous(), expanded_labels.contiguous()


def affine_memory_variants(imgs):
    """Template variants for real plate crops.

    End-to-end segmentation often makes characters slightly slanted or
    horizontally squeezed. These deterministic variants keep the method
    single-shot while narrowing the gap between clean folders and real crops.
    """
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
        else:
            scores.append(torch.max(scaled[:, class_mask], dim=-1).values)
    return torch.stack(scores, dim=-1)


def build_hopfield_ensemble(memory, device):
    return [
        ModernHopfieldNetwork(memory, beta=28.0, metric="dot", normalize=True, feature_mode="binary").to(device),
        ModernHopfieldNetwork(memory, beta=32.0, metric="dot", normalize=True, feature_mode="centered").to(device),
        ModernHopfieldNetwork(memory, beta=30.0, metric="dot", normalize=True, feature_mode="binary_centered").to(device),
        ModernHopfieldNetwork(memory, beta=26.0, metric="dot", normalize=True, feature_mode="hybrid_shape").to(device),
        ModernHopfieldNetwork(memory, beta=30.0, metric="dot", normalize=True, feature_mode="profile").to(device),
        ModernHopfieldNetwork(memory, beta=2.5, metric="euclidean", normalize=False, feature_mode="profile").to(device),
    ]


def select_best_template_in_class(sim_scores, template_labels, class_idx):
    labels = template_labels.to(sim_scores.device)
    mask = labels == int(class_idx)
    return torch.argmax(sim_scores[0].masked_fill(~mask, -1e9)).item()


def ensemble_scores(models, q, template_labels, num_classes, template_mask=None):
    log_prob_parts = []
    first_sim = None
    first_retrieved = None
    for model in models:
        retrieved, _, sim_scores = model(q, template_mask=template_mask, return_similarity=True)
        scores = class_free_energy_scores(sim_scores, template_labels, model.beta, num_classes, template_mask)
        log_probs = torch.log_softmax(scores, dim=-1)
        log_prob_parts.append(log_probs)
        if first_sim is None:
            first_sim = sim_scores
            first_retrieved = retrieved
    fused = torch.logsumexp(torch.stack(log_prob_parts, dim=0), dim=0) - torch.log(
        first_sim.new_tensor(float(len(log_prob_parts)))
    )
    return fused, first_sim, first_retrieved


def tensor_to_rgb_image(tensor):
    arr = tensor.detach().cpu().view(64, 32).numpy()
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)


def normalize_char_image(arr):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape != (64, 32):
        arr = cv2.resize(arr, (32, 64), interpolation=cv2.INTER_NEAREST)
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary > 0) > 0.55:
        binary = cv2.bitwise_not(binary)

    ys, xs = np.where(binary > 0)
    if len(xs) < 6 or len(ys) < 6:
        return binary

    coords = np.column_stack([xs, ys]).astype(np.float32)
    rect = cv2.minAreaRect(coords)
    angle = rect[2]
    rw, rh = rect[1]
    if rw > 1 and rh > 1:
        if rw > rh:
            angle += 90.0
        if angle > 45.0:
            angle -= 90.0
        if angle < -45.0:
            angle += 90.0
        if 2.0 <= abs(angle) <= 22.0:
            matrix = cv2.getRotationMatrix2D((16, 32), angle, 1.0)
            binary = cv2.warpAffine(binary, matrix, (32, 64), flags=cv2.INTER_NEAREST, borderValue=0)

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


def affine_char_variants(tensor):
    base = tensor.detach().cpu().view(64, 32).numpy()
    base = normalize_char_image(np.clip(base * 255, 0, 255))
    variants = [base]
    for angle in (-14, -8, 8, 14):
        matrix = cv2.getRotationMatrix2D((16, 32), angle, 1.0)
        variants.append(normalize_char_image(cv2.warpAffine(base, matrix, (32, 64), flags=cv2.INTER_NEAREST, borderValue=0)))
    for shear in (-0.16, -0.08, 0.08, 0.16):
        matrix = np.array([[1.0, shear, -shear * 32.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        variants.append(normalize_char_image(cv2.warpAffine(base, matrix, (32, 64), flags=cv2.INTER_NEAREST, borderValue=0)))

    unique = []
    seen = set()
    for item in variants:
        key = item.tobytes()
        if key in seen:
            continue
        seen.add(key)
        unique.append(torch.tensor(item, dtype=torch.float32).view(1, -1) / 255.0)
    return torch.cat(unique, dim=0)


def robust_char_query_variants(tensor_img):
    if tensor_img.dim() == 2:
        tensor_img = tensor_img.unsqueeze(0)
    normalized = normalize_char_tensor(tensor_img, img_size=(32, 64))
    base = np.clip(normalized.detach().cpu().view(64, 32).numpy() * 255, 0, 255).astype(np.uint8)

    variants = [base, normalize_char_image(base)]
    blurred = cv2.GaussianBlur(base, (3, 3), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(otsu > 0) > 0.55:
        otsu = cv2.bitwise_not(otsu)
    variants.append(normalize_char_image(otsu))

    for scale, offset in ((1.25, -18), (1.45, -32)):
        contrast = np.clip(base.astype(np.float32) * scale + offset, 0, 255).astype(np.uint8)
        variants.append(normalize_char_image(contrast))

    kernels = [cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))]
    variants.append(normalize_char_image(cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernels[0])))
    variants.append(normalize_char_image(cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernels[0])))
    variants.append(normalize_char_image(cv2.medianBlur(otsu, 3)))
    variants.append(normalize_char_image(_keep_likely_character_components(otsu)))

    unique = []
    seen = set()
    for item in variants:
        if item is None or item.size == 0:
            continue
        item = normalize_char_image(item)
        ink = float(np.mean(item > 0))
        if ink < 0.01 or ink > 0.70:
            continue
        key = item.tobytes()
        if key in seen:
            continue
        seen.add(key)
        unique.append(torch.tensor(item, dtype=torch.float32).view(1, -1) / 255.0)
    if not unique:
        unique.append(normalized.view(1, -1))
    return torch.cat(unique, dim=0)


def _keep_likely_character_components(binary):
    work = (binary > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return work
    h_img, w_img = work.shape[:2]
    scored = []
    center_x = w_img / 2.0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area < max(3, 0.004 * h_img * w_img):
            continue
        if h < 0.10 * h_img and area < 0.04 * h_img * w_img:
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


def collect_full_car_samples():
    roots = [
        "./data/full_cars/ccpd_base",
        "./data/full_cars/ccpd_blur",
        "./data/full_cars/ccpd_challenge",
        "./data/full_cars/ccpd_weather",
    ]
    choices = []
    for root in roots:
        if not os.path.exists(root):
            continue
        for name in sorted(os.listdir(root)):
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                path = os.path.join(root, name)
                choices.append((os.path.relpath(path, os.getcwd()), path))
    return choices


print("Loading MCHN memory and YOLO-first LPR pipeline...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = TemplateLoader(["./data/chars2", "./data/charsChinese"], img_size=(32, 64), cache_path=default_cache_path())
if loader.memory_matrix.shape[0] == 0:
    raise RuntimeError("Template memory is empty. Please check ./data/chars2 and ./data/charsChinese.")

memory, template_labels = augment_hopfield_memory(loader.memory_matrix, loader.labels)
memory = memory.to(device)
template_labels = template_labels.to(device)
mchn_models = build_hopfield_ensemble(memory, device)
pipeline = LPRPipeline()
chinese_mask, alnum_mask, plate_tail_mask, letter_mask, digit_mask = build_template_masks(loader, template_labels, device)
pollution_choices = ["none", "mask", "noise", "salt_pepper", "blur", "fog", "dirt", "affine", "mixed"]
full_car_samples = collect_full_car_samples()

label_to_paths = defaultdict(list)
for idx, path in enumerate(loader.template_paths):
    label = loader.idx_to_label[int(loader.labels[idx])]
    rel = os.path.relpath(path, os.getcwd()) if os.path.exists(path) else path
    label_to_paths[label].append((rel, path))
class_choices = list(label_to_paths.keys())

pil_transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 32)),
        transforms.ToTensor(),
    ]
)


def recognize_tensor(tensor, template_mask=None, return_debug=False):
    if tensor.dim() == 2 and tensor.shape[-1] == 64 * 32:
        tensor_img = tensor.view(-1, 1, 64, 32)[0]
    elif tensor.dim() == 3:
        tensor_img = tensor
    else:
        tensor_img = tensor.view(1, 64, 32)
    with torch.no_grad():
        q = robust_char_query_variants(tensor_img).to(device)
        class_scores, sim_scores, retrieved = ensemble_scores(
            mchn_models, q, template_labels, len(loader.idx_to_label), template_mask=template_mask
        )
    # End-to-end crops are less template-like than folder samples. Blend
    # consensus with a small best-variant vote so a clean, well-normalized crop
    # is not drowned out by harsher preprocessing variants.
    mean_scores = torch.logsumexp(class_scores, dim=0) - torch.log(class_scores.new_tensor(float(class_scores.shape[0])))
    max_scores = torch.max(class_scores, dim=0).values
    pooled_scores = 0.62 * mean_scores + 0.38 * max_scores
    class_idx = int(torch.argmax(pooled_scores).item())
    variant_idx = int(torch.argmax(class_scores[:, class_idx]).item())
    best_template_idx = select_best_template_in_class(sim_scores[variant_idx : variant_idx + 1], template_labels, class_idx)
    prob = torch.softmax(pooled_scores, dim=-1)[class_idx].item()
    best_q = q[variant_idx : variant_idx + 1]
    template_sim = F.cosine_similarity(best_q, memory[best_template_idx].view(1, -1)).item()
    recon_sim = F.cosine_similarity(best_q, retrieved[variant_idx : variant_idx + 1]).item()
    top_text = format_top_predictions(pooled_scores)
    if return_debug:
        debug = {
            "query": tensor_to_rgb_image(best_q.detach().cpu()),
            "matched": tensor_to_rgb_image(memory[best_template_idx].detach().cpu()),
            "variant_index": variant_idx,
            "variant_count": int(q.shape[0]),
        }
        return class_idx, best_template_idx, prob, template_sim, recon_sim, top_text, debug
    return class_idx, best_template_idx, prob, template_sim, recon_sim, top_text


def format_top_predictions(scores, k=5):
    probs = torch.softmax(scores, dim=-1)
    values, indices = torch.topk(probs, k=min(k, probs.numel()))
    parts = []
    for value, idx in zip(values.tolist(), indices.tolist()):
        parts.append(f"{loader.idx_to_label[int(idx)]}:{value * 100:.1f}%")
    return " | ".join(parts)


def compose_mchn_debug_image(segmented_img, query_img, matched_img):
    def prepare(img):
        if img is None:
            return np.zeros((96, 48, 3), dtype=np.uint8)
        arr = img.copy()
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        return cv2.resize(arr, (48, 96), interpolation=cv2.INTER_NEAREST)

    parts = [prepare(segmented_img), prepare(query_img), prepare(matched_img)]
    sep = np.full((96, 4, 3), 230, dtype=np.uint8)
    return np.hstack([parts[0], sep, parts[1], sep, parts[2]])


def plate_position_mask(position):
    if position == 0:
        return chinese_mask
    if position == 1 and bool(letter_mask.any().item()):
        return letter_mask
    return plate_tail_mask if bool(plate_tail_mask.any().item()) else alnum_mask


def plate_detector_status_message():
    weights = pipeline.detector.weights_path
    if pipeline.detector.is_ready:
        return f"\u5f53\u524d\u5df2\u52a0\u8f7d\u8f66\u724c\u68c0\u6d4b\u6743\u91cd: {weights}\u3002{char_detector_status_message()}"
    if weights:
        return f"\u5f53\u524d\u8f66\u724c\u68c0\u6d4b\u6743\u91cd\u8def\u5f84\u4e0d\u5b58\u5728\u6216\u52a0\u8f7d\u5931\u8d25: {weights}\u3002{char_detector_status_message()}"
    return "\u5f53\u524d\u6ca1\u6709\u52a0\u8f7d\u8f66\u724c\u68c0\u6d4b\u6743\u91cd\u3002\u8bf7\u5148\u8bad\u7ec3 YOLO\uff0c\u6216\u8bbe\u7f6e PLATE_DETECTOR_WEIGHTS \u540e\u91cd\u542f app\u3002"


def char_detector_status_message():
    weights = pipeline.char_detector.weights_path
    if pipeline.char_detector.is_ready:
        return f"\u5df2\u52a0\u8f7d\u5b57\u7b26\u68c0\u6d4b\u6743\u91cd: {weights}"
    if weights:
        return f"\u5b57\u7b26\u68c0\u6d4b\u6743\u91cd\u8def\u5f84\u4e0d\u5b58\u5728\u6216\u52a0\u8f7d\u5931\u8d25: {weights}\uff0c\u5c06\u9000\u56de OpenCV \u5b57\u7b26\u5206\u5272\u3002"
    return "\u672a\u52a0\u8f7d\u5b57\u7b26\u68c0\u6d4b\u6743\u91cd\uff0c\u5c06\u9000\u56de OpenCV \u5b57\u7b26\u5206\u5272\uff1b\u5efa\u8bae\u8bbe\u7f6e CHAR_DETECTOR_WEIGHTS\u3002"

def predict_plate(image):
    if image is None:
        return "\u8bf7\u5148\u4e0a\u4f20\u56fe\u7247\u3002", None, pd.DataFrame(), [], []

    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    plate_img, chars = pipeline.process_image(img_bgr)
    if plate_img is None:
        return "\u65e0\u6cd5\u5b9a\u4f4d\u8f66\u724c\u3002" + plate_detector_status_message() + "\u5982\u9700 OpenCV \u5907\u9009\uff0c\u53ef\u8bbe\u7f6e PLATE_OPENCV_FALLBACK=1\u3002", None, pd.DataFrame(), [], []
    if not chars:
        return "\u5b57\u7b26\u5207\u5272\u5931\u8d25\u3002", cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB), pd.DataFrame(), [], []

    plate_text = ""
    rows = []
    char_images_display = [cv2.cvtColor(c, cv2.COLOR_GRAY2RGB) for c in chars]
    debug_images = []

    for i, char_img in enumerate(chars):
        tensor = torch.tensor(cv2.resize(char_img, (32, 64)), dtype=torch.float32).view(1, 64, 32) / 255.0
        current_mask = plate_position_mask(i)
        class_idx, best_template_idx, prob, template_sim, recon_sim, top_text, debug = recognize_tensor(tensor, current_mask, return_debug=True)
        char = loader.idx_to_label[class_idx]
        plate_text += char
        char_type = "\u6c49\u5b57" if "\u4e00" <= char <= "\u9fff" else ("\u6570\u5b57" if char.isdigit() else "\u5b57\u6bcd")
        debug_images.append(compose_mchn_debug_image(char_images_display[i], debug["query"], debug["matched"]))
        rows.append(
            {
                "\u4f4d\u7f6e": f"\u7b2c{i + 1}\u4f4d",
                "\u7c7b\u578b": char_type,
                "\u8bc6\u522b": char,
                "\u7c7b\u522b\u7f6e\u4fe1\u5ea6": f"{prob * 100:.2f}%",
                "\u5019\u9009Top5": top_text,
                "\u6a21\u677f\u76f8\u4f3c\u5ea6": f"{template_sim * 100:.2f}%",
                "\u91cd\u6784\u76f8\u4f3c\u5ea6": f"{recon_sim * 100:.2f}%",
                "\u53d8\u4f53": f"{debug['variant_index'] + 1}/{debug['variant_count']}",
            }
        )

    if len(chars) != 7:
        plate_text += f"  (\u8b66\u544a: \u5f53\u524d\u5207\u51fa {len(chars)} \u4e2a\u5b57\u7b26)"

    return f"\u6700\u7ec8\u8bc6\u522b: {plate_text}", cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB), pd.DataFrame(rows), char_images_display, debug_images


def predict_plate_from_sample(sample_rel_path):
    if not sample_rel_path:
        return "\u8bf7\u5148\u9009\u62e9\u6837\u4f8b\u56fe\u7247\u3002", None, pd.DataFrame(), [], []
    path_lookup = dict(full_car_samples)
    path = path_lookup.get(sample_rel_path)
    if not path or not os.path.exists(path):
        return "\u627e\u4e0d\u5230\u9009\u4e2d\u7684\u6837\u4f8b\u56fe\u7247\u3002", None, pd.DataFrame(), [], []
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        return "\u56fe\u7247\u8bfb\u53d6\u5931\u8d25\u3002", None, pd.DataFrame(), [], []

    plate_img, chars = pipeline.process_image(path)
    if plate_img is None:
        return "\u65e0\u6cd5\u5b9a\u4f4d\u8f66\u724c\u3002" + plate_detector_status_message() + "\u5982\u9700 OpenCV \u5907\u9009\uff0c\u53ef\u8bbe\u7f6e PLATE_OPENCV_FALLBACK=1\u3002", cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), pd.DataFrame(), [], []
    if not chars:
        return "\u5b57\u7b26\u5207\u5272\u5931\u8d25\u3002", cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB), pd.DataFrame(), [], []

    plate_text = ""
    rows = []
    gallery = [cv2.cvtColor(c, cv2.COLOR_GRAY2RGB) for c in chars]
    debug_images = []
    for i, char_img in enumerate(chars):
        tensor = torch.tensor(cv2.resize(char_img, (32, 64)), dtype=torch.float32).view(1, 64, 32) / 255.0
        current_mask = plate_position_mask(i)
        class_idx, best_template_idx, prob, template_sim, recon_sim, top_text, debug = recognize_tensor(tensor, current_mask, return_debug=True)
        char = loader.idx_to_label[class_idx]
        plate_text += char
        char_type = "\u6c49\u5b57" if "\u4e00" <= char <= "\u9fff" else ("\u6570\u5b57" if char.isdigit() else "\u5b57\u6bcd")
        debug_images.append(compose_mchn_debug_image(gallery[i], debug["query"], debug["matched"]))
        rows.append(
            {
                "\u4f4d\u7f6e": f"\u7b2c{i + 1}\u4f4d",
                "\u7c7b\u578b": char_type,
                "\u8bc6\u522b": char,
                "\u7c7b\u522b\u7f6e\u4fe1\u5ea6": f"{prob * 100:.2f}%",
                "\u5019\u9009Top5": top_text,
                "\u6a21\u677f\u76f8\u4f3c\u5ea6": f"{template_sim * 100:.2f}%",
                "\u91cd\u6784\u76f8\u4f3c\u5ea6": f"{recon_sim * 100:.2f}%",
                "\u53d8\u4f53": f"{debug['variant_index'] + 1}/{debug['variant_count']}",
            }
        )
    return f"\u6700\u7ec8\u8bc6\u522b: {plate_text}", cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB), pd.DataFrame(rows), gallery, debug_images


def update_sample_choices(class_label):
    choices = [item[0] for item in label_to_paths.get(class_label, [])]
    value = choices[0] if choices else None
    return gr.update(choices=choices, value=value)


def run_single_char_test(class_label, sample_rel_path, pollution_type, severity, seed):
    if not class_label or not sample_rel_path:
        return None, None, None, "\u8bf7\u5148\u9009\u62e9\u5b57\u7b26\u6837\u672c\u3002", pd.DataFrame()

    path_lookup = dict(label_to_paths[class_label])
    img_path = path_lookup.get(sample_rel_path)
    if not img_path or not os.path.exists(img_path):
        return None, None, None, "\u627e\u4e0d\u5230\u9009\u4e2d\u7684\u56fe\u7247\u3002", pd.DataFrame()

    with Image.open(img_path) as img:
        clean_tensor = normalize_char_tensor(pil_transform(img), img_size=(32, 64))
    polluter = CharPolluter(img_h=64, img_w=32, seed=int(seed))
    polluted_tensor = polluter.pollute(clean_tensor, pollution_type=pollution_type, severity=float(severity))
    class_idx, best_template_idx, prob, template_sim, recon_sim, top_text = recognize_tensor(polluted_tensor)
    pred_label = loader.idx_to_label[class_idx]
    matched_tensor = memory[best_template_idx].detach().cpu().view(1, 64, 32)

    rows = pd.DataFrame(
        [
            {
                "\u771f\u5b9e\u7c7b\u522b": class_label,
                "\u9884\u6d4b\u7c7b\u522b": pred_label,
                "\u6c61\u67d3\u7c7b\u578b": pollution_type,
                "\u6c61\u67d3\u5f3a\u5ea6": f"{float(severity):.2f}",
                "\u7c7b\u522b\u7f6e\u4fe1\u5ea6": f"{prob * 100:.2f}%",
                "\u5019\u9009Top5": top_text,
                "\u6a21\u677f\u76f8\u4f3c\u5ea6": f"{template_sim * 100:.2f}%",
                "\u91cd\u6784\u76f8\u4f3c\u5ea6": f"{recon_sim * 100:.2f}%",
            }
        ]
    )
    result = f"\u9884\u6d4b: {pred_label}    \u771f\u5b9e: {class_label}"
    return tensor_to_rgb_image(clean_tensor), tensor_to_rgb_image(polluted_tensor), tensor_to_rgb_image(matched_tensor), result, rows


with gr.Blocks(title="MCHN Polluted License Plate Recognition") as demo:
    gr.Markdown("# " + zh("title"))
    with gr.Tab(zh("plate_tab")):
        gr.Markdown(zh("detector_hint"))
        gr.Markdown(plate_detector_status_message())
        with gr.Row():
            with gr.Column(scale=1):
                img_in = gr.Image(label=zh("upload_plate"), type="numpy")
                btn = gr.Button(zh("run_lpr"), variant="primary")
                car_sample_dropdown = gr.Dropdown(
                    label="\u6216\u4ece full_cars \u6587\u4ef6\u5939\u9009\u62e9\u6837\u4f8b\u56fe\u7247",
                    choices=[item[0] for item in full_car_samples],
                    value=full_car_samples[0][0] if full_car_samples else None,
                    filterable=True,
                )
                sample_btn = gr.Button("\u8bc6\u522b\u9009\u4e2d\u6837\u4f8b\u56fe\u7247")
            with gr.Column(scale=1):
                txt_out = gr.Textbox(label=zh("overall"))
                img_out = gr.Image(label=zh("plate_img"))
                char_gallery = gr.Gallery(label=zh("char_gallery"), show_label=True, columns=7, height=120, object_fit="contain")
        df_out = gr.Dataframe(
            headers=["\u4f4d\u7f6e", "\u7c7b\u578b", "\u8bc6\u522b", "\u7c7b\u522b\u7f6e\u4fe1\u5ea6", "\u5019\u9009Top5", "\u6a21\u677f\u76f8\u4f3c\u5ea6", "\u91cd\u6784\u76f8\u4f3c\u5ea6", "\u53d8\u4f53"],
            label=zh("char_table"),
        )
        debug_gallery = gr.Gallery(label=zh("debug_gallery"), show_label=True, columns=7, height=150, object_fit="contain")
        btn.click(predict_plate, img_in, [txt_out, img_out, df_out, char_gallery, debug_gallery])
        sample_btn.click(predict_plate_from_sample, car_sample_dropdown, [txt_out, img_out, df_out, char_gallery, debug_gallery])

    with gr.Tab(zh("char_tab")):
        with gr.Row():
            with gr.Column(scale=1):
                class_dropdown = gr.Dropdown(label=zh("choose_class"), choices=class_choices, value=class_choices[0] if class_choices else None)
                sample_dropdown = gr.Dropdown(
                    label=zh("choose_sample"),
                    choices=[item[0] for item in label_to_paths[class_choices[0]]] if class_choices else [],
                    value=label_to_paths[class_choices[0]][0][0] if class_choices and label_to_paths[class_choices[0]] else None,
                    filterable=True,
                )
                pollution_dropdown = gr.Dropdown(label=zh("pollution"), choices=pollution_choices, value="mask")
                severity_slider = gr.Slider(label=zh("severity"), minimum=0.0, maximum=1.0, value=0.5, step=0.05)
                seed_number = gr.Number(label=zh("seed"), value=2026, precision=0)
                char_btn = gr.Button(zh("run_char"), variant="primary")
            with gr.Column(scale=2):
                char_result = gr.Textbox(label=zh("char_result"))
                with gr.Row():
                    original_img = gr.Image(label=zh("original"))
                    polluted_img = gr.Image(label=zh("polluted"))
                    matched_img = gr.Image(label=zh("matched"))
                char_df = gr.Dataframe(
                    headers=["\u771f\u5b9e\u7c7b\u522b", "\u9884\u6d4b\u7c7b\u522b", "\u6c61\u67d3\u7c7b\u578b", "\u6c61\u67d3\u5f3a\u5ea6", "\u7c7b\u522b\u7f6e\u4fe1\u5ea6", "\u5019\u9009Top5", "\u6a21\u677f\u76f8\u4f3c\u5ea6", "\u91cd\u6784\u76f8\u4f3c\u5ea6"]
                )
        class_dropdown.change(update_sample_choices, class_dropdown, sample_dropdown)
        char_btn.click(
            run_single_char_test,
            [class_dropdown, sample_dropdown, pollution_dropdown, severity_slider, seed_number],
            [original_img, polluted_img, matched_img, char_result, char_df],
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    share = os.environ.get("GRADIO_SHARE", "1" if is_kaggle_runtime() else "0") == "1"
    demo.launch(server_name="0.0.0.0", server_port=port, share=share)
