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

from dataset.lp_dataset import CharPolluter, TemplateLoader
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
    }
    return texts[text]


def is_kaggle_runtime():
    return os.path.exists("/kaggle/working") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def default_cache_path():
    cache_dir = "/kaggle/working/mchn_cache" if is_kaggle_runtime() else os.path.join(os.getcwd(), "results", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, "template_cache_32x64.pt")


def build_template_masks(loader, num_templates, device):
    chinese_mask = torch.zeros(num_templates, dtype=torch.bool, device=device)
    alnum_mask = torch.zeros(num_templates, dtype=torch.bool, device=device)
    letter_mask = torch.zeros(num_templates, dtype=torch.bool, device=device)
    digit_mask = torch.zeros(num_templates, dtype=torch.bool, device=device)
    if loader.chinese_indices:
        chinese_mask[loader.chinese_indices] = True
    if loader.alnum_indices:
        alnum_mask[loader.alnum_indices] = True
    for idx, label_idx in enumerate(loader.labels.tolist()):
        label = loader.idx_to_label[int(label_idx)]
        if len(label) == 1 and "A" <= label <= "Z":
            letter_mask[idx] = True
        elif label.isdigit() and len(label) == 1:
            digit_mask[idx] = True
    return chinese_mask, alnum_mask, letter_mask, digit_mask


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
            scores.append(torch.logsumexp(scaled[:, class_mask], dim=-1) - torch.log(scaled.new_tensor(float(count))))
    return torch.stack(scores, dim=-1)


def build_hopfield_ensemble(memory, device):
    return [
        ModernHopfieldNetwork(memory, beta=60.0, metric="dot", normalize=True, feature_mode="binary").to(device),
        ModernHopfieldNetwork(memory, beta=80.0, metric="dot", normalize=True, feature_mode="centered").to(device),
        ModernHopfieldNetwork(memory, beta=55.0, metric="dot", normalize=True, feature_mode="hybrid_shape").to(device),
    ]


def select_best_template_in_class(sim_scores, template_labels, class_idx):
    labels = template_labels.to(sim_scores.device)
    mask = labels == int(class_idx)
    return torch.argmax(sim_scores[0].masked_fill(~mask, -1e9)).item()


def ensemble_scores(models, q, template_labels, num_classes, template_mask=None):
    fused = None
    first_sim = None
    first_retrieved = None
    for model in models:
        retrieved, _, sim_scores = model(q, template_mask=template_mask, return_similarity=True)
        scores = class_free_energy_scores(sim_scores, template_labels, model.beta, num_classes, template_mask)
        log_probs = torch.log_softmax(scores, dim=-1)
        fused = log_probs if fused is None else fused + log_probs
        if first_sim is None:
            first_sim = sim_scores
            first_retrieved = retrieved
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


print("Loading MCHN memory and OpenCV LPR pipeline...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = TemplateLoader(["./data/chars2", "./data/charsChinese"], img_size=(32, 64), cache_path=default_cache_path())
if loader.memory_matrix.shape[0] == 0:
    raise RuntimeError("Template memory is empty. Please check ./data/chars2 and ./data/charsChinese.")

memory = loader.memory_matrix.to(device)
mchn_models = build_hopfield_ensemble(memory, device)
pipeline = LPRPipeline()
template_labels = loader.labels.to(device)
chinese_mask, alnum_mask, letter_mask, digit_mask = build_template_masks(loader, memory.shape[0], device)
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


def recognize_tensor(tensor, template_mask=None):
    base_q = tensor.view(1, -1).to(device) if tensor.dim() != 2 else tensor.to(device)
    with torch.no_grad():
        base_scores, base_sim, base_retrieved = ensemble_scores(
            mchn_models, base_q, template_labels, len(loader.idx_to_label), template_mask=template_mask
        )
        base_probs = torch.softmax(base_scores[0], dim=-1)
        base_conf, base_class = torch.max(base_probs, dim=-1)
        if float(base_conf.item()) >= 0.72:
            class_idx = int(base_class.item())
            best_template_idx = select_best_template_in_class(base_sim, template_labels, class_idx)
            template_sim = F.cosine_similarity(base_q, memory[best_template_idx].view(1, -1)).item()
            recon_sim = F.cosine_similarity(base_q, base_retrieved[0:1]).item()
            return class_idx, best_template_idx, float(base_conf.item()), template_sim, recon_sim

        q = affine_char_variants(tensor).to(device)
        class_scores, sim_scores, retrieved = ensemble_scores(
            mchn_models, q, template_labels, len(loader.idx_to_label), template_mask=template_mask
        )
    pooled_scores = torch.logsumexp(class_scores, dim=0) - torch.log(class_scores.new_tensor(float(class_scores.shape[0])))
    class_idx = int(torch.argmax(pooled_scores).item())
    variant_idx = int(torch.argmax(class_scores[:, class_idx]).item())
    best_template_idx = select_best_template_in_class(sim_scores[variant_idx : variant_idx + 1], template_labels, class_idx)
    prob = torch.softmax(pooled_scores, dim=-1)[class_idx].item()
    best_q = q[variant_idx : variant_idx + 1]
    template_sim = F.cosine_similarity(best_q, memory[best_template_idx].view(1, -1)).item()
    recon_sim = F.cosine_similarity(best_q, retrieved[variant_idx : variant_idx + 1]).item()
    return class_idx, best_template_idx, prob, template_sim, recon_sim


def plate_position_mask(position):
    if position == 0:
        return chinese_mask
    if position == 1 and bool(letter_mask.any().item()):
        return letter_mask
    return alnum_mask


def predict_plate(image):
    if image is None:
        return "\u8bf7\u5148\u4e0a\u4f20\u56fe\u7247\u3002", None, pd.DataFrame(), []

    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    plate_img, chars = pipeline.process_image(img_bgr)
    if plate_img is None:
        return "\u65e0\u6cd5\u5b9a\u4f4d\u8f66\u724c\u3002", None, pd.DataFrame(), []
    if not chars:
        return "\u5b57\u7b26\u5207\u5272\u5931\u8d25\u3002", cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB), pd.DataFrame(), []

    plate_text = ""
    rows = []
    char_images_display = [cv2.cvtColor(c, cv2.COLOR_GRAY2RGB) for c in chars]

    for i, char_img in enumerate(chars):
        tensor = torch.tensor(cv2.resize(char_img, (32, 64)), dtype=torch.float32) / 255.0
        tensor = tensor.view(1, -1)
        current_mask = plate_position_mask(i)
        class_idx, best_template_idx, prob, template_sim, recon_sim = recognize_tensor(tensor, current_mask)
        char = loader.idx_to_label[class_idx]
        plate_text += char
        char_type = "\u6c49\u5b57" if "\u4e00" <= char <= "\u9fff" else ("\u6570\u5b57" if char.isdigit() else "\u5b57\u6bcd")
        rows.append(
            {
                "\u4f4d\u7f6e": f"\u7b2c{i + 1}\u4f4d",
                "\u7c7b\u578b": char_type,
                "\u8bc6\u522b": char,
                "\u7c7b\u522b\u7f6e\u4fe1\u5ea6": f"{prob * 100:.2f}%",
                "\u6a21\u677f\u76f8\u4f3c\u5ea6": f"{template_sim * 100:.2f}%",
                "\u91cd\u6784\u76f8\u4f3c\u5ea6": f"{recon_sim * 100:.2f}%",
            }
        )

    if len(chars) != 7:
        plate_text += f"  (\u8b66\u544a: \u5f53\u524d\u5207\u51fa {len(chars)} \u4e2a\u5b57\u7b26)"

    return f"\u6700\u7ec8\u8bc6\u522b: {plate_text}", cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB), pd.DataFrame(rows), char_images_display


def predict_plate_from_sample(sample_rel_path):
    if not sample_rel_path:
        return "\u8bf7\u5148\u9009\u62e9\u6837\u4f8b\u56fe\u7247\u3002", None, pd.DataFrame(), []
    path_lookup = dict(full_car_samples)
    path = path_lookup.get(sample_rel_path)
    if not path or not os.path.exists(path):
        return "\u627e\u4e0d\u5230\u9009\u4e2d\u7684\u6837\u4f8b\u56fe\u7247\u3002", None, pd.DataFrame(), []
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        return "\u56fe\u7247\u8bfb\u53d6\u5931\u8d25\u3002", None, pd.DataFrame(), []

    plate_img, chars = pipeline.process_image(path)
    if plate_img is None:
        return "\u65e0\u6cd5\u5b9a\u4f4d\u8f66\u724c\u3002", cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), pd.DataFrame(), []
    if not chars:
        return "\u5b57\u7b26\u5207\u5272\u5931\u8d25\u3002", cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB), pd.DataFrame(), []

    plate_text = ""
    rows = []
    gallery = [cv2.cvtColor(c, cv2.COLOR_GRAY2RGB) for c in chars]
    for i, char_img in enumerate(chars):
        tensor = torch.tensor(cv2.resize(char_img, (32, 64)), dtype=torch.float32) / 255.0
        current_mask = plate_position_mask(i)
        class_idx, best_template_idx, prob, template_sim, recon_sim = recognize_tensor(tensor.view(1, -1), current_mask)
        char = loader.idx_to_label[class_idx]
        plate_text += char
        char_type = "\u6c49\u5b57" if "\u4e00" <= char <= "\u9fff" else ("\u6570\u5b57" if char.isdigit() else "\u5b57\u6bcd")
        rows.append(
            {
                "\u4f4d\u7f6e": f"\u7b2c{i + 1}\u4f4d",
                "\u7c7b\u578b": char_type,
                "\u8bc6\u522b": char,
                "\u7c7b\u522b\u7f6e\u4fe1\u5ea6": f"{prob * 100:.2f}%",
                "\u6a21\u677f\u76f8\u4f3c\u5ea6": f"{template_sim * 100:.2f}%",
                "\u91cd\u6784\u76f8\u4f3c\u5ea6": f"{recon_sim * 100:.2f}%",
            }
        )
    return f"\u6700\u7ec8\u8bc6\u522b: {plate_text}", cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB), pd.DataFrame(rows), gallery


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
        clean_tensor = pil_transform(img)
    polluter = CharPolluter(img_h=64, img_w=32, seed=int(seed))
    polluted_tensor = polluter.pollute(clean_tensor, pollution_type=pollution_type, severity=float(severity))
    class_idx, best_template_idx, prob, template_sim, recon_sim = recognize_tensor(polluted_tensor)
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
            headers=["\u4f4d\u7f6e", "\u7c7b\u578b", "\u8bc6\u522b", "\u7c7b\u522b\u7f6e\u4fe1\u5ea6", "\u6a21\u677f\u76f8\u4f3c\u5ea6", "\u91cd\u6784\u76f8\u4f3c\u5ea6"],
            label=zh("char_table"),
        )
        btn.click(predict_plate, img_in, [txt_out, img_out, df_out, char_gallery])
        sample_btn.click(predict_plate_from_sample, car_sample_dropdown, [txt_out, img_out, df_out, char_gallery])

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
                    headers=["\u771f\u5b9e\u7c7b\u522b", "\u9884\u6d4b\u7c7b\u522b", "\u6c61\u67d3\u7c7b\u578b", "\u6c61\u67d3\u5f3a\u5ea6", "\u7c7b\u522b\u7f6e\u4fe1\u5ea6", "\u6a21\u677f\u76f8\u4f3c\u5ea6", "\u91cd\u6784\u76f8\u4f3c\u5ea6"]
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
