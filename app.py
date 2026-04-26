import os

import cv2
import gradio as gr
import pandas as pd
import torch
import torch.nn.functional as F

from dataset.lp_dataset import TemplateLoader
from models.mchn import ModernHopfieldNetwork
from utils.image_processing import LPRPipeline


def build_template_masks(loader, num_templates, device):
    chinese_mask = torch.zeros(num_templates, dtype=torch.bool, device=device)
    if loader.chinese_indices:
        chinese_mask[loader.chinese_indices] = True

    alnum_mask = torch.zeros(num_templates, dtype=torch.bool, device=device)
    if loader.alnum_indices:
        alnum_mask[loader.alnum_indices] = True

    return chinese_mask, alnum_mask


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
        scores.append(
            torch.logsumexp(scaled[:, class_mask], dim=-1)
            - torch.log(scaled.new_tensor(float(count)))
        )
    return torch.stack(scores, dim=-1)


def select_best_template_in_class(sim_scores, template_labels, class_idx):
    labels = template_labels.to(sim_scores.device)
    mask = labels == int(class_idx)
    return torch.argmax(sim_scores[0].masked_fill(~mask, -1e9)).item()


print("Loading MCHN memory and OpenCV LPR pipeline...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = TemplateLoader(data_roots=["./data/chars2", "./data/charsChinese"], img_size=(32, 64))
if loader.memory_matrix.shape[0] == 0:
    raise RuntimeError("Template memory is empty. Please check ./data/chars2 and ./data/charsChinese.")

mchn = ModernHopfieldNetwork(
    loader.memory_matrix.to(device),
    beta=60.0,
    metric="dot",
    normalize=True,
    feature_mode="binary",
).to(device)
mchn_centered = ModernHopfieldNetwork(
    loader.memory_matrix.to(device),
    beta=80.0,
    metric="dot",
    normalize=True,
    feature_mode="centered",
).to(device)
pipeline = LPRPipeline()
template_labels = loader.labels.to(device)
chinese_mask, alnum_mask = build_template_masks(loader, mchn.num_templates, device)


def predict_analysis(image):
    if image is None:
        return "请先上传图片。", None, pd.DataFrame(), []

    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    enhanced = pipeline.enhancer.dehaze(img_bgr)
    plate_img = pipeline.segmenter.locate_plate(enhanced)

    if plate_img is None:
        return "无法定位车牌。", cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB), pd.DataFrame(), []

    chars = pipeline.segmenter.segment_characters(plate_img)
    if not chars:
        return "字符切割失败。", cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB), pd.DataFrame(), []

    plate_text = ""
    rows = []
    char_images_display = [cv2.cvtColor(c, cv2.COLOR_GRAY2RGB) for c in chars]

    for i, char_img in enumerate(chars):
        tensor = torch.tensor(cv2.resize(char_img, (32, 64)), dtype=torch.float32, device=device) / 255.0
        tensor = tensor.view(1, -1)
        current_mask = chinese_mask if i == 0 else alnum_mask

        with torch.no_grad():
            retrieved, _, sim_scores = mchn(tensor, template_mask=current_mask, return_similarity=True)
            class_scores_binary = class_free_energy_scores(
                sim_scores,
                template_labels,
                beta=mchn.beta,
                num_classes=len(loader.idx_to_label),
                template_mask=current_mask,
            )
            _, _, sim_scores_centered = mchn_centered(tensor, template_mask=current_mask, return_similarity=True)
            class_scores_centered = class_free_energy_scores(
                sim_scores_centered,
                template_labels,
                beta=mchn_centered.beta,
                num_classes=len(loader.idx_to_label),
                template_mask=current_mask,
            )
            class_scores = torch.log_softmax(class_scores_binary, dim=-1) + torch.log_softmax(
                class_scores_centered, dim=-1
            )

        class_idx = int(torch.argmax(class_scores, dim=-1).item())
        char = loader.idx_to_label[class_idx]
        plate_text += char
        best_template_idx = select_best_template_in_class(sim_scores, template_labels, class_idx)

        prob = torch.softmax(class_scores, dim=-1)[0, class_idx].item()
        recon_sim = F.cosine_similarity(tensor, loader.memory_matrix[best_template_idx].to(device).view(1, -1)).item()
        char_type = "汉字" if "\u4e00" <= char <= "\u9fff" else ("数字" if char.isdigit() else "字母")
        rows.append(
            {
                "位置": f"第{i + 1}位",
                "类型": char_type,
                "识别": char,
                "类别置信度": f"{prob * 100:.2f}%",
                "模板相似度": f"{recon_sim * 100:.2f}%",
            }
        )

    if len(chars) != 7:
        plate_text += f"  (警告: 当前切出 {len(chars)} 个字符)"

    return (
        f"最终识别: {plate_text}",
        cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB),
        pd.DataFrame(rows),
        char_images_display,
    )


with gr.Blocks(title="MCHN Polluted License Plate Recognition") as demo:
    gr.Markdown("# 现代 Hopfield 网络污染车牌字符识别")
    with gr.Row():
        with gr.Column(scale=1):
            img_in = gr.Image(label="上传车牌或整车图片", type="numpy")
            btn = gr.Button("提交 MCHN 推理", variant="primary")
        with gr.Column(scale=1):
            txt_out = gr.Textbox(label="总体识别结果")
            img_out = gr.Image(label="车牌定位与透视矫正")
            char_gallery = gr.Gallery(
                label="字符切割结果",
                show_label=True,
                columns=7,
                height=120,
                object_fit="contain",
            )

    df_out = gr.Dataframe(
        headers=["位置", "类型", "识别", "类别置信度", "模板相似度"],
        label="逐字符识别分析",
    )

    btn.click(predict_analysis, img_in, [txt_out, img_out, df_out, char_gallery])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
