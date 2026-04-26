
import os
import sys
import time
import subprocess
import threading
import torch
import cv2
import numpy as np

current_dir = os.getcwd()
project_folder = "my-mchn-lpr-full-project"
if project_folder in os.listdir(current_dir):
    sys.path.append(os.path.join(current_dir, project_folder))
    os.chdir(os.path.join(current_dir, project_folder))

try:
    import gradio as gr
    import pandas as pd
except ImportError:
    os.system("pip install gradio pandas -q")
    import gradio as gr
    import pandas as pd

from dataset.lp_dataset import TemplateLoader
from models.mchn import ModernHopfieldNetwork
from utils.image_processing import LPRPipeline

print("🚀 正在加载 MCHN 记忆矩阵与算法流水线...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = TemplateLoader(data_roots=["./data/chars2", "./data/charsChinese"], img_size=(32, 64))
mchn = ModernHopfieldNetwork(loader.memory_matrix.to(device), beta=100.0).to(device)
pipeline = LPRPipeline()

# 提前创建好汉字掩码和字母数字掩码
chinese_mask = torch.zeros(mchn.num_templates, dtype=torch.bool).to(device)
if loader.chinese_indices:
    chinese_mask[loader.chinese_indices] = True

alnum_mask = torch.zeros(mchn.num_templates, dtype=torch.bool).to(device)
if loader.alnum_indices:
    alnum_mask[loader.alnum_indices] = True

def predict_analysis(image):
    if image is None: return "⚠️ 未上传", None, pd.DataFrame(), []
    
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    enhanced = pipeline.enhancer.dehaze(img_bgr)
    plate_img = pipeline.segmenter.locate_plate(enhanced)
    
    if plate_img is None: return "❌ 无法定位车牌", cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB), pd.DataFrame(), []
    
    chars = pipeline.segmenter.segment_characters(plate_img)
    if not chars: return "❌ 字符切割失败", cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB), pd.DataFrame(), []
    
    res_str = ""
    stats = []
    char_images_display = [cv2.cvtColor(c, cv2.COLOR_GRAY2RGB) for c in chars]
    
    for i, c in enumerate(chars):
        t = torch.tensor(cv2.resize(c, (32, 64))).float().to(device) / 255.0
        t_flat = t.view(-1).unsqueeze(0)
        
        # 第 0 位强制使用汉字掩码，后面的强制使用字母数字掩码
        current_mask = chinese_mask if i == 0 else alnum_mask
        
        with torch.no_grad():
            retrieved, p_idx = mchn(t_flat, template_mask=current_mask)
            sim = torch.nn.functional.cosine_similarity(t_flat, retrieved).item()
            conf = max(0, min(100, sim * 100))
            
        char = loader.idx_to_label[loader.labels[p_idx.item()]]
        res_str += char
        c_type = "汉字" if '\u4e00' <= char <= '\u9fa5' else ("数字" if char.isdigit() else "字母")
        stats.append({"位置": f"第 {i+1} 块", "类型": c_type, "识别": char, "匹配置信度": f"{conf:.2f}%"})

    if len(chars) != 7:
        res_str += f" (⚠️ 警告: 切出了 {len(chars)} 个字符)"

    return f"最终识别：【 {res_str} 】", cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB), pd.DataFrame(stats), char_images_display

with gr.Blocks(title="MCHN Remote Server") as demo:
    gr.Markdown("# 🚗 现代 Hopfield 网络污染车牌识别系统")
    with gr.Row():
        with gr.Column(scale=1):
            img_in = gr.Image(label="点击上传本地图片", type="numpy")
            btn = gr.Button("⚡ 提交 MCHN 推理", variant="primary")
        with gr.Column(scale=1):
            txt_out = gr.Textbox(label="总体识别结果")
            img_out = gr.Image(label="车牌透视矫正与定位")
            
            gr.Markdown("### ✂️ 单字符切割结果展示")
            char_gallery = gr.Gallery(label="字符阵列", show_label=False, elem_id="gallery", columns=7, height=120, object_fit="contain")
            
    gr.Markdown("### 📊 逐字符识别准确率分析报告 (MCHN 余弦置信度)")
    df_out = gr.Dataframe(headers=["位置", "类型", "识别", "匹配置信度"])
    
    btn.click(predict_analysis, img_in, [txt_out, img_out, df_out, char_gallery])

def run_tunnel():
    time.sleep(5) 
    print("\n" + "="*50)
    print("🔑 正在获取你的隧道访问密码 (Endpoint IP)...")
    os.system("curl ipv4.icanhazip.com")
    print("\n🌍 正在建立公网隧道... 成功后请点击下方生成的 .loca.lt 链接")
    os.system("npx --yes localtunnel --port 7860")

if __name__ == "__main__":
    os.system("fuser -k 7860/tcp > /dev/null 2>&1")
    threading.Thread(target=run_tunnel, daemon=True).start()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, inline=False)