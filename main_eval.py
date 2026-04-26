
import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader

current_dir = os.getcwd()
project_folder = "my-mchn-lpr-full-project"
if project_folder in os.listdir(current_dir):
    sys.path.append(os.path.join(current_dir, project_folder))
    os.chdir(os.path.join(current_dir, project_folder))

from dataset.lp_dataset import TemplateLoader, PollutedCharDataset
from models.mchn import ModernHopfieldNetwork
from utils.image_processing import LPRPipeline
from utils.metric_visuals import MetricVisualizer

def run_reconstruction_demo(mchn, dataset, visualizer):
    print("\n" + "="*50)
    print("🚀 任务一：生成 MCHN 重构对比图...")
    
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    polluted_q, clean_m, labels_idx = next(iter(dataloader))
    polluted_q = polluted_q.cuda()
    
    with torch.no_grad():
        reconstructed_z, preds = mchn(polluted_q)
        
    labels_str = [dataset.idx_to_label[idx.item()] for idx in labels_idx]
    
    visualizer.plot_reconstruction_grid(
        clean_qs=clean_m, 
        polluted_qs=polluted_q, 
        reconstructed_zs=reconstructed_z, 
        labels=labels_str,
        filename="mchn_reconstruction_demo.png"
    )

def run_robustness_evaluation(loader, visualizer):
    print("\n" + "="*50)
    print("🚀 任务二：启动极限鲁棒性压测 (生成折线图)...")
    
    severities = [0.1, 0.2, 0.4, 0.6, 0.8]
    acc_mchn = []
    acc_baseline = []
    
    mchn = ModernHopfieldNetwork(loader.memory_matrix.cuda(), beta=100.0, metric='manhattan').cuda()
    baseline = ModernHopfieldNetwork(loader.memory_matrix.cuda(), beta=1.0, metric='dot').cuda()

    for sev in severities:
        print(f"正在测试遮挡比例: {sev*100:.0f}% ...")
        test_dataset = PollutedCharDataset(loader, virtual_size=1000, pollution_type='mask', severity=sev)
        test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)
        
        correct_mchn = 0
        correct_base = 0
        total = 0
        
        with torch.no_grad():
            for q, _, labels in test_loader:
                q, labels = q.cuda(), labels.cuda()
                _, preds_mchn = mchn(q)
                pred_classes_mchn = torch.tensor([loader.labels[p.item()] for p in preds_mchn]).cuda()
                correct_mchn += (pred_classes_mchn == labels).sum().item()
                
                _, preds_base = baseline(q)
                pred_classes_base = torch.tensor([loader.labels[p.item()] for p in preds_base]).cuda()
                correct_base += (pred_classes_base == labels).sum().item()
                
                total += labels.size(0)
                
        acc_mchn.append((correct_mchn / total) * 100)
        acc_baseline.append((correct_base / total) * 100)
        
    visualizer.plot_robustness_curve(
        severities, acc_mchn, baseline_acc=acc_baseline, 
        pollution_type="Masking", filename="robustness_masking_curve.png"
    )

def run_end_to_end_system(loader):
    print("\n" + "="*50)
    print("🚀 任务三：真实环境车牌端到端测试...")
    
    test_dir = "./data/full_cars/ccpd_weather"
    if not os.path.exists(test_dir) or len(os.listdir(test_dir)) == 0:
        print(f"⚠️ 找不到测试图片，跳过本任务。")
        return

    pipeline = LPRPipeline(use_synthetic_pollution=False)
    mchn = ModernHopfieldNetwork(loader.memory_matrix.cuda(), beta=100.0, metric='manhattan').cuda()
    
    # 🌟 核心准备：生成测试掩码
    chinese_mask = torch.zeros(mchn.num_templates, dtype=torch.bool).cuda()
    if loader.chinese_indices: chinese_mask[loader.chinese_indices] = True
    alnum_mask = torch.zeros(mchn.num_templates, dtype=torch.bool).cuda()
    if loader.alnum_indices: alnum_mask[loader.alnum_indices] = True
    
    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.jpg')][:3]
    
    for img_path in test_files:
        print(f"\n📸 正在处理图片: {os.path.basename(img_path)}")
        plate_img, chars_img_list = pipeline.process_image(img_path)
        
        if plate_img is None or len(chars_img_list) == 0:
            print("❌ OpenCV 定位或切割失败。")
            continue
            
        plate_result = ""
        for i, char_img in enumerate(chars_img_list):
            char_resized = cv2.resize(char_img, (32, 64))
            char_tensor = torch.tensor(char_resized).float() / 255.0
            char_tensor = char_tensor.view(-1).unsqueeze(0).cuda() 
            
            current_mask = chinese_mask if i == 0 else alnum_mask
            
            with torch.no_grad():
                _, pred_idx = mchn(char_tensor, template_mask=current_mask)
                
            best_template_idx = pred_idx.item()
            class_idx = loader.labels[best_template_idx]
            char_str = loader.idx_to_label[class_idx]
            
            plate_result += char_str
            
        print(f"✅ MCHN 识别结果: 【 {plate_result} 】")

if __name__ == "__main__":
    print("🌟 现代 Hopfield 网络污染车牌识别系统 - 启动！🌟")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists("./results"): os.makedirs("./results")
    visualizer = MetricVisualizer(save_dir="./results")
    data_roots = ['./data/chars2', './data/charsChinese'] 
    loader = TemplateLoader(data_roots=data_roots, img_size=(32, 64))
    if loader.memory_matrix.shape[0] == 0:
        print("❌ 记忆库为空。程序终止。")
        exit()

    demo_dataset = PollutedCharDataset(loader, virtual_size=10, pollution_type='mask', severity=0.5)
    run_reconstruction_demo(mchn=ModernHopfieldNetwork(loader.memory_matrix.cuda(), beta=100.0).cuda(), 
                            dataset=demo_dataset, visualizer=visualizer)
    run_robustness_evaluation(loader, visualizer)
    run_end_to_end_system(loader)
    print("\n🎉 全部任务执行完毕！")