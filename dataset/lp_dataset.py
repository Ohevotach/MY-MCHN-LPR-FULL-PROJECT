
import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class TemplateLoader:
    def __init__(self, data_roots, img_size=(32, 64)):
        self.data_roots = data_roots
        self.img_size = img_size
        self.templates = []
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        
        self.chinese_indices = []
        self.alnum_indices = []
        
        self.pinyin_map = {
            "zh_jing": "京", "zh_jin": "津", "zh_ji": "冀", "zh_jin1": "晋", "zh_meng": "蒙",
            "zh_liao": "辽", "zh_ji1": "吉", "zh_hei": "黑", "zh_hu": "沪", "zh_su": "苏",
            "zh_zhe": "浙", "zh_wan": "皖", "zh_min": "闽", "zh_gan": "赣", "zh_lu": "鲁",
            "zh_yu": "豫", "zh_e": "鄂", "zh_xiang": "湘", "zh_yue": "粤", "zh_gui1": "桂",
            "zh_qiong": "琼", "zh_yu1": "渝", "zh_chuan": "川", "zh_gui": "贵", "zh_yun": "云",
            "zh_zang": "藏", "zh_shan": "陕", "zh_gan1": "甘", "zh_qing": "青", "zh_ning": "宁",
            "zh_xin": "新"
        }
        
        self.valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((self.img_size[1], self.img_size[0])), 
            transforms.ToTensor()
        ])
        
        self._load_all_templates()
        
    def _load_all_templates(self):
        print("🚀 启动 MCHN 记忆库构建引擎...")
        current_idx = 0
        for root_dir in self.data_roots:
            if not os.path.exists(root_dir): continue
            
            for folder_name in sorted(os.listdir(root_dir)):
                class_path = os.path.join(root_dir, folder_name)
                if not os.path.isdir(class_path): continue
                
                display_label = self.pinyin_map.get(folder_name, folder_name)
                if display_label not in self.label_to_idx:
                    self.label_to_idx[display_label] = current_idx
                    self.idx_to_label[current_idx] = display_label
                    current_idx += 1
                
                for file_name in os.listdir(class_path):
                    if file_name.lower().endswith(self.valid_extensions):
                        img_path = os.path.join(class_path, file_name)
                        try:
                            with Image.open(img_path) as img:
                                tensor_img = self.transform(img)
                                self.templates.append(tensor_img.view(-1))
                                self.labels.append(self.label_to_idx[display_label])
                                
                                global_template_idx = len(self.templates) - 1
                                # 🌟 绝对修复：使用 Unicode 判断，无视文件夹命名
                                is_chinese = '\u4e00' <= display_label <= '\u9fa5'
                                if is_chinese:
                                    self.chinese_indices.append(global_template_idx)
                                else:
                                    self.alnum_indices.append(global_template_idx)
                        except Exception:
                            pass
                            
        self.memory_matrix = torch.stack(self.templates)
        print(f"✅ 记忆矩阵构建成功！中文字符模板 {len(self.chinese_indices)} 张，字母数字模板 {len(self.alnum_indices)} 张。")

    def get_memory_matrix(self):
        return self.memory_matrix, self.labels, self.idx_to_label

class PollutedCharDataset(Dataset):
    def __init__(self, template_loader, virtual_size=5000, pollution_type='noise', severity=0.5):
        self.M, self.L, self.idx_to_label = template_loader.get_memory_matrix()
        self.virtual_size = virtual_size
        self.pollution_type = pollution_type
        self.severity = severity
        self.img_h, self.img_w = template_loader.img_size[1], template_loader.img_size[0]

    def __len__(self): return self.virtual_size

    def __getitem__(self, idx):
        target_idx = random.randint(0, self.M.shape[0] - 1)
        clean_q = self.M[target_idx]
        polluted_q = clean_q.clone() # 演示用途简写，详见原有增强逻辑
        return polluted_q, clean_q, self.L[target_idx]