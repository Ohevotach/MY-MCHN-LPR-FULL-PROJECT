import os
import random

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


class TemplateLoader:
    def __init__(self, data_roots, img_size=(32, 64), cache_path=None, use_cache=True):
        self.data_roots = data_roots
        self.img_size = img_size
        self.cache_path = cache_path or f"./data/template_cache_{img_size[0]}x{img_size[1]}.pt"
        self.use_cache = use_cache
        self.templates = []
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.template_paths = []

        self.chinese_indices = []
        self.alnum_indices = []

        self.pinyin_map = {
            "zh_jing": "京",
            "zh_jin": "津",
            "zh_ji": "冀",
            "zh_jin1": "晋",
            "zh_meng": "蒙",
            "zh_liao": "辽",
            "zh_ji1": "吉",
            "zh_hei": "黑",
            "zh_hu": "沪",
            "zh_su": "苏",
            "zh_zhe": "浙",
            "zh_wan": "皖",
            "zh_min": "闽",
            "zh_gan": "赣",
            "zh_lu": "鲁",
            "zh_yu": "豫",
            "zh_e": "鄂",
            "zh_xiang": "湘",
            "zh_yue": "粤",
            "zh_gui1": "桂",
            "zh_qiong": "琼",
            "zh_yu1": "渝",
            "zh_chuan": "川",
            "zh_gui": "贵",
            "zh_yun": "云",
            "zh_zang": "藏",
            "zh_shan": "陕",
            "zh_gan1": "甘",
            "zh_qing": "青",
            "zh_ning": "宁",
            "zh_xin": "新",
            "zh_sx": "晋",
        }

        self.valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((self.img_size[1], self.img_size[0])),
                transforms.ToTensor(),
            ]
        )

        self.cache_signature = self._build_cache_signature()
        if self.use_cache and self._try_load_cache():
            return

        self._load_all_templates()
        if self.use_cache:
            self._save_cache()

    def _build_cache_signature(self):
        signature = {
            "img_size": tuple(self.img_size),
            "label_map_version": 2,
            "roots": [],
        }
        for root_dir in self.data_roots:
            root_info = {"root": os.path.abspath(root_dir), "file_count": 0, "latest_mtime": 0.0}
            if os.path.exists(root_dir):
                for current_root, _, files in os.walk(root_dir):
                    for file_name in files:
                        if not file_name.lower().endswith(self.valid_extensions):
                            continue
                        file_path = os.path.join(current_root, file_name)
                        root_info["file_count"] += 1
                        root_info["latest_mtime"] = max(root_info["latest_mtime"], os.path.getmtime(file_path))
            signature["roots"].append(root_info)
        return signature

    def _try_load_cache(self):
        if not os.path.exists(self.cache_path):
            return False
        try:
            try:
                cache = torch.load(self.cache_path, map_location="cpu", weights_only=True)
            except TypeError:
                cache = torch.load(self.cache_path, map_location="cpu")
        except Exception as exc:
            print(f"Warning: failed to load template cache: {exc}")
            return False

        if cache.get("signature") != self.cache_signature:
            print("Template cache is stale; rebuilding memory matrix...")
            return False

        self.memory_matrix = cache["memory_matrix"].float()
        self.labels = cache["labels"].long()
        self.label_to_idx = cache["label_to_idx"]
        self.idx_to_label = cache["idx_to_label"]
        self.template_paths = cache.get("template_paths", [])
        self.chinese_indices = cache.get("chinese_indices", [])
        self.alnum_indices = cache.get("alnum_indices", [])
        print(
            "Loaded template cache: "
            f"{self.memory_matrix.shape[0]} templates, "
            f"{len(self.idx_to_label)} classes."
        )
        return True

    def _save_cache(self):
        if self.memory_matrix.shape[0] == 0:
            return
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        cache = {
            "signature": self.cache_signature,
            "memory_matrix": self.memory_matrix.cpu(),
            "labels": self.labels.cpu(),
            "label_to_idx": self.label_to_idx,
            "idx_to_label": self.idx_to_label,
            "template_paths": self.template_paths,
            "chinese_indices": self.chinese_indices,
            "alnum_indices": self.alnum_indices,
        }
        try:
            torch.save(cache, self.cache_path)
            print(f"Saved template cache: {self.cache_path}")
        except Exception as exc:
            print(f"Warning: failed to save template cache: {exc}")

    def _load_all_templates(self):
        print("Building MCHN memory matrix...")
        current_idx = 0

        for root_dir in self.data_roots:
            if not os.path.exists(root_dir):
                print(f"Warning: data root not found: {root_dir}")
                continue

            for folder_name in sorted(os.listdir(root_dir)):
                class_path = os.path.join(root_dir, folder_name)
                if not os.path.isdir(class_path):
                    continue

                display_label = self.pinyin_map.get(folder_name, folder_name)
                if display_label not in self.label_to_idx:
                    self.label_to_idx[display_label] = current_idx
                    self.idx_to_label[current_idx] = display_label
                    current_idx += 1

                for file_name in sorted(os.listdir(class_path)):
                    if not file_name.lower().endswith(self.valid_extensions):
                        continue

                    img_path = os.path.join(class_path, file_name)
                    try:
                        with Image.open(img_path) as img:
                            tensor_img = self.transform(img)
                    except Exception as exc:
                        print(f"Warning: failed to load {img_path}: {exc}")
                        continue

                    self.templates.append(tensor_img.view(-1))
                    self.labels.append(self.label_to_idx[display_label])
                    self.template_paths.append(img_path)

                    template_idx = len(self.templates) - 1
                    if self._is_chinese_label(display_label):
                        self.chinese_indices.append(template_idx)
                    else:
                        self.alnum_indices.append(template_idx)

        if self.templates:
            self.memory_matrix = torch.stack(self.templates).float()
            self.labels = torch.tensor(self.labels, dtype=torch.long)
        else:
            feature_dim = self.img_size[0] * self.img_size[1]
            self.memory_matrix = torch.empty((0, feature_dim), dtype=torch.float32)
            self.labels = torch.empty((0,), dtype=torch.long)

        print(
            "Memory matrix ready: "
            f"{self.memory_matrix.shape[0]} templates, "
            f"{len(self.idx_to_label)} classes, "
            f"{len(self.chinese_indices)} Chinese templates, "
            f"{len(self.alnum_indices)} alnum templates."
        )

    @staticmethod
    def _is_chinese_label(label):
        return len(label) == 1 and "\u4e00" <= label <= "\u9fff"

    def get_memory_matrix(self):
        return self.memory_matrix, self.labels, self.idx_to_label


class PollutedCharDataset(Dataset):
    def __init__(
        self,
        template_loader,
        virtual_size=5000,
        pollution_type="mixed",
        severity=0.5,
        seed=None,
    ):
        self.M, self.L, self.idx_to_label = template_loader.get_memory_matrix()
        self.virtual_size = virtual_size
        self.pollution_type = pollution_type
        self.severity = float(max(0.0, min(1.0, severity)))
        self.img_h = template_loader.img_size[1]
        self.img_w = template_loader.img_size[0]
        self.rng = random.Random(seed)

    def __len__(self):
        return self.virtual_size

    def __getitem__(self, idx):
        if self.M.shape[0] == 0:
            raise RuntimeError("Template memory is empty. Check data_roots.")

        target_idx = self.rng.randint(0, self.M.shape[0] - 1)
        clean_q = self.M[target_idx].clone()
        clean_img = clean_q.view(1, self.img_h, self.img_w)
        polluted_img = self._pollute(clean_img)
        polluted_q = polluted_img.clamp(0.0, 1.0).view(-1)
        return polluted_q, clean_q, self.L[target_idx]

    def _pollute(self, img):
        pollution = self.pollution_type
        if pollution == "mixed":
            choices = ["mask", "noise", "salt_pepper", "blur", "fog", "dirt", "affine"]
            count = 1 if self.severity < 0.35 else 2 if self.severity < 0.7 else 3
            out = img.clone()
            for name in self.rng.sample(choices, count):
                out = self._apply_one(out, name)
            return out
        return self._apply_one(img.clone(), pollution)

    def _apply_one(self, img, pollution):
        if pollution == "none":
            return img
        if pollution == "mask":
            return self._random_mask(img)
        if pollution == "noise":
            sigma = 0.05 + 0.45 * self.severity
            return img + torch.randn_like(img) * sigma
        if pollution == "salt_pepper":
            prob = 0.02 + 0.35 * self.severity
            rnd = torch.rand_like(img)
            out = img.clone()
            out[rnd < prob / 2] = 0.0
            out[(rnd >= prob / 2) & (rnd < prob)] = 1.0
            return out
        if pollution == "blur":
            kernel = int(3 + 8 * self.severity)
            kernel = kernel + 1 if kernel % 2 == 0 else kernel
            return TF.gaussian_blur(img, kernel_size=[kernel, kernel])
        if pollution == "fog":
            fog = 0.25 + 0.6 * self.severity
            return img * (1.0 - fog) + fog
        if pollution == "dirt":
            return self._random_dirt(img)
        if pollution == "affine":
            angle = self.rng.uniform(-12, 12) * self.severity
            translate = [
                int(self.rng.uniform(-4, 4) * self.severity),
                int(self.rng.uniform(-4, 4) * self.severity),
            ]
            scale = 1.0 + self.rng.uniform(-0.18, 0.18) * self.severity
            shear = self.rng.uniform(-8, 8) * self.severity
            return TF.affine(img, angle=angle, translate=translate, scale=scale, shear=[shear, 0.0])
        raise ValueError(f"Unsupported pollution_type: {pollution}")

    def _random_mask(self, img):
        out = img.clone()
        block_count = 1 + int(4 * self.severity)
        for _ in range(block_count):
            max_h = max(2, int(self.img_h * (0.12 + 0.35 * self.severity)))
            max_w = max(2, int(self.img_w * (0.12 + 0.45 * self.severity)))
            h = self.rng.randint(2, max_h)
            w = self.rng.randint(2, max_w)
            y = self.rng.randint(0, max(0, self.img_h - h))
            x = self.rng.randint(0, max(0, self.img_w - w))
            value = 0.0 if self.rng.random() < 0.7 else 1.0
            out[:, y : y + h, x : x + w] = value
        return out

    def _random_dirt(self, img):
        out = img.clone()
        yy, xx = torch.meshgrid(
            torch.arange(self.img_h, dtype=torch.float32),
            torch.arange(self.img_w, dtype=torch.float32),
            indexing="ij",
        )
        spot_count = 1 + int(5 * self.severity)
        for _ in range(spot_count):
            cx = self.rng.uniform(0, self.img_w - 1)
            cy = self.rng.uniform(0, self.img_h - 1)
            radius = self.rng.uniform(2, 4 + 12 * self.severity)
            mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius**2
            value = self.rng.uniform(0.0, 0.35)
            out[:, mask] = value
        return out


def build_class_memory(template_loader, reduce="mean"):
    memory, labels, _ = template_loader.get_memory_matrix()
    if memory.shape[0] == 0:
        return memory, labels

    class_vectors = []
    class_labels = []
    for label in sorted(labels.unique().tolist()):
        samples = memory[labels == label]
        if reduce == "mean":
            vector = samples.mean(dim=0)
        elif reduce == "first":
            vector = samples[0]
        elif reduce == "medoid":
            center = samples.mean(dim=0, keepdim=True)
            distances = torch.cdist(samples, center, p=2.0).squeeze(1)
            vector = samples[torch.argmin(distances)]
        else:
            raise ValueError(f"Unsupported reduce mode: {reduce}")
        class_vectors.append(vector)
        class_labels.append(label)

    return torch.stack(class_vectors), torch.tensor(class_labels, dtype=torch.long)
