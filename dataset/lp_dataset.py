import os
import random

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


def normalize_char_tensor(tensor_img, img_size=(32, 64)):
    """Normalize a character image to white foreground on black background."""
    if tensor_img.dim() == 2:
        tensor_img = tensor_img.unsqueeze(0)
    img_h, img_w = img_size[1], img_size[0]
    x = tensor_img.float().clamp(0.0, 1.0)
    if tuple(x.shape[-2:]) != (img_h, img_w):
        x = F.interpolate(x.unsqueeze(0), size=(img_h, img_w), mode="nearest").squeeze(0)

    threshold = x.mean() + 0.10 * x.std().clamp_min(1e-6)
    binary = (x > threshold).float()
    border = torch.cat([binary[:, 0, :].flatten(), binary[:, -1, :].flatten(), binary[:, :, 0].flatten(), binary[:, :, -1].flatten()])
    if border.mean() > 0.45:
        binary = 1.0 - binary

    ys, xs = torch.where(binary[0] > 0)
    if ys.numel() == 0 or xs.numel() == 0:
        return binary

    y1 = max(0, int(ys.min().item()) - 1)
    y2 = min(img_h, int(ys.max().item()) + 2)
    x1 = max(0, int(xs.min().item()) - 1)
    x2 = min(img_w, int(xs.max().item()) + 2)
    crop = binary[:, y1:y2, x1:x2]
    h, w = crop.shape[-2:]
    scale = min(56.0 / max(1, h), 26.0 / max(1, w))
    new_h = max(1, min(62, int(round(h * scale))))
    new_w = max(1, min(30, int(round(w * scale))))
    resized = F.interpolate(crop.unsqueeze(0), size=(new_h, new_w), mode="nearest").squeeze(0)
    canvas = torch.zeros((1, img_h, img_w), dtype=resized.dtype)
    y = (img_h - new_h) // 2
    x0 = (img_w - new_w) // 2
    canvas[:, y : y + new_h, x0 : x0 + new_w] = resized
    return canvas


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
            "zh_jing": "\u4eac",
            "zh_jin": "\u6d25",
            "zh_ji": "\u5180",
            "zh_jin1": "\u664b",
            "zh_meng": "\u8499",
            "zh_liao": "\u8fbd",
            "zh_ji1": "\u5409",
            "zh_hei": "\u9ed1",
            "zh_hu": "\u6caa",
            "zh_su": "\u82cf",
            "zh_zhe": "\u6d59",
            "zh_wan": "\u7696",
            "zh_min": "\u95fd",
            "zh_gan": "\u8d63",
            "zh_lu": "\u9c81",
            "zh_yu": "\u8c6b",
            "zh_e": "\u9102",
            "zh_xiang": "\u6e58",
            "zh_yue": "\u7ca4",
            "zh_gui1": "\u6842",
            "zh_qiong": "\u743c",
            "zh_yu1": "\u6e1d",
            "zh_chuan": "\u5ddd",
            "zh_cuan": "\u5ddd",
            "zh_gui": "\u8d35",
            "zh_yun": "\u4e91",
            "zh_zang": "\u85cf",
            "zh_shan": "\u9655",
            "zh_gan1": "\u7518",
            "zh_qing": "\u9752",
            "zh_ning": "\u5b81",
            "zh_xin": "\u65b0",
            "zh_sx": "\u664b",
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
        signature = {"img_size": tuple(self.img_size), "label_map_version": 5, "roots": []}
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
        print(f"Loaded template cache: {self.memory_matrix.shape[0]} templates, {len(self.idx_to_label)} classes.")
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
                            tensor_img = normalize_char_tensor(self.transform(img), self.img_size)
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
        sample_indices=None,
    ):
        self.M, self.L, self.idx_to_label = template_loader.get_memory_matrix()
        self.virtual_size = virtual_size
        self.pollution_type = pollution_type
        self.severity = severity
        self.img_h = template_loader.img_size[1]
        self.img_w = template_loader.img_size[0]
        self.rng = random.Random(seed)
        self.sample_indices = list(sample_indices) if sample_indices is not None else None

    def __len__(self):
        return self.virtual_size

    def __getitem__(self, idx):
        if self.M.shape[0] == 0:
            raise RuntimeError("Template memory is empty. Check data_roots.")
        if self.sample_indices:
            target_idx = self.rng.choice(self.sample_indices)
        else:
            target_idx = self.rng.randint(0, self.M.shape[0] - 1)
        clean_q = self.M[target_idx].clone()
        clean_img = clean_q.view(1, self.img_h, self.img_w)
        polluted_img = self._pollute(clean_img)
        return polluted_img.clamp(0.0, 1.0).view(-1), clean_q, self.L[target_idx]

    def _severity_value(self):
        if isinstance(self.severity, (tuple, list)):
            low, high = float(self.severity[0]), float(self.severity[1])
            return max(0.0, min(1.0, self.rng.uniform(low, high)))
        return max(0.0, min(1.0, float(self.severity)))

    def _pollute(self, img):
        severity = self._severity_value()
        if self.pollution_type == "mixed":
            choices = ["mask", "noise", "salt_pepper", "blur", "fog", "dirt", "affine"]
            count = 1 if severity < 0.35 else 2 if severity < 0.7 else 3
            out = img.clone()
            for name in self.rng.sample(choices, count):
                out = self._apply_one(out, name, severity)
            return out
        return self._apply_one(img.clone(), self.pollution_type, severity)

    def _apply_one(self, img, pollution, severity):
        if pollution == "none":
            return img
        if pollution == "mask":
            return self._random_mask(img, severity)
        if pollution == "noise":
            sigma = 0.03 + 0.30 * severity
            return img + torch.randn_like(img) * sigma
        if pollution == "salt_pepper":
            prob = 0.01 + 0.22 * severity
            rnd = torch.rand_like(img)
            out = img.clone()
            out[rnd < prob / 2] = 0.0
            out[(rnd >= prob / 2) & (rnd < prob)] = 1.0
            return out
        if pollution == "blur":
            kernel = int(3 + 6 * severity)
            kernel = kernel + 1 if kernel % 2 == 0 else kernel
            return TF.gaussian_blur(img, kernel_size=[kernel, kernel])
        if pollution == "fog":
            fog = 0.15 + 0.45 * severity
            return img * (1.0 - fog) + fog
        if pollution == "dirt":
            return self._random_dirt(img, severity)
        if pollution == "affine":
            angle = self.rng.uniform(-10, 10) * severity
            translate = [
                int(self.rng.uniform(-3, 3) * severity),
                int(self.rng.uniform(-3, 3) * severity),
            ]
            scale = 1.0 + self.rng.uniform(-0.12, 0.12) * severity
            shear = self.rng.uniform(-6, 6) * severity
            return TF.affine(img, angle=angle, translate=translate, scale=scale, shear=[shear, 0.0])
        raise ValueError(f"Unsupported pollution_type: {pollution}")

    def _random_mask(self, img, severity):
        out = img.clone()
        block_count = 1 + int(3 * severity)
        for _ in range(block_count):
            max_h = max(2, int(self.img_h * (0.10 + 0.28 * severity)))
            max_w = max(2, int(self.img_w * (0.10 + 0.35 * severity)))
            h = self.rng.randint(2, max_h)
            w = self.rng.randint(2, max_w)
            y = self.rng.randint(0, max(0, self.img_h - h))
            x = self.rng.randint(0, max(0, self.img_w - w))
            out[:, y : y + h, x : x + w] = 0.0 if self.rng.random() < 0.7 else 1.0
        return out

    def _random_dirt(self, img, severity):
        out = img.clone()
        yy, xx = torch.meshgrid(
            torch.arange(self.img_h, dtype=torch.float32),
            torch.arange(self.img_w, dtype=torch.float32),
            indexing="ij",
        )
        spot_count = 1 + int(4 * severity)
        for _ in range(spot_count):
            cx = self.rng.uniform(0, self.img_w - 1)
            cy = self.rng.uniform(0, self.img_h - 1)
            radius = self.rng.uniform(2, 4 + 10 * severity)
            mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius**2
            out[:, mask] = self.rng.uniform(0.0, 0.35)
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


class CharPolluter:
    def __init__(self, img_h=64, img_w=32, seed=None):
        self.img_h = img_h
        self.img_w = img_w
        self.rng = random.Random(seed)

    def pollute(self, img, pollution_type="none", severity=0.5):
        severity = max(0.0, min(1.0, float(severity)))
        if img.dim() == 2:
            img = img.unsqueeze(0)
        if pollution_type == "mixed":
            choices = ["mask", "noise", "salt_pepper", "blur", "fog", "dirt", "affine"]
            count = 1 if severity < 0.35 else 2 if severity < 0.7 else 3
            out = img.clone()
            for name in self.rng.sample(choices, count):
                out = self._apply_one(out, name, severity)
            return out.clamp(0.0, 1.0)
        return self._apply_one(img.clone(), pollution_type, severity).clamp(0.0, 1.0)

    def _apply_one(self, img, pollution, severity):
        if pollution == "none":
            return img
        if pollution == "mask":
            return self._random_mask(img, severity)
        if pollution == "noise":
            sigma = 0.03 + 0.30 * severity
            return img + torch.randn_like(img) * sigma
        if pollution == "salt_pepper":
            prob = 0.01 + 0.22 * severity
            rnd = torch.rand_like(img)
            out = img.clone()
            out[rnd < prob / 2] = 0.0
            out[(rnd >= prob / 2) & (rnd < prob)] = 1.0
            return out
        if pollution == "blur":
            kernel = int(3 + 6 * severity)
            kernel = kernel + 1 if kernel % 2 == 0 else kernel
            return TF.gaussian_blur(img, kernel_size=[kernel, kernel])
        if pollution == "fog":
            fog = 0.15 + 0.45 * severity
            return img * (1.0 - fog) + fog
        if pollution == "dirt":
            return self._random_dirt(img, severity)
        if pollution == "affine":
            angle = self.rng.uniform(-10, 10) * severity
            translate = [
                int(self.rng.uniform(-3, 3) * severity),
                int(self.rng.uniform(-3, 3) * severity),
            ]
            scale = 1.0 + self.rng.uniform(-0.12, 0.12) * severity
            shear = self.rng.uniform(-6, 6) * severity
            return TF.affine(img, angle=angle, translate=translate, scale=scale, shear=[shear, 0.0])
        raise ValueError(f"Unsupported pollution_type: {pollution}")

    def _random_mask(self, img, severity):
        out = img.clone()
        block_count = 1 + int(3 * severity)
        for _ in range(block_count):
            max_h = max(2, int(self.img_h * (0.10 + 0.28 * severity)))
            max_w = max(2, int(self.img_w * (0.10 + 0.35 * severity)))
            h = self.rng.randint(2, max_h)
            w = self.rng.randint(2, max_w)
            y = self.rng.randint(0, max(0, self.img_h - h))
            x = self.rng.randint(0, max(0, self.img_w - w))
            out[:, y : y + h, x : x + w] = 0.0 if self.rng.random() < 0.7 else 1.0
        return out

    def _random_dirt(self, img, severity):
        out = img.clone()
        yy, xx = torch.meshgrid(
            torch.arange(self.img_h, dtype=torch.float32),
            torch.arange(self.img_w, dtype=torch.float32),
            indexing="ij",
        )
        spot_count = 1 + int(4 * severity)
        for _ in range(spot_count):
            cx = self.rng.uniform(0, self.img_w - 1)
            cy = self.rng.uniform(0, self.img_h - 1)
            radius = self.rng.uniform(2, 4 + 10 * severity)
            mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius**2
            out[:, mask] = self.rng.uniform(0.0, 0.35)
        return out
