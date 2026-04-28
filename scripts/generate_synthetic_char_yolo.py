import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PLATE_W, PLATE_H = 400, 120


def list_templates(root):
    samples = []
    for class_dir in sorted(Path(root).iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMG_EXTS:
                samples.append((label, path))
    return samples


def load_char(path):
    img = Image.open(path).convert("L")
    arr = np.array(img)
    arr = cv2.resize(arr, (32, 64), interpolation=cv2.INTER_NEAREST)
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]])
    if np.mean(border > 0) > 0.45:
        binary = cv2.bitwise_not(binary)
    ys, xs = np.where(binary > 0)
    if len(xs) and len(ys):
        binary = binary[max(0, ys.min() - 1) : min(64, ys.max() + 2), max(0, xs.min() - 1) : min(32, xs.max() + 2)]
    return binary


def paste_char(plate, char, x1, y1, x2, y2):
    target_w, target_h = x2 - x1, y2 - y1
    h, w = char.shape[:2]
    scale = min(target_w / max(1, w), target_h / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(char, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    x = x1 + (target_w - new_w) // 2
    y = y1 + (target_h - new_h) // 2
    mask = resized > 0
    plate[y : y + new_h, x : x + new_w][mask] = 255
    return x, y, x + new_w, y + new_h


def write_label(path, boxes):
    with open(path, "w", encoding="utf-8") as f:
        for x1, y1, x2, y2 in boxes:
            cx = ((x1 + x2) / 2.0) / PLATE_W
            cy = ((y1 + y2) / 2.0) / PLATE_H
            bw = (x2 - x1) / PLATE_W
            bh = (y2 - y1) / PLATE_H
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic YOLO char-box dataset from char templates.")
    parser.add_argument("--chars2", default="./data/chars2")
    parser.add_argument("--chars-chinese", default="./data/charsChinese")
    parser.add_argument("--out", default="./dataset/yolo_chars_synth")
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    chinese = list_templates(args.chars_chinese)
    alnum = list_templates(args.chars2)
    letters = [(label, p) for label, p in alnum if label.upper().isalpha()]
    if not chinese or not alnum or not letters:
        raise RuntimeError("Template folders are empty or missing required classes.")

    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    ratios = np.array([1.10, 0.95, 0.22, 0.95, 0.95, 0.95, 0.95, 0.95], dtype=np.float32)
    unit = 340.0 / float(ratios.sum())
    for i in range(args.count):
        split = "val" if rng.random() < args.val_ratio else "train"
        plate = np.zeros((PLATE_H, PLATE_W), dtype=np.uint8)
        cv2.rectangle(plate, (10, 10), (390, 110), 255, 2)
        choices = [rng.choice(chinese), rng.choice(letters)] + [rng.choice(alnum) for _ in range(5)]
        x = 30.0
        boxes = []
        char_idx = 0
        for slot_idx, ratio in enumerate(ratios):
            slot_w = unit * float(ratio)
            if slot_idx == 2:
                cv2.circle(plate, (int(x + slot_w / 2), 62), 3, 255, -1)
                x += slot_w
                continue
            char = load_char(choices[char_idx][1])
            jitter_x = rng.randint(-3, 3)
            jitter_y = rng.randint(-3, 3)
            box = paste_char(
                plate,
                char,
                int(x + 4 + jitter_x),
                24 + jitter_y,
                int(x + slot_w - 4 + jitter_x),
                96 + jitter_y,
            )
            boxes.append(box)
            char_idx += 1
            x += slot_w

        if rng.random() < 0.5:
            plate = cv2.GaussianBlur(plate, (3, 3), 0)
        image_name = f"synth_{i:06d}.jpg"
        cv2.imwrite(str(out / "images" / split / image_name), plate)
        write_label(out / "labels" / split / image_name.replace(".jpg", ".txt"), boxes)

    print(f"Generated {args.count} synthetic plate images in {out}")


if __name__ == "__main__":
    main()
