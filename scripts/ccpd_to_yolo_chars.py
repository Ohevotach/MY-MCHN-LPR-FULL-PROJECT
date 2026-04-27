import argparse
import random
import re
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
PLATE_W, PLATE_H = 400, 120


def parse_coord_pairs(text):
    return [
        (float(x), float(y))
        for x, y in re.findall(r"(\d+(?:\.\d+)?)[&xX](\d+(?:\.\d+)?)", text)
    ]


def parse_ccpd_points(path):
    parts = Path(path).stem.split("-")
    if len(parts) >= 4:
        points = parse_coord_pairs(parts[3])
        if len(points) >= 4:
            return points[:4]
    if len(parts) >= 3:
        box_points = parse_coord_pairs(parts[2])
        if len(box_points) == 2:
            (x1, y1), (x2, y2) = box_points
            return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return None


def order_points(points):
    pts = np.array(points, dtype=np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def warp_plate(img, points):
    src = order_points(points)
    dst = np.array([[0, 0], [PLATE_W - 1, 0], [PLATE_W - 1, PLATE_H - 1], [0, PLATE_H - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, matrix, (PLATE_W, PLATE_H))


def char_boxes():
    # Province, letter, separator, then five alphanumeric characters.
    ratios = np.array([1.10, 0.95, 0.22, 0.95, 0.95, 0.95, 0.95, 0.95], dtype=np.float32)
    unit = 340.0 / float(ratios.sum())
    x = 30.0
    boxes = []
    for slot_idx, ratio in enumerate(ratios):
        slot_w = unit * float(ratio)
        if slot_idx == 2:
            x += slot_w
            continue
        pad_x = max(2.0, slot_w * 0.08)
        x1 = x + pad_x
        x2 = x + slot_w - pad_x
        y1 = 22.0
        y2 = 100.0
        boxes.append((x1, y1, x2, y2))
        x += slot_w
    return boxes


def write_yolo_label(path, boxes):
    with open(path, "w", encoding="utf-8") as f:
        for x1, y1, x2, y2 in boxes:
            cx = ((x1 + x2) / 2.0) / PLATE_W
            cy = ((y1 + y2) / 2.0) / PLATE_H
            bw = (x2 - x1) / PLATE_W
            bh = (y2 - y1) / PLATE_H
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def collect_images(src):
    if not src.exists():
        raise FileNotFoundError(f"Source directory not found: {src}")
    return [p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS]


def main():
    parser = argparse.ArgumentParser(description="Build YOLO character-box data from CCPD-style full-car images.")
    parser.add_argument("--src", required=True, help="Source image directory, e.g. ./data/full_cars/ccpd_base.")
    parser.add_argument("--out", default="./dataset/yolo_chars")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    files = [p for p in collect_images(src) if parse_ccpd_points(p) is not None]
    if not files:
        raise SystemExit("No CCPD coordinate labels found in filenames.")

    random.Random(args.seed).shuffle(files)
    val_count = max(1, int(round(len(files) * args.val_ratio))) if len(files) > 1 else 0
    splits = {"val": files[:val_count], "train": files[val_count:]}
    boxes = char_boxes()

    for split, split_files in splits.items():
        image_dir = out / "images" / split
        label_dir = out / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for src_path in split_files:
            img = cv2.imread(str(src_path))
            if img is None:
                continue
            plate = warp_plate(img, parse_ccpd_points(src_path))
            image_name = f"{src_path.stem}.jpg"
            cv2.imwrite(str(image_dir / image_name), plate)
            write_yolo_label(label_dir / f"{src_path.stem}.txt", boxes)

    print(f"Converted {len(files)} full-car images into rectified character-box data at {out}")
    print(f"Train images: {len(splits['train'])}, val images: {len(splits['val'])}")


if __name__ == "__main__":
    main()
