import argparse
import random
import shutil
from pathlib import Path

import cv2


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_ccpd_box(path):
    stem = Path(path).stem
    parts = stem.split("-")
    if len(parts) < 4:
        return None
    points = []
    try:
        for token in parts[3].split("_"):
            x_str, y_str = token.split("&")
            points.append((float(x_str), float(y_str)))
    except Exception:
        return None
    if len(points) != 4:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def write_yolo_label(label_path, box, width, height):
    x1, y1, x2, y2 = box
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(width - 1), x2), min(float(height - 1), y2)
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    with open(label_path, "w", encoding="utf-8") as f:
        f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def collect_images(src):
    if not src.exists():
        raise FileNotFoundError(f"Source directory not found: {src}")
    return [p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS]


def main():
    parser = argparse.ArgumentParser(description="Convert CCPD-style filenames to YOLO plate detection labels.")
    parser.add_argument("--src", required=True, help="Source image directory. You can pass ./data/full_cars.")
    parser.add_argument("--out", default="./dataset/yolo_plate")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--allow-empty", action="store_true", help="Create folders even if no CCPD labels are found.")
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    all_images = collect_images(src)
    files = [p for p in all_images if parse_ccpd_box(p) is not None]

    print(f"Source: {src}")
    print(f"Found image files: {len(all_images)}")
    print(f"Found CCPD-labeled files: {len(files)}")
    if all_images and not files:
        print("No CCPD corner labels were parsed from filenames.")
        print("Example filenames:")
        for sample in all_images[:5]:
            print(f"  - {sample.name}")
        print("Expected filename contains a field like: x1&y1_x2&y2_x3&y3_x4&y4")
    if not files and not args.allow_empty:
        raise SystemExit(
            "No labeled images to convert. Use --src ./data/full_cars if you pointed at an empty subset, "
            "or label your images in YOLO format manually if filenames do not contain CCPD corner points."
        )

    random.Random(args.seed).shuffle(files)
    val_count = max(1, int(round(len(files) * args.val_ratio))) if len(files) > 1 else 0
    splits = {"val": files[:val_count], "train": files[val_count:]}

    for split, split_files in splits.items():
        image_dir = out / "images" / split
        label_dir = out / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for src_path in split_files:
            img = cv2.imread(str(src_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            dst_img = image_dir / src_path.name
            shutil.copy2(src_path, dst_img)
            write_yolo_label(label_dir / f"{src_path.stem}.txt", parse_ccpd_box(src_path), w, h)

    print(f"Converted {len(files)} images to {out}")
    print(f"Train images: {len(splits['train'])}, val images: {len(splits['val'])}")


if __name__ == "__main__":
    main()
