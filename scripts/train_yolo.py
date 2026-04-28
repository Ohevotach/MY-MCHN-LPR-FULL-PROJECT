import argparse
import os
import shutil
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO11 models for plate or character detection.")
    parser.add_argument("--model", default="yolo11n.pt", help="Path to YOLO11 checkpoint, e.g. yolo11n.pt.")
    parser.add_argument("--data", required=True, help="YOLO dataset yaml path.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="./runs/yolo")
    parser.add_argument("--name", default="train")
    parser.add_argument("--device", default=None, help="Examples: 0, cpu. Leave unset for auto.")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--save-artifacts-dir", default="./saved_weights")
    parser.add_argument("--artifact-role", choices=["auto", "plate", "char"], default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Please install ultralytics first: pip install ultralytics") from exc

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset yaml not found: {args.data}")
    check_yolo_dataset(args.data)

    model = YOLO(args.model)
    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "patience": args.patience,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device
    model.train(**train_kwargs)
    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    if save_dir is None:
        save_dir = Path(args.project) / args.name
    copy_training_artifacts(Path(save_dir), args)


def check_yolo_dataset(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    root = Path(data.get("path", "."))
    if not root.is_absolute():
        root = Path.cwd() / root
    train_dir = root / data.get("train", "")
    val_dir = root / data.get("val", "")
    train_images = list_images(train_dir)
    val_images = list_images(val_dir)
    print(f"Dataset root: {root}")
    print(f"Train images: {len(train_images)} ({train_dir})")
    print(f"Val images: {len(val_images)} ({val_dir})")
    if not train_images:
        raise SystemExit(
            "YOLO train set is empty. Put ordinary YOLO images/labels under dataset/.../images/train "
            "and dataset/.../labels/train, or check the config path."
        )
    if not val_images:
        raise SystemExit("YOLO val set is empty. Add validation images or lower --val-ratio conversion settings.")


def list_images(path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not path.exists():
        return []
    return [p for p in path.rglob("*") if p.suffix.lower() in exts]


def infer_artifact_role(args):
    if args.artifact_role != "auto":
        return args.artifact_role
    text = f"{args.name} {args.data}".lower()
    if "char" in text:
        return "char"
    if "plate" in text or "license" in text:
        return "plate"
    return "model"


def copy_training_artifacts(save_dir, args):
    role = infer_artifact_role(args)
    out_dir = Path(args.save_artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    weights_dir = save_dir / "weights"
    for src_name, suffix in (("best.pt", "best.pt"), ("last.pt", "last.pt")):
        src = weights_dir / src_name
        if src.is_file():
            dst = out_dir / f"{role}_{suffix}"
            shutil.copy2(src, dst)
            copied.append(dst)

    for src_name in ("args.yaml", "results.csv"):
        src = save_dir / src_name
        if src.is_file():
            dst = out_dir / f"{role}_{src_name}"
            shutil.copy2(src, dst)
            copied.append(dst)

    if copied:
        print("Saved reusable training artifacts:")
        for path in copied:
            print(f"  - {path}")
    else:
        print(f"Warning: no reusable training artifacts found under {save_dir}")


if __name__ == "__main__":
    main()
