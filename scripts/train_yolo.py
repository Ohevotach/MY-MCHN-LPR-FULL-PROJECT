import argparse
import os
import shutil
from pathlib import Path

import torch
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
    parser.add_argument("--save-dir", default="./saved_weights", help="Directory for reusable weights and optimizer states.")
    parser.add_argument("--resume", action="store_true", help="Resume training from --model, usually a saved *_last.pt file.")
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
    if args.resume:
        train_kwargs["resume"] = True
    results = model.train(**train_kwargs)
    archive_training_artifacts(results, args)


def archive_training_artifacts(results, args):
    run_dir = resolve_run_dir(results, args)
    if run_dir is None:
        print("Warning: could not resolve YOLO run directory; saved_weights was not updated.")
        return

    weights_dir = run_dir / "weights"
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for weight_name in ("best.pt", "last.pt"):
        src = weights_dir / weight_name
        if src.exists():
            dst = save_dir / f"{args.name}_{weight_name}"
            shutil.copy2(src, dst)
            print(f"Saved reusable weight: {dst}")
        else:
            print(f"Warning: YOLO weight not found: {src}")

    optimizer_src = weights_dir / "optimizer.pt"
    if optimizer_src.exists():
        dst = save_dir / f"{args.name}_optimizer.pt"
        shutil.copy2(optimizer_src, dst)
        print(f"Saved optimizer state: {dst}")
        return

    last_weight = weights_dir / "last.pt"
    if last_weight.exists():
        optimizer_dst = save_dir / f"{args.name}_optimizer.pt"
        if extract_optimizer_state(last_weight, optimizer_dst):
            print(f"Saved optimizer state: {optimizer_dst}")
        else:
            print("Warning: optimizer state was not found in last.pt; last.pt can still be used for YOLO resume.")


def resolve_run_dir(results, args):
    save_dir = getattr(results, "save_dir", None)
    if save_dir:
        return Path(save_dir)

    trainer = getattr(results, "trainer", None)
    save_dir = getattr(trainer, "save_dir", None)
    if save_dir:
        return Path(save_dir)

    expected = Path(args.project) / args.name
    if expected.exists():
        return expected

    matches = sorted(Path(args.project).glob(f"{args.name}*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def extract_optimizer_state(checkpoint_path, output_path):
    try:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        print(f"Warning: failed to read checkpoint for optimizer extraction: {exc}")
        return False

    if not isinstance(checkpoint, dict) or checkpoint.get("optimizer") is None:
        return False

    torch.save(
        {
            "optimizer": checkpoint.get("optimizer"),
            "epoch": checkpoint.get("epoch"),
            "train_args": checkpoint.get("train_args"),
            "date": checkpoint.get("date"),
        },
        output_path,
    )
    return True


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


if __name__ == "__main__":
    main()
