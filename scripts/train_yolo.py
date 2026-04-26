import argparse
import os


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


if __name__ == "__main__":
    main()
