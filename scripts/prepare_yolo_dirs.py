import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Create standard YOLO folder structure.")
    parser.add_argument("--out", required=True, help="Output dataset directory.")
    parser.add_argument("--task", choices=["plate", "char"], default="plate")
    args = parser.parse_args()

    root = Path(args.out)
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    class_name = "license_plate" if args.task == "plate" else "plate_char"
    print(f"Created YOLO folders at {root}")
    print("Put images and labels like this:")
    print(f"  {root}/images/train/xxx.jpg")
    print(f"  {root}/labels/train/xxx.txt")
    print("Each label line:")
    print(f"  0 cx cy w h    # class 0 = {class_name}")


if __name__ == "__main__":
    main()
