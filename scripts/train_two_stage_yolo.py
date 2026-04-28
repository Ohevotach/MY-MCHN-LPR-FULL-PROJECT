import argparse
import subprocess
import sys


def run(cmd):
    print(" ".join(cmd))
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser(description="Train plate detector and character detector separately.")
    parser.add_argument("--model", default="./yolo11n.pt")
    parser.add_argument("--plate-data", default="./configs/plate_detection.yaml")
    parser.add_argument("--char-data", default="./configs/char_detection.yaml")
    parser.add_argument("--plate-epochs", type=int, default=80)
    parser.add_argument("--char-epochs", type=int, default=80)
    parser.add_argument("--plate-imgsz", type=int, default=640)
    parser.add_argument("--char-imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
<<<<<<< HEAD
    parser.add_argument("--save-dir", default="./saved_weights")
=======
>>>>>>> 049f4e4ed3456bfaa618da80df38b05d7d2f1d5b
    args = parser.parse_args()

    base = [
        sys.executable,
        "scripts/train_yolo.py",
        "--model",
        args.model,
        "--batch",
        str(args.batch),
<<<<<<< HEAD
        "--save-dir",
        args.save_dir,
=======
>>>>>>> 049f4e4ed3456bfaa618da80df38b05d7d2f1d5b
    ]
    if args.device is not None:
        base += ["--device", args.device]

    run(
        base
        + [
            "--data",
            args.plate_data,
            "--epochs",
            str(args.plate_epochs),
            "--imgsz",
            str(args.plate_imgsz),
            "--name",
            "plate_yolo11n",
        ]
    )
    run(
        base
        + [
            "--data",
            args.char_data,
            "--epochs",
            str(args.char_epochs),
            "--imgsz",
            str(args.char_imgsz),
            "--name",
            "char_yolo11n",
        ]
    )


if __name__ == "__main__":
    main()
