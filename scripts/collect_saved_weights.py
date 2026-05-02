import argparse
import shutil
from pathlib import Path


ROLE_PATTERNS = {
    "plate": [
        "runs/yolo/plate*/weights/best.pt",
        "runs/detect/plate*/weights/best.pt",
        "runs/detect/train/weights/best.pt",
    ],
    "char": [
        "runs/yolo/char*/weights/best.pt",
        "runs/detect/char*/weights/best.pt",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Collect reusable YOLO/MCHN artifacts into saved_weights.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="./saved_weights")
    parser.add_argument("--include-last", action="store_true")
    parser.add_argument("--include-kaggle-cache", action="store_true")
    return parser.parse_args()


def newest_match(root, patterns):
    matches = []
    for pattern in patterns:
        matches.extend(path for path in root.glob(pattern) if path.is_file())
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def copy_if_exists(src, dst):
    if src is None or not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"saved {dst} <- {src}")
    return True


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)

    copied = 0
    for role, patterns in ROLE_PATTERNS.items():
        best = newest_match(root, patterns)
        copied += int(copy_if_exists(best, out / f"{role}_best.pt"))
        if args.include_last and best is not None:
            last = best.parent / "last.pt"
            copied += int(copy_if_exists(last, out / f"{role}_last.pt"))

    for cache in [
        root / "results" / "cache" / "template_cache_32x64.pt",
        root / "results" / "template_cache_32x64.pt",
        root / "saved_weights" / "template_cache_32x64.pt",
    ]:
        copied += int(copy_if_exists(cache, out / "template_cache_32x64.pt"))
        if (out / "template_cache_32x64.pt").is_file():
            break

    if args.include_kaggle_cache:
        copied += int(copy_if_exists(Path("/kaggle/working/mchn_cache/template_cache_32x64.pt"), out / "template_cache_32x64.pt"))

    if copied == 0:
        raise SystemExit("No reusable artifacts found yet. Train/evaluate first, then run this script again.")


if __name__ == "__main__":
    main()
