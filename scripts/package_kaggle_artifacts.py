import argparse
import os
import zipfile
from pathlib import Path


DEFAULT_PATTERNS = [
<<<<<<< HEAD
=======
    "saved_weights/*.pt",
    "saved_weights/*.onnx",
    "saved_weights/*.yaml",
    "saved_weights/*.csv",
>>>>>>> 049f4e4ed3456bfaa618da80df38b05d7d2f1d5b
    "runs/**/weights/best.pt",
    "runs/**/weights/last.pt",
    "runs/**/*.yaml",
    "results/**/*.pt",
    "results/**/*.csv",
    "results/**/*.png",
    "results/**/*.jpg",
    "results/**/*.jpeg",
    "results/**/*.json",
    "configs/*.yaml",
    "docs/*.md",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Package Kaggle training artifacts for download without including datasets."
    )
    parser.add_argument("--root", default=".", help="Project root. Use the project folder on Kaggle.")
    parser.add_argument("--out", default="./results/kaggle_training_artifacts.zip")
    parser.add_argument(
        "--include-kaggle-cache",
        action="store_true",
        help="Also include /kaggle/working/mchn_cache/*.pt if it exists.",
    )
    return parser.parse_args()


def is_data_path(path):
    parts = {part.lower() for part in path.parts}
    return "data" in parts or "dataset" in parts and "yolo" not in parts


def iter_artifacts(root, include_kaggle_cache=False):
    root = Path(root).resolve()
    seen = set()
    for pattern in DEFAULT_PATTERNS:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen or is_data_path(resolved.relative_to(root)):
                continue
            seen.add(resolved)
            yield resolved, resolved.relative_to(root)

    if include_kaggle_cache:
        cache_root = Path("/kaggle/working/mchn_cache")
        if cache_root.exists():
            for path in cache_root.glob("*.pt"):
                if path.is_file():
                    resolved = path.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        yield resolved, Path("kaggle_working_mchn_cache") / path.name


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    artifacts = list(iter_artifacts(root, include_kaggle_cache=args.include_kaggle_cache))
    if not artifacts:
        raise SystemExit("No artifacts found. Train YOLO/evaluate first, then run this script again.")

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arcname in artifacts:
            zf.write(src, arcname.as_posix())

    total_mb = sum(os.path.getsize(src) for src, _ in artifacts) / (1024 * 1024)
    print(f"Saved: {out_path}")
    print(f"Files: {len(artifacts)}")
    print(f"Uncompressed size: {total_mb:.2f} MB")
    for _, arcname in artifacts:
        print(f"  - {arcname.as_posix()}")


if __name__ == "__main__":
    main()
