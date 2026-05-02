import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main_eval import display_label
from utils.metric_visuals import MetricVisualizer


def load_confusion_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        labels = [display_label(label) for label in header[1:]]
        rows = []
        for row in reader:
            rows.append([int(float(value)) for value in row[1:]])
    return np.asarray(rows, dtype=np.int64), labels


def title_from_name(path):
    name = path.stem
    if not name.startswith("confusion_"):
        return name
    rest = name[len("confusion_") :]
    parts = rest.split("_")
    if len(parts) < 2:
        return name
    pollution = parts[0]
    method = "_".join(parts[1:]).replace("_", " ")
    return f"{method} confusion matrix ({pollution})"


def main():
    parser = argparse.ArgumentParser(description="Replot confusion matrix PNG files with ASCII pinyin province labels.")
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--pattern", default="confusion_*_*.csv")
    parser.add_argument("--suffix", default="", help="Optional suffix before .png, e.g. _pinyin.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    visualizer = MetricVisualizer(save_dir=str(results_dir))
    count = 0
    for csv_path in sorted(results_dir.glob(args.pattern)):
        if csv_path.name.startswith("top_confusions_"):
            continue
        matrix, labels = load_confusion_csv(csv_path)
        output_name = f"{csv_path.stem}{args.suffix}.png"
        visualizer.plot_confusion_matrix(matrix, labels, title_from_name(csv_path), output_name)
        count += 1
        print(f"Replotted {os.path.join(args.results_dir, output_name)}")
    print(f"Done. Replotted {count} confusion matrices.")


if __name__ == "__main__":
    main()
