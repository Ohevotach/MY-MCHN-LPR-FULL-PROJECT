# YOLO Two-Stage Training

The main pipeline is:

```text
full image -> plate_yolo11n.pt -> plate crop -> char_yolo11n.pt -> character crops -> Modern Hopfield recognition
```

Do not rely on CCPD filenames for the main workflow. Use ordinary YOLO labels.

## 1. Prepare Plate Detection Data

Create folders:

```bash
python scripts/prepare_yolo_dirs.py --out ./dataset/yolo_plate --task plate
```

Put your own full-car images and YOLO labels here:

```text
dataset/yolo_plate/images/train/a.jpg
dataset/yolo_plate/labels/train/a.txt
dataset/yolo_plate/images/val/b.jpg
dataset/yolo_plate/labels/val/b.txt
```

Each label file contains one or more plate boxes:

```text
0 cx cy w h
```

Class `0` means `license_plate`.

## 2. Prepare Character Detection Data

Create folders:

```bash
python scripts/prepare_yolo_dirs.py --out ./dataset/yolo_chars --task char
```

Use cropped/rectified plate images. Label every visible character box:

```text
dataset/yolo_chars/images/train/plate_001.jpg
dataset/yolo_chars/labels/train/plate_001.txt
```

Each label line:

```text
0 cx cy w h
```

Class `0` means `plate_char`. Do not label the dot separator as a character.

## 3. Train Separately

```bash
python scripts/train_yolo.py --model ./yolo11n.pt --data ./configs/plate_detection.yaml --epochs 80 --imgsz 640 --batch 16 --name plate_yolo11n
```

```bash
python scripts/train_yolo.py --model ./yolo11n.pt --data ./configs/char_detection.yaml --epochs 80 --imgsz 416 --batch 16 --name char_yolo11n
```

Or train both:

```bash
python scripts/train_two_stage_yolo.py --model ./yolo11n.pt --batch 16
```

## 4. Run App

```bash
export PLATE_DETECTOR_WEIGHTS=./runs/yolo/plate_yolo11n/weights/best.pt
export CHAR_DETECTOR_WEIGHTS=./runs/yolo/char_yolo11n/weights/best.pt
python app.py
```

In Kaggle notebooks, use Python environment variables:

```python
import os
os.environ["PLATE_DETECTOR_WEIGHTS"] = "./runs/yolo/plate_yolo11n/weights/best.pt"
os.environ["CHAR_DETECTOR_WEIGHTS"] = "./runs/yolo/char_yolo11n/weights/best.pt"
!python app.py
```

## Optional CCPD Helper

`scripts/ccpd_to_yolo_plate.py` is only a helper for quickly producing plate boxes from CCPD-like filenames. The app and the normal training flow do not require CCPD.
