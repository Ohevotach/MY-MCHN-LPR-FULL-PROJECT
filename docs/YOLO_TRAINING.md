# YOLO Two-Stage Training

The main pipeline is:

```text
full image -> plate_yolo11n.pt -> plate crop -> char_yolo11n.pt -> character crops -> Modern Hopfield recognition
```

The application is YOLO-first. OpenCV color/morphology plate localization is disabled by default and is only used when:

```bash
export PLATE_OPENCV_FALLBACK=1
```

On Windows PowerShell:

```powershell
$env:PLATE_OPENCV_FALLBACK="1"
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

For CCPD-style full-car images, you can generate an initial real character-box dataset without manual labeling. The helper rectifies each plate from the filename coordinates and writes seven approximate character boxes:

```bash
python scripts/ccpd_to_yolo_chars.py --src ./data/full_cars/ccpd_base --out ./dataset/yolo_chars --val-ratio 0.2
```

This is much more reliable than OpenCV threshold segmentation for blurry or reflective plates. The labels are slot-based, so inspect a few generated plate crops before long training. If the plate is badly tilted or the filename coordinates are wrong, remove those samples or relabel them manually.

## 3. Train Separately

```bash
python scripts/train_yolo.py --model ./yolo11n.pt --data ./configs/plate_detection.yaml --epochs 80 --imgsz 640 --batch 16 --name plate_yolo11n
```

```bash
python scripts/train_yolo.py --model ./yolo11n.pt --data ./configs/char_detection.yaml --epochs 80 --imgsz 416 --batch 16 --name char_yolo11n
```

For about 200 CCPD images, use a smaller batch if needed and train a little longer:

```bash
python scripts/train_yolo.py --model ./yolo11n.pt --data ./configs/char_detection.yaml --epochs 120 --imgsz 416 --batch 8 --name char_yolo11n_real
```

Or train both:

```bash
python scripts/train_two_stage_yolo.py --model ./yolo11n.pt --batch 16
```

## How Many Plate Images?

For a single-class license-plate detector, a small YOLO model can start working with fewer images than a full multi-class detector, but the image diversity matters more than the raw count.

Practical guidance:

```text
200-300 images: usable prototype if labels are accurate and scenes are similar.
500-1000 images: recommended minimum for a steadier demo.
2000+ images: better for different weather, blur, camera angles, distances, and plate colors.
```

If `data/full_cars/ccpd_base` has only about 200 images, use it as the first training set, but keep expectations modest. It may detect plates in similar CCPD-style images, yet fail on blur, night, strong angle, green new-energy plates, cropped cars, or very small plates.

Suggested first run for about 200 images:

```bash
python scripts/ccpd_to_yolo_plate.py --src ./data/full_cars/ccpd_base --out ./dataset/yolo_plate --val-ratio 0.2
python scripts/train_yolo.py --model ./yolo11n.pt --data ./configs/plate_detection.yaml --epochs 120 --imgsz 640 --batch 8 --name plate_yolo11n_ccpd_base
```

After training, reusable checkpoints are copied to `./saved_weights`:

```text
saved_weights/plate_yolo11n_ccpd_base_best.pt
saved_weights/plate_yolo11n_ccpd_base_last.pt
saved_weights/plate_yolo11n_ccpd_base_optimizer.pt
```

The Modern Hopfield associative memory is also saved in `saved_weights` when you run the app or evaluation:

```text
saved_weights/mchn_memory_32x64.pt
saved_weights/mchn_eval_memory_32x64.pt
```

Use the `*_last.pt` file when you want to continue training:

```bash
python scripts/train_yolo.py --model ./saved_weights/plate_yolo11n_ccpd_base_last.pt --resume --data ./configs/plate_detection.yaml --epochs 160 --imgsz 640 --batch 8 --name plate_yolo11n_ccpd_base
```

On Windows PowerShell, run the app with the trained detector:

```powershell
$env:PLATE_DETECTOR_WEIGHTS="./saved_weights/plate_yolo11n_ccpd_base_best.pt"
python app.py
```

If validation precision is high but recall is low, add more images with small/tilted/dirty plates. If validation is unstable, inspect labels first; with only 200 images, a few bad boxes can noticeably hurt training.

## 4. Run App

```bash
export PLATE_DETECTOR_WEIGHTS=./saved_weights/plate_yolo11n_best.pt
export CHAR_DETECTOR_WEIGHTS=./saved_weights/char_yolo11n_real_best.pt
python app.py
```

If these variables are not set, the app searches `./saved_weights` first and then falls back to `./runs`.

If no plate detector weights are set, the app will not silently rely on OpenCV plate localization. To allow the old rule-based fallback for comparison only:

```bash
export PLATE_OPENCV_FALLBACK=1
```

In Kaggle notebooks, use Python environment variables:

```python
import os
os.environ["PLATE_DETECTOR_WEIGHTS"] = "./saved_weights/plate_yolo11n_best.pt"
os.environ["CHAR_DETECTOR_WEIGHTS"] = "./saved_weights/char_yolo11n_real_best.pt"
!python app.py
```

## Optional CCPD Helper

`scripts/ccpd_to_yolo_plate.py` is only a helper for quickly producing plate boxes from CCPD-like filenames. The app and the normal training flow do not require CCPD.
