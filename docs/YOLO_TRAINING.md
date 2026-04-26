# YOLO Training Guide

This project uses Modern Hopfield for character recognition. YOLO is only used to make the front end reliable:

1. `plate detector`: finds the license plate box in the full car image.
2. `character detector`: finds seven character boxes on a rectified plate.

## Recommended Models

- Use your `yolo11n.pt` as the starting model for plate detection.
- Also use `yolo11n.pt` for character-box detection. This is a one-class detector named `plate_char`, not a recognizer.

## Plate Detection Dataset

YOLO labels need one box per image:

```text
class cx cy w h
0 0.512 0.733 0.312 0.098
```

If your full-car images use CCPD-style filenames with four plate corner points, convert them with the whole `full_cars` folder:

```powershell
python scripts\ccpd_to_yolo_plate.py --src .\data\full_cars --out .\dataset\yolo_plate
```

If it says `Found CCPD-labeled files: 0`, your folder is empty or the filenames do not contain CCPD corner labels. In that case you need manually labeled YOLO files under `dataset/yolo_plate/images/...` and `dataset/yolo_plate/labels/...`.

Then train:

```powershell
python scripts\train_yolo.py --model .\yolo11n.pt --data .\configs\plate_detection.yaml --epochs 100 --imgsz 640 --batch 16 --name plate_yolo11n
```

After training, use the best checkpoint:

```powershell
$env:PLATE_DETECTOR_WEIGHTS=".\runs\yolo\plate_yolo11n\weights\best.pt"
python app.py
```

## Character Box Dataset

The easiest first version is synthetic training from your existing character templates:

```powershell
python scripts\generate_synthetic_char_yolo.py --chars2 .\data\chars2 --chars-chinese .\data\charsChinese --out .\dataset\yolo_chars_synth --count 8000
```

Train a one-class character detector:

```powershell
python scripts\train_yolo.py --model .\yolo11n.pt --data .\configs\char_detection.yaml --epochs 80 --imgsz 416 --batch 32 --name char_yolo11n
```

Use the character detector in the app:

```powershell
$env:CHAR_DETECTOR_WEIGHTS=".\runs\yolo\char_yolo11n\weights\best.pt"
python app.py
```

If `CHAR_DETECTOR_WEIGHTS` is not set, the app uses geometry-based slots on the rectified plate.

## Practical Order

1. Train plate detection first. This fixes most current errors.
2. Keep geometry slot splitting as the first character segmentation method.
3. Train character-box detection only if slot splitting still fails on tilted or badly cropped plates.
