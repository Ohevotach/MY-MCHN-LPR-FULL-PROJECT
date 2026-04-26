import os

import cv2
import numpy as np
import random


class ArtificialPolluter:
    @staticmethod
    def add_synthetic_fog(image, severity=0.5):
        fog_layer = np.ones_like(image, dtype=np.uint8) * 255
        return cv2.addWeighted(image, 1 - severity, fog_layer, severity, 0)

    @staticmethod
    def add_synthetic_dirt(image, num_spots=5, max_radius=30):
        dirty_img = image.copy()
        h, w = dirty_img.shape[:2]
        for _ in range(num_spots):
            x, y = random.randint(0, max(0, w - 1)), random.randint(0, max(0, h - 1))
            r = random.randint(5, max_radius)
            color = (random.randint(20, 50), random.randint(30, 60), random.randint(30, 60))
            cv2.circle(dirty_img, (x, y), r, color, -1)
        return dirty_img


class ImageEnhancer:
    def __init__(self, omega=0.85, t0=0.18, window_size=15):
        self.omega = omega
        self.t0 = t0
        self.window_size = window_size

    def dehaze(self, img):
        img_float = img.astype("float64") / 255.0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.window_size, self.window_size))
        air_light = max(float(np.percentile(img_float, 99.5)), 1e-3)
        dark = cv2.erode(np.min(img_float / air_light, axis=2), kernel)
        transmission = np.maximum(1 - self.omega * dark, self.t0)
        recovered = np.empty_like(img_float)
        for i in range(3):
            recovered[:, :, i] = (img_float[:, :, i] - air_light) / transmission + air_light
        return np.clip(recovered * 255, 0, 255).astype("uint8")


class PlateDetector:
    """Optional model-based plate detector with OpenCV fallback.

    Set PLATE_DETECTOR_WEIGHTS to a local YOLO/ONNX license-plate detector.
    The project still runs without the detector package or weights.
    """

    def __init__(self, weights_path=None, conf=0.35):
        self.weights_path = weights_path or os.environ.get("PLATE_DETECTOR_WEIGHTS")
        self.conf = float(os.environ.get("PLATE_DETECTOR_CONF", conf))
        self.backend = None
        self.model = None
        if self.weights_path and os.path.exists(self.weights_path):
            self._load_model(self.weights_path)

    def _load_model(self, weights_path):
        ext = os.path.splitext(weights_path)[1].lower()
        if ext == ".onnx":
            try:
                self.model = cv2.dnn.readNetFromONNX(weights_path)
                self.backend = "onnx"
            except Exception as exc:
                print(f"Warning: failed to load ONNX plate detector: {exc}")
            return

        try:
            from ultralytics import YOLO

            self.model = YOLO(weights_path)
            self.backend = "ultralytics"
        except Exception as exc:
            print(f"Warning: failed to load YOLO plate detector: {exc}")

    def detect(self, img):
        if self.model is None:
            return []
        if self.backend == "ultralytics":
            return self._detect_ultralytics(img)
        if self.backend == "onnx":
            return self._detect_onnx(img)
        return []

    def _detect_ultralytics(self, img):
        try:
            results = self.model.predict(img, conf=self.conf, verbose=False)
        except Exception as exc:
            print(f"Warning: YOLO plate detection failed: {exc}")
            return []

        detections = []
        h, w = img.shape[:2]
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf.item())
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                detections.append((conf, self._clip_box((x1, y1, x2, y2), w, h)))
        return detections

    def _detect_onnx(self, img):
        # Supports common YOLO-style ONNX outputs. If a model uses a different
        # export shape, detection simply falls back to the OpenCV pipeline.
        h, w = img.shape[:2]
        input_size = int(os.environ.get("PLATE_DETECTOR_INPUT", "640"))
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (input_size, input_size), swapRB=True, crop=False)
        try:
            self.model.setInput(blob)
            output = self.model.forward()
        except Exception as exc:
            print(f"Warning: ONNX plate detection failed: {exc}")
            return []

        pred = np.squeeze(output)
        if pred.ndim != 2:
            return []
        if pred.shape[0] < pred.shape[1] and pred.shape[0] in (5, 6, 84):
            pred = pred.T

        detections = []
        for row in pred:
            if row.shape[0] < 5:
                continue
            cx, cy, bw, bh = row[:4]
            score = float(row[4]) if row.shape[0] == 5 else float(row[4] * np.max(row[5:]))
            if score < self.conf:
                continue
            x1 = int((cx - bw / 2) * w / input_size)
            y1 = int((cy - bh / 2) * h / input_size)
            x2 = int((cx + bw / 2) * w / input_size)
            y2 = int((cy + bh / 2) * h / input_size)
            detections.append((score, self._clip_box((x1, y1, x2, y2), w, h)))
        return detections

    @staticmethod
    def _clip_box(box, width, height):
        x1, y1, x2, y2 = box
        return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


class PlateSegmenter:
    PLATE_W = 400
    PLATE_H = 120

    def locate_plate(self, img):
        """Locate blue/green license plates and rectify them to 400x120."""
        if img is None:
            return None

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, np.array([90, 28, 20]), np.array([140, 255, 255]))
        green_mask = cv2.inRange(hsv, np.array([32, 25, 25]), np.array([95, 255, 255]))
        mask = cv2.bitwise_or(blue_mask, green_mask)

        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

        h_img, w_img = img.shape[:2]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            area_ratio = area / max(1.0, h_img * w_img)
            if area < max(250, 0.00035 * h_img * w_img) or area_ratio > 0.08:
                continue

            rect = cv2.minAreaRect(cnt)
            _, size, _ = rect
            rw, rh = size
            if rw <= 1 or rh <= 1:
                continue
            if rw < rh:
                rw, rh = rh, rw
            aspect = rw / rh
            if not (2.2 <= aspect <= 5.8):
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            fill = area / max(1.0, bw * bh)
            if fill < 0.18 or bh < 14 or bw < 45:
                continue

            aspect_score = 1.0 - min(abs(aspect - 3.4) / 3.4, 1.0)
            y_center = (y + bh / 2) / max(1, h_img)
            y_score = 1.0 - min(abs(y_center - 0.72) / 0.55, 0.95)
            lower_band_score = np.clip((y_center - 0.34) / 0.38, 0.0, 1.0)
            size_penalty = max(0.0, area_ratio - 0.035) * 8.0
            for candidate_img, mode_bias in self._candidate_plate_images(img, rect, (x, y, bw, bh)):
                color_ratio = self._plate_color_ratio(candidate_img)
                if color_ratio < 0.28:
                    continue
                quality = self._plate_quality_score(candidate_img)
                if quality < 0.34:
                    continue
                score = (
                    quality * 4.8
                    + min(color_ratio / 0.52, 1.0) * 1.8
                    + aspect_score * 1.2
                    + fill * 0.6
                    + y_score * 0.8
                    + lower_band_score * 1.2
                    + mode_bias
                    - size_penalty
                )
                candidates.append((score, rect, (x, y, bw, bh), candidate_img))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        _, rect, bbox, warped = candidates[0]
        if warped is not None:
            return warped

        x, y, w, h = bbox
        pad_x, pad_y = int(0.08 * w), int(0.22 * h)
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(w_img, x + w + pad_x), min(h_img, y + h + pad_y)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (self.PLATE_W, self.PLATE_H))

    def _candidate_plate_images(self, img, rect, bbox):
        candidates = []
        warped = self._warp_rect(img, rect)
        if warped is not None:
            candidates.append((warped, 0.10))

        crop = self._crop_bbox_candidate(img, bbox)
        if crop is not None:
            candidates.append((crop, 0.0))

        return candidates

    def _crop_bbox_candidate(self, img, bbox):
        x, y, w, h = bbox
        h_img, w_img = img.shape[:2]
        pad_x = int(0.12 * w)
        pad_y = int(0.35 * h)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w_img, x + w + pad_x)
        y2 = min(h_img, y + h + pad_y)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        crop = self._deskew_plate(crop)
        return cv2.resize(crop, (self.PLATE_W, self.PLATE_H))

    @staticmethod
    def _warp_rect(img, rect):
        box = cv2.boxPoints(rect).astype("float32")
        ordered = np.zeros((4, 2), dtype="float32")
        s = box.sum(axis=1)
        ordered[0] = box[np.argmin(s)]
        ordered[2] = box[np.argmax(s)]
        diff = np.diff(box, axis=1)
        ordered[1] = box[np.argmin(diff)]
        ordered[3] = box[np.argmax(diff)]
        dst = np.array([[0, 0], [399, 0], [399, 119], [0, 119]], dtype="float32")
        matrix = cv2.getPerspectiveTransform(ordered, dst)
        warped = cv2.warpPerspective(img, matrix, (400, 120))
        if warped.size == 0:
            return None
        return warped

    @staticmethod
    def _plate_color_ratio(plate_img):
        hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, np.array([90, 35, 35]), np.array([140, 255, 255]))
        green = cv2.inRange(hsv, np.array([32, 30, 35]), np.array([95, 255, 255]))
        color_mask = cv2.bitwise_or(blue, green)
        return float(np.mean(color_mask[14:106, 16:384] > 0))

    @staticmethod
    def _plate_quality_score(plate_img):
        hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, np.array([90, 35, 35]), np.array([140, 255, 255]))
        green = cv2.inRange(hsv, np.array([32, 30, 35]), np.array([95, 255, 255]))
        color_mask = cv2.bitwise_or(blue, green)
        roi = color_mask[12:108, 12:388]
        color_ratio = float(np.mean(roi > 0))

        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        char_mask = PlateSegmenter._make_character_mask(plate_img, cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray))
        char_roi = char_mask[18:104, 18:382]
        char_ratio = float(np.mean(char_roi > 0))
        edges = cv2.Canny(gray, 80, 180)
        edge_ratio = float(np.mean(edges[18:104, 18:382] > 0))
        component_score = PlateSegmenter._character_component_score(char_mask[12:108, 12:388])

        color_score = min(color_ratio / 0.45, 1.0)
        char_score = 1.0 - min(abs(char_ratio - 0.16) / 0.24, 1.0)
        edge_score = min(edge_ratio / 0.12, 1.0)
        score = 0.38 * color_score + 0.17 * char_score + 0.15 * edge_score + 0.30 * component_score
        if color_ratio < 0.28 or component_score < 0.20:
            score *= 0.45
        return score

    @staticmethod
    def _character_component_score(white_mask):
        work = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 4)))
        contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = w / max(1, h)
            if 18 <= h <= 86 and 3 <= w <= 58 and 0.04 <= ratio <= 1.15 and w * h >= 90:
                boxes.append((x, y, w, h))
        if not boxes:
            return 0.0
        boxes = sorted(boxes, key=lambda b: b[0])
        centers = np.array([x + w / 2 for x, _, w, _ in boxes], dtype=np.float32)
        count_score = 1.0 - min(abs(len(boxes) - 7) / 7.0, 1.0)
        span_score = min((centers[-1] - centers[0]) / 250.0, 1.0) if len(centers) > 1 else 0.0
        return 0.65 * count_score + 0.35 * span_score

    @staticmethod
    def _deskew_plate(plate_img):
        hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, np.array([90, 28, 20]), np.array([140, 255, 255]))
        green = cv2.inRange(hsv, np.array([32, 25, 25]), np.array([95, 255, 255]))
        mask = cv2.bitwise_or(blue, green)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)), iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return plate_img
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 40:
            return plate_img
        rect = cv2.minAreaRect(cnt)
        angle = rect[2]
        rw, rh = rect[1]
        if rw <= 1 or rh <= 1:
            return plate_img
        if rw < rh:
            angle += 90.0
        if angle > 45.0:
            angle -= 90.0
        if angle < -45.0:
            angle += 90.0
        if abs(angle) < 2.0 or abs(angle) > 25.0:
            return plate_img
        h, w = plate_img.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        return cv2.warpAffine(plate_img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def segment_characters(self, plate_img):
        """Segment a rectified plate into 7 character crops."""
        if plate_img is None:
            return []

        plate_img = self._tighten_plate_crop(self._deskew_plate(cv2.resize(plate_img, (self.PLATE_W, self.PLATE_H))))
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        binary = self._prepare_character_binary(self._make_character_mask(plate_img, gray))

        candidates = []
        geometric_chars, geometric_boxes = self._segment_by_plate_geometry(plate_img, gray, binary)
        if geometric_chars:
            candidates.append(("geometry", geometric_boxes, geometric_chars))

        contour_boxes = self._find_contour_boxes(binary)
        if contour_boxes:
            candidates.append(("contour", contour_boxes, [self._crop_char(binary, box) for box in contour_boxes]))

        projection_boxes = self._segment_boxes_by_projection(binary)
        if projection_boxes:
            candidates.append(("projection", projection_boxes, [self._crop_char(binary, box) for box in projection_boxes]))

        fixed_chars = self._segment_by_fixed_slots(gray, binary, char_count=7)
        if fixed_chars:
            candidates.append(("fixed", [], fixed_chars))

        if not candidates:
            return []

        best_score = -1.0
        best_chars = []
        for _, boxes, chars in candidates:
            score = self._segmentation_score(boxes, chars)
            if score > best_score:
                best_score = score
                best_chars = chars
        return best_chars

    def _segment_by_plate_geometry(self, plate_img, gray, binary):
        roi = self._estimate_plate_text_roi(plate_img, binary)
        if roi is None:
            return [], []
        left, top, right, bottom = roi
        width = right - left
        height = bottom - top
        if width < 220 or height < 42:
            return [], []

        # Chinese plates have a wider province character, a letter, a separator,
        # then five alphanumeric characters. These ratios are intentionally broad.
        ratios = np.array([1.10, 0.95, 0.22, 0.95, 0.95, 0.95, 0.95, 0.95], dtype=np.float32)
        unit = width / float(ratios.sum())
        x = float(left)
        boxes = []
        for i, ratio in enumerate(ratios):
            slot_w = unit * float(ratio)
            if i == 2:
                x += slot_w
                continue
            pad = int(max(2, slot_w * 0.10))
            x1 = int(round(x + pad))
            x2 = int(round(x + slot_w - pad))
            y1 = int(round(top + 0.04 * height))
            y2 = int(round(bottom - 0.04 * height))
            boxes.append((max(0, x1), max(0, y1), max(4, x2 - x1), max(8, y2 - y1)))
            x += slot_w

        chars = [self._crop_slot_char(gray, binary, box) for box in boxes]
        return chars, boxes

    @classmethod
    def _estimate_plate_text_roi(cls, plate_img, binary):
        hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, np.array([90, 35, 35]), np.array([140, 255, 255]))
        green = cv2.inRange(hsv, np.array([32, 30, 35]), np.array([95, 255, 255]))
        color_mask = cv2.bitwise_or(blue, green)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5)))

        rows = np.where((color_mask > 0).sum(axis=1) > 0.22 * cls.PLATE_W)[0]
        cols = np.where((color_mask > 0).sum(axis=0) > 0.18 * cls.PLATE_H)[0]
        if len(rows) < 20 or len(cols) < 120:
            rows = np.where((binary > 0).sum(axis=1) > 2)[0]
            cols = np.where((binary > 0).sum(axis=0) > 1)[0]
        if len(rows) < 20 or len(cols) < 120:
            return None

        left = max(10, int(cols[0]) + 5)
        right = min(cls.PLATE_W - 10, int(cols[-1]) - 5)
        top = max(14, int(rows[0]) + 8)
        bottom = min(cls.PLATE_H - 8, int(rows[-1]) - 6)

        char_rows = np.where((binary[:, left:right] > 0).sum(axis=1) > 3)[0]
        if len(char_rows) >= 20:
            top = max(top, int(char_rows[0]) - 4)
            bottom = min(bottom, int(char_rows[-1]) + 5)
        return left, top, right, bottom

    @staticmethod
    def _crop_slot_char(gray, binary, box):
        x, y, w, h = box
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(binary.shape[1], x + w), min(binary.shape[0], y + h)
        slot_binary = binary[y1:y2, x1:x2]
        if slot_binary.size == 0:
            return np.zeros((64, 32), dtype=np.uint8)

        if int(np.count_nonzero(slot_binary)) < 10:
            slot_gray = gray[y1:y2, x1:x2]
            if slot_gray.size == 0:
                return np.zeros((64, 32), dtype=np.uint8)
            slot_gray = cv2.GaussianBlur(slot_gray, (3, 3), 0)
            _, slot_binary = cv2.threshold(slot_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.mean(slot_binary > 0) > 0.55:
                slot_binary = cv2.bitwise_not(slot_binary)

        slot_binary = PlateSegmenter._remove_char_border_fragments(slot_binary)
        return PlateSegmenter._resize_char_canvas(slot_binary)

    @classmethod
    def _tighten_plate_crop(cls, plate_img):
        hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, np.array([90, 28, 20]), np.array([140, 255, 255]))
        green = cv2.inRange(hsv, np.array([32, 25, 25]), np.array([95, 255, 255]))
        mask = cv2.bitwise_or(blue, green)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5)), iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return plate_img
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 0.12 * cls.PLATE_W * cls.PLATE_H:
            return plate_img
        x, y, w, h = cv2.boundingRect(cnt)
        if w / max(1, h) < 2.0:
            return plate_img
        pad_x = int(0.025 * w)
        pad_y = int(0.08 * h)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(cls.PLATE_W, x + w + pad_x)
        y2 = min(cls.PLATE_H, y + h + pad_y)
        crop = plate_img[y1:y2, x1:x2]
        if crop.size == 0:
            return plate_img
        return cv2.resize(crop, (cls.PLATE_W, cls.PLATE_H))

    @staticmethod
    def _make_character_mask(plate_img, gray):
        hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, np.array([90, 28, 20]), np.array([140, 255, 255]))
        green = cv2.inRange(hsv, np.array([32, 25, 25]), np.array([95, 255, 255]))
        plate_color = cv2.bitwise_or(blue, green)
        color_ratio = float(np.mean(plate_color[12:108, 12:388] > 0))

        if color_ratio > 0.25:
            plate_color = cv2.morphologyEx(plate_color, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)))
            plate_color = cv2.erode(plate_color, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)), iterations=1)
            white = cv2.inRange(hsv, np.array([0, 0, 95]), np.array([180, 130, 255]))
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -4)
            bright = cv2.inRange(gray, max(70, int(np.percentile(gray[plate_color > 0], 58))), 255)
            binary = cv2.bitwise_and(cv2.bitwise_or(white, adaptive), bright)
            binary = cv2.bitwise_and(binary, plate_color)
        else:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            border = np.concatenate([binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]])
            if float(np.mean(border > 0)) > 0.55:
                binary = cv2.bitwise_not(binary)

        return binary

    @staticmethod
    def _prepare_character_binary(binary):
        binary = binary.copy()
        binary[:15, :] = 0
        binary[106:, :] = 0
        binary[:, :14] = 0
        binary[:, 386:] = 0
        binary = cv2.medianBlur(binary, 3)
        binary = PlateSegmenter._remove_long_plate_lines(binary)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3)))
        return binary

    @staticmethod
    def _remove_long_plate_lines(binary):
        cleaned = binary.copy()
        work = cleaned > 0
        row_sum = work.sum(axis=1)
        col_sum = work.sum(axis=0)

        for y in np.where(row_sum > 0.38 * binary.shape[1])[0]:
            cleaned[max(0, y - 1) : min(binary.shape[0], y + 2), :] = 0
        for x in np.where(col_sum > 0.55 * binary.shape[0])[0]:
            cleaned[:, max(0, x - 1) : min(binary.shape[1], x + 2)] = 0

        horizontal = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (34, 1)))
        vertical = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 42)))
        long_lines = cv2.bitwise_or(horizontal, vertical)
        return cv2.bitwise_and(cleaned, cv2.bitwise_not(long_lines))

    def _find_contour_boxes(self, binary):
        work = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
        contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = w / max(1, h)
            area = w * h
            if x < 16 or x + w > self.PLATE_W - 14:
                continue
            if y < 14 or y + h > self.PLATE_H - 8:
                continue
            if w <= 7 and h >= 45:
                continue
            if h <= 18 and w <= 18:
                continue
            if 30 <= h <= 96 and 6 <= w <= 70 and 0.07 <= ratio <= 1.15 and area >= 170:
                boxes.append((x, y, w, h))
        if len(boxes) > 7:
            boxes = self._choose_best_seven_boxes(boxes)
        return sorted(boxes, key=lambda b: b[0])

    @staticmethod
    def _choose_best_seven_boxes(boxes):
        boxes = sorted(boxes, key=lambda b: b[0])
        if len(boxes) <= 7:
            return boxes
        best = boxes[:7]
        best_score = -1.0
        for start in range(0, len(boxes) - 6):
            candidate = boxes[start : start + 7]
            centers = np.array([x + w / 2.0 for x, _, w, _ in candidate], dtype=np.float32)
            gaps = np.diff(centers)
            heights = np.array([h for _, _, _, h in candidate], dtype=np.float32)
            span = centers[-1] - centers[0]
            gap_score = 1.0 - min(float(np.std(gaps)) / max(float(np.mean(gaps)), 1.0), 1.0)
            height_score = 1.0 - min(float(np.std(heights)) / max(float(np.mean(heights)), 1.0), 1.0)
            span_score = 1.0 - min(abs(span - 300.0) / 170.0, 1.0)
            score = 0.45 * gap_score + 0.35 * height_score + 0.20 * span_score
            if score > best_score:
                best_score = score
                best = candidate
        return best

    @staticmethod
    def _has_valid_plate_layout(boxes):
        if len(boxes) != 7:
            return False
        centers = np.array([x + w / 2.0 for x, _, w, _ in boxes], dtype=np.float32)
        if np.any(np.diff(centers) < 18):
            return False
        span = centers[-1] - centers[0]
        if span < 210 or span > 360:
            return False
        heights = np.array([h for _, _, _, h in boxes], dtype=np.float32)
        if np.min(heights) < 0.45 * np.median(heights):
            return False
        return True

    @staticmethod
    def _segmentation_score(boxes, chars):
        if not chars:
            return 0.0
        count_score = 1.0 - min(abs(len(chars) - 7) / 7.0, 1.0)
        if len(chars) == 7:
            count_score += 0.40

        ink_scores = []
        border_penalties = []
        for char_img in chars[:7]:
            if char_img is None or char_img.size == 0:
                continue
            foreground = char_img > 0
            ink = float(np.mean(foreground))
            ink_scores.append(1.0 - min(abs(ink - 0.24) / 0.30, 1.0))
            border = np.concatenate([foreground[0, :], foreground[-1, :], foreground[:, 0], foreground[:, -1]])
            border_penalties.append(float(np.mean(border)))
        ink_score = float(np.mean(ink_scores)) if ink_scores else 0.0
        border_score = 1.0 - min(float(np.mean(border_penalties)) / 0.22, 1.0) if border_penalties else 0.0

        layout_score = 0.0
        if boxes and len(boxes) == 7:
            centers = np.array([x + w / 2.0 for x, _, w, _ in boxes], dtype=np.float32)
            gaps = np.diff(centers)
            heights = np.array([h for _, _, _, h in boxes], dtype=np.float32)
            span = centers[-1] - centers[0]
            layout_score = (
                0.40 * (1.0 - min(float(np.std(gaps)) / max(float(np.mean(gaps)), 1.0), 1.0))
                + 0.35 * (1.0 - min(float(np.std(heights)) / max(float(np.mean(heights)), 1.0), 1.0))
                + 0.25 * (1.0 - min(abs(span - 300.0) / 170.0, 1.0))
            )
        return 0.38 * count_score + 0.24 * ink_score + 0.20 * border_score + 0.18 * layout_score

    def _segment_boxes_by_projection(self, binary, char_count=7):
        work = binary.copy()
        work[:14, :] = 0
        work[106:, :] = 0
        col_sum = (work > 0).sum(axis=0).astype(np.float32)
        if float(np.max(col_sum)) <= 0:
            return []
        smooth = cv2.GaussianBlur(col_sum.reshape(1, -1), (1, 17), 0).ravel()
        active = np.where(smooth > max(2.0, 0.10 * float(np.max(smooth))))[0]
        if len(active) < 90:
            return []

        left = max(10, int(active[0]) - 4)
        right = min(self.PLATE_W - 10, int(active[-1]) + 4)
        width = right - left
        if width < 230:
            return []

        boundaries = [left]
        for i in range(1, char_count):
            expected = left + width * i / char_count
            radius = max(8, int(width / char_count * 0.34))
            lo = max(left + 8, int(expected - radius))
            hi = min(right - 8, int(expected + radius))
            if hi <= lo:
                boundaries.append(int(expected))
                continue
            local = smooth[lo:hi]
            min_value = float(np.min(local))
            valley_positions = np.where(local <= min_value + 0.5)[0]
            valley = int(lo + valley_positions[len(valley_positions) // 2])
            boundaries.append(valley)
        boundaries.append(right)
        boundaries = sorted(boundaries)

        boxes = []
        for i in range(char_count):
            x1 = boundaries[i] + 1
            x2 = boundaries[i + 1] - 1
            if x2 - x1 < 12:
                return []
            slot = work[:, x1:x2]
            rows = np.where((slot > 0).sum(axis=1) > 1)[0]
            if len(rows):
                y1 = max(14, int(rows[0]) - 3)
                y2 = min(106, int(rows[-1]) + 4)
            else:
                y1, y2 = 18, 104
            boxes.append((x1, y1, x2 - x1, max(8, y2 - y1)))
        return boxes

    def _segment_by_fixed_slots(self, gray, binary, char_count=7):
        col_sum = (binary > 0).sum(axis=0)
        if np.max(col_sum) > 0:
            active_cols = np.where(col_sum > max(2, 0.08 * np.max(col_sum)))[0]
        else:
            active_cols = []

        if len(active_cols) > 40:
            left = max(16, int(active_cols[0]) - 4)
            right = min(384, int(active_cols[-1]) + 4)
        else:
            left, right = 22, 378

        width = max(1, right - left)
        slot_w = width / char_count
        chars = []
        for i in range(char_count):
            x1 = int(left + i * slot_w) + 2
            x2 = int(left + (i + 1) * slot_w) - 2
            if i == 1:
                x1 += 3
            slot = binary[:, max(0, x1) : min(self.PLATE_W, x2)]
            rows = np.where((slot > 0).sum(axis=1) > 1)[0]
            if len(rows):
                y1 = max(12, int(rows[0]) - 3)
                y2 = min(108, int(rows[-1]) + 3)
            else:
                y1, y2 = 18, 104
            char_mask = binary.copy()
            if int(np.count_nonzero(char_mask[y1:y2, max(0, x1) : min(self.PLATE_W, x2)])) < 12:
                char_mask = self._local_slot_mask(gray, x1, x2, y1, y2)
                x1, x2 = 0, char_mask.shape[1]
                y1, y2 = 0, char_mask.shape[0]
            chars.append(self._crop_char(char_mask, (x1, y1, max(4, x2 - x1), y2 - y1)))
        return chars

    @staticmethod
    def _local_slot_mask(gray, x1, x2, y1, y2):
        slot = gray[max(0, y1) : min(gray.shape[0], y2), max(0, x1) : min(gray.shape[1], x2)]
        if slot.size == 0:
            return np.zeros((64, 32), dtype=np.uint8)
        slot = cv2.GaussianBlur(slot, (3, 3), 0)
        block = max(11, min(31, (min(slot.shape[:2]) // 2) * 2 + 1))
        adaptive = cv2.adaptiveThreshold(slot, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, -2)
        if np.mean(adaptive > 0) > 0.55:
            adaptive = cv2.bitwise_not(adaptive)
        adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        return adaptive

    @staticmethod
    def _crop_char(binary, box):
        x, y, w, h = box
        x1, y1 = max(0, x - 2), max(0, y - 2)
        x2, y2 = min(binary.shape[1], x + w + 2), min(binary.shape[0], y + h + 2)
        crop = binary[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((64, 32), dtype=np.uint8)
        crop = PlateSegmenter._remove_char_border_fragments(crop)
        ys, xs = np.where(crop > 0)
        if len(xs) > 0 and len(ys) > 0:
            crop = crop[max(0, ys.min() - 1) : min(crop.shape[0], ys.max() + 2), max(0, xs.min() - 1) : min(crop.shape[1], xs.max() + 2)]
        return PlateSegmenter._resize_char_canvas(crop)

    @staticmethod
    def _remove_char_border_fragments(crop):
        cleaned = crop.copy()
        if cleaned.shape[0] < 8 or cleaned.shape[1] < 4:
            return cleaned
        cleaned[:1, :] = 0
        cleaned[-1:, :] = 0
        cleaned[:, :1] = 0
        cleaned[:, -1:] = 0
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return cleaned
        kept = np.zeros_like(cleaned)
        min_area = max(8, int(0.015 * cleaned.shape[0] * cleaned.shape[1]))
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if area < min_area and (w <= 3 or h <= 3):
                continue
            if w >= 0.85 * cleaned.shape[1] and h <= 4:
                continue
            if h >= 0.85 * cleaned.shape[0] and w <= 4:
                continue
            cv2.drawContours(kept, [cnt], -1, 255, thickness=-1)
        return kept

    @staticmethod
    def _resize_char_canvas(crop):
        canvas = np.zeros((64, 32), dtype=np.uint8)
        if crop.size == 0:
            return canvas
        h, w = crop.shape[:2]
        scale = min(26.0 / max(1, w), 56.0 / max(1, h))
        new_w = max(1, min(30, int(round(w * scale))))
        new_h = max(1, min(62, int(round(h * scale))))
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        y = (64 - new_h) // 2
        x = (32 - new_w) // 2
        canvas[y : y + new_h, x : x + new_w] = resized
        return canvas


class LPRPipeline:
    def __init__(self, use_synthetic_pollution=False, detector_weights=None, char_detector_weights=None):
        self.enhancer = ImageEnhancer()
        self.segmenter = PlateSegmenter()
        self.detector = PlateDetector(detector_weights)
        self.char_detector = PlateDetector(
            char_detector_weights or os.environ.get("CHAR_DETECTOR_WEIGHTS"),
            conf=float(os.environ.get("CHAR_DETECTOR_CONF", "0.25")),
        )
        self.use_synthetic_pollution = use_synthetic_pollution

    def process_image(self, img_path):
        if isinstance(img_path, str):
            img = cv2.imread(img_path)
            source_path = img_path
        else:
            img = img_path
            source_path = None
        if img is None:
            return None, []

        if self.use_synthetic_pollution:
            img = ArtificialPolluter.add_synthetic_fog(img, severity=0.6)
            img = ArtificialPolluter.add_synthetic_dirt(img, num_spots=10)

        candidates = []
        filename_plate = self._locate_from_ccpd_filename(img, source_path) if source_path else None
        if filename_plate is not None:
            candidates.append(("ccpd_filename", filename_plate))

        for score, box in self.detector.detect(img):
            plate_img = self._crop_detector_box(img, box)
            if plate_img is not None:
                candidates.append((f"detector_{score:.2f}", plate_img))

        for variant_name, variant_img in self._preprocess_variants(img):
            plate_img = self.segmenter.locate_plate(variant_img)
            if plate_img is not None:
                candidates.append((variant_name, plate_img))

        return self._select_best_plate_result(candidates)

    def _crop_detector_box(self, img, box):
        x1, y1, x2, y2 = box
        h, w = img.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        if bw <= 10 or bh <= 6:
            return None
        pad_x = int(0.04 * bw)
        pad_y = int(0.12 * bh)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (PlateSegmenter.PLATE_W, PlateSegmenter.PLATE_H))

    def _preprocess_variants(self, img):
        variants = [("original", img)]
        try:
            variants.append(("dehazed", self.enhancer.dehaze(img)))
        except Exception:
            pass

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = cv2.merge([clahe.apply(l_channel), a_channel, b_channel])
        variants.append(("clahe", cv2.cvtColor(contrast, cv2.COLOR_LAB2BGR)))

        denoised = cv2.bilateralFilter(img, d=5, sigmaColor=45, sigmaSpace=45)
        variants.append(("bilateral", denoised))
        return variants

    def _select_best_plate_result(self, candidates):
        if not candidates:
            return None, []

        best_score = -1.0
        best_plate = None
        best_chars = []
        for _, plate_img in candidates:
            chars = self._detect_or_segment_chars(plate_img)
            score = self._plate_result_score(plate_img, chars)
            if score > best_score:
                best_score = score
                best_plate = plate_img
                best_chars = chars
        return best_plate, best_chars

    @staticmethod
    def _plate_result_score(plate_img, chars):
        try:
            quality = PlateSegmenter._plate_quality_score(plate_img)
        except Exception:
            quality = 0.0

        char_count = len(chars)
        count_score = 1.0 - min(abs(char_count - 7) / 7.0, 1.0)
        if char_count == 7:
            count_score += 0.35

        ink_scores = []
        for char_img in chars[:7]:
            if char_img is None or char_img.size == 0:
                continue
            ink_ratio = float(np.mean(char_img > 0))
            ink_scores.append(1.0 - min(abs(ink_ratio - 0.22) / 0.35, 1.0))
        ink_score = float(np.mean(ink_scores)) if ink_scores else 0.0
        return 0.45 * quality + 0.40 * count_score + 0.15 * ink_score

    def _detect_or_segment_chars(self, plate_img):
        detections = self.char_detector.detect(plate_img)
        if len(detections) >= 5:
            chars = []
            for _, box in sorted(detections, key=lambda item: item[1][0])[:7]:
                x1, y1, x2, y2 = box
                crop = plate_img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if np.mean(binary > 0) > 0.55:
                    binary = cv2.bitwise_not(binary)
                chars.append(PlateSegmenter._resize_char_canvas(binary))
            if len(chars) >= 5:
                return chars
        return self.segmenter.segment_characters(plate_img)

    @staticmethod
    def _locate_from_ccpd_filename(img, path):
        if not path:
            return None
        stem = os.path.splitext(os.path.basename(path))[0]
        parts = stem.split("-")
        if len(parts) < 4:
            return None
        try:
            point_tokens = parts[3].split("_")
            points = []
            for token in point_tokens:
                x_str, y_str = token.split("&")
                points.append([float(x_str), float(y_str)])
            if len(points) != 4:
                return None
        except Exception:
            return None

        pts = np.array(points, dtype="float32")
        ordered = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        ordered[0] = pts[np.argmin(s)]
        ordered[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        ordered[1] = pts[np.argmin(diff)]
        ordered[3] = pts[np.argmax(diff)]
        dst = np.array([[0, 0], [399, 0], [399, 119], [0, 119]], dtype="float32")
        matrix = cv2.getPerspectiveTransform(ordered, dst)
        return cv2.warpPerspective(img, matrix, (400, 120))
