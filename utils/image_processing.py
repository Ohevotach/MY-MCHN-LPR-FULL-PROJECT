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


class PlateSegmenter:
    PLATE_W = 400
    PLATE_H = 120

    def locate_plate(self, img):
        """Locate blue/green license plates and rectify them to 400x120."""
        if img is None:
            return None

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, np.array([95, 45, 35]), np.array([135, 255, 255]))
        green_mask = cv2.inRange(hsv, np.array([35, 35, 35]), np.array([90, 255, 255]))
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
            y_score = 1.0 - min(abs(y_center - 0.68) / 0.68, 0.85)
            warped = self._warp_rect(img, rect)
            if warped is None:
                continue
            quality = self._plate_quality_score(warped)
            if quality < 0.18:
                continue
            size_penalty = max(0.0, area_ratio - 0.035) * 8.0
            score = quality * 4.0 + aspect_score * 1.5 + fill * 0.8 + y_score * 0.8 - size_penalty
            candidates.append((score, rect, (x, y, bw, bh), warped))

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
    def _plate_quality_score(plate_img):
        hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, np.array([95, 45, 35]), np.array([135, 255, 255]))
        green = cv2.inRange(hsv, np.array([35, 35, 35]), np.array([90, 255, 255]))
        color_mask = cv2.bitwise_or(blue, green)
        roi = color_mask[12:108, 12:388]
        color_ratio = float(np.mean(roi > 0))

        white = cv2.inRange(hsv, np.array([0, 0, 115]), np.array([180, 95, 255]))
        white_roi = white[18:104, 18:382]
        white_ratio = float(np.mean(white_roi > 0))

        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 180)
        edge_ratio = float(np.mean(edges[18:104, 18:382] > 0))

        color_score = min(color_ratio / 0.45, 1.0)
        white_score = 1.0 - min(abs(white_ratio - 0.16) / 0.22, 1.0)
        edge_score = min(edge_ratio / 0.12, 1.0)
        return 0.55 * color_score + 0.25 * white_score + 0.20 * edge_score

    def segment_characters(self, plate_img):
        """Segment a rectified plate into 7 character crops."""
        if plate_img is None:
            return []

        plate_img = cv2.resize(plate_img, (self.PLATE_W, self.PLATE_H))
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        binary = self._make_character_mask(plate_img, gray)

        binary[:12, :] = 0
        binary[108:, :] = 0
        binary[:, :10] = 0
        binary[:, 390:] = 0
        binary = cv2.medianBlur(binary, 3)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3)))

        contour_boxes = self._find_contour_boxes(binary)
        if self._has_valid_plate_layout(contour_boxes):
            return [self._crop_char(binary, box) for box in contour_boxes]
        return self._segment_by_fixed_slots(gray, binary, char_count=7)

    @staticmethod
    def _make_character_mask(plate_img, gray):
        hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, np.array([95, 45, 35]), np.array([135, 255, 255]))
        green = cv2.inRange(hsv, np.array([35, 35, 35]), np.array([90, 255, 255]))
        plate_color = cv2.bitwise_or(blue, green)
        color_ratio = float(np.mean(plate_color[12:108, 12:388] > 0))

        if color_ratio > 0.25:
            white = cv2.inRange(hsv, np.array([0, 0, 120]), np.array([180, 105, 255]))
            bright = cv2.inRange(gray, 125, 255)
            binary = cv2.bitwise_and(white, bright)
        else:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            border = np.concatenate([binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]])
            if float(np.mean(border > 0)) > 0.55:
                binary = cv2.bitwise_not(binary)

        return binary

    def _find_contour_boxes(self, binary):
        work = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
        contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = w / max(1, h)
            area = w * h
            if 32 <= h <= 100 and 5 <= w <= 62 and 0.05 <= ratio <= 0.95 and area >= 160:
                boxes.append((x, y, w, h))
        if len(boxes) > 7:
            boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:7]
        return sorted(boxes, key=lambda b: b[0])

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
            chars.append(self._crop_char(binary, (x1, y1, max(4, x2 - x1), y2 - y1)))
        return chars

    @staticmethod
    def _crop_char(binary, box):
        x, y, w, h = box
        x1, y1 = max(0, x - 2), max(0, y - 2)
        x2, y2 = min(binary.shape[1], x + w + 2), min(binary.shape[0], y + h + 2)
        crop = binary[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((64, 32), dtype=np.uint8)
        ys, xs = np.where(crop > 0)
        if len(xs) > 0 and len(ys) > 0:
            crop = crop[max(0, ys.min() - 1) : min(crop.shape[0], ys.max() + 2), max(0, xs.min() - 1) : min(crop.shape[1], xs.max() + 2)]
        return PlateSegmenter._resize_char_canvas(crop)

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
    def __init__(self, use_synthetic_pollution=False):
        self.enhancer = ImageEnhancer()
        self.segmenter = PlateSegmenter()
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

        plate_img = self._locate_from_ccpd_filename(img, source_path) if source_path else None
        if plate_img is None:
            plate_img = self.segmenter.locate_plate(img)
        if plate_img is None:
            plate_img = self.segmenter.locate_plate(self.enhancer.dehaze(img))
        chars = self.segmenter.segment_characters(plate_img)
        return plate_img, chars

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
