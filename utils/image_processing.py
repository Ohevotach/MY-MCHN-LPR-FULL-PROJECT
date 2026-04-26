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
    def locate_plate(self, img):
        """Locate blue/green license plates and rectify them to 400x120."""
        if img is None:
            return None

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, np.array([95, 45, 35]), np.array([135, 255, 255]))
        green_mask = cv2.inRange(hsv, np.array([35, 35, 35]), np.array([90, 255, 255]))
        mask = cv2.bitwise_or(blue_mask, green_mask)

        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

        h_img, w_img = img.shape[:2]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < max(350, 0.0005 * h_img * w_img):
                continue

            rect = cv2.minAreaRect(cnt)
            _, size, _ = rect
            rw, rh = size
            if rw <= 1 or rh <= 1:
                continue
            if rw < rh:
                rw, rh = rh, rw
            aspect = rw / rh
            if not (2.0 <= aspect <= 6.3):
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            fill = area / max(1.0, bw * bh)
            if fill < 0.22:
                continue

            aspect_score = 1.0 - min(abs(aspect - 3.4) / 3.4, 1.0)
            y_center = (y + bh / 2) / max(1, h_img)
            y_score = 1.0 - min(abs(y_center - 0.62) / 0.62, 0.7)
            score = area * (0.4 + aspect_score) * (0.5 + fill) * (0.8 + y_score)
            candidates.append((score, rect, (x, y, bw, bh)))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        _, rect, bbox = candidates[0]
        warped = self._warp_rect(img, rect)
        if warped is not None:
            return warped

        x, y, w, h = bbox
        pad_x, pad_y = int(0.08 * w), int(0.22 * h)
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(w_img, x + w + pad_x), min(h_img, y + h + pad_y)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (400, 120))

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

    def segment_characters(self, plate_img):
        """Segment a rectified plate into 7 character crops."""
        if plate_img is None:
            return []

        plate_img = cv2.resize(plate_img, (400, 120))
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if self._white_border_ratio(binary) > 0.55:
            binary = cv2.bitwise_not(binary)

        binary[:12, :] = 0
        binary[108:, :] = 0
        binary[:, :10] = 0
        binary[:, 390:] = 0
        binary = cv2.medianBlur(binary, 3)

        contour_chars = self._segment_by_contours(gray, binary)
        if len(contour_chars) == 7:
            return contour_chars
        return self._segment_by_fixed_slots(gray, binary, char_count=7)

    @staticmethod
    def _white_border_ratio(binary):
        border = np.concatenate([binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]])
        return float(np.mean(border > 0))

    def _segment_by_contours(self, gray, binary):
        work = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
        contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = w / max(1, h)
            if 35 <= h <= 105 and 6 <= w <= 70 and 0.06 <= ratio <= 1.1:
                boxes.append((x, y, w, h))
        if len(boxes) > 7:
            boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:7]
        boxes = sorted(boxes, key=lambda b: b[0])
        return [self._crop_char(gray, box) for box in boxes]

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
            slot = binary[:, max(0, x1) : min(400, x2)]
            rows = np.where((slot > 0).sum(axis=1) > 1)[0]
            if len(rows):
                y1 = max(12, int(rows[0]) - 3)
                y2 = min(108, int(rows[-1]) + 3)
            else:
                y1, y2 = 18, 104
            chars.append(self._crop_char(gray, (x1, y1, max(4, x2 - x1), y2 - y1)))
        return chars

    @staticmethod
    def _crop_char(gray, box):
        x, y, w, h = box
        x1, y1 = max(0, x - 2), max(0, y - 2)
        x2, y2 = min(gray.shape[1], x + w + 2), min(gray.shape[0], y + h + 2)
        crop = gray[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((64, 32), dtype=np.uint8)
        _, char_bin = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        corners = [char_bin[0, 0], char_bin[0, -1], char_bin[-1, 0], char_bin[-1, -1]]
        if sum(v > 0 for v in corners) >= 3:
            char_bin = cv2.bitwise_not(char_bin)
        return char_bin


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
