
import cv2
import numpy as np
import os
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
            x, y = random.randint(0, w), random.randint(0, h)
            r = random.randint(5, max_radius)
            color = (random.randint(20, 50), random.randint(30, 60), random.randint(30, 60))
            cv2.circle(dirty_img, (x, y), r, color, -1)
        return dirty_img

class ImageEnhancer:
    def __init__(self, omega=0.95, t0=0.1, window_size=15):
        self.omega = omega
        self.t0 = t0
        self.window_size = window_size

    def dehaze(self, img):
        img_float = img.astype('float64') / 255.0
        min_channel = np.min(img_float, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.window_size, self.window_size))
        dark = cv2.erode(min_channel, kernel)
        
        A = np.max(img_float) 
        transmission = 1 - self.omega * cv2.erode(np.min(img_float / A, axis=2), kernel)
        transmission = np.maximum(transmission, self.t0)
        
        recovered = np.empty_like(img_float)
        for i in range(3):
            recovered[:, :, i] = (img_float[:, :, i] - A) / transmission + A
            
        return np.clip(recovered * 255, 0, 255).astype('uint8')

class PlateSegmenter:
    def locate_plate(self, img):
        """利用透视变换(Perspective Transform)解决严重倾斜"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 43, 46])
        upper_blue = np.array([124, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        morph = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for cnt in contours:
            if cv2.contourArea(cnt) < 1000: continue
            rect = cv2.minAreaRect(cnt)
            center, size, angle = rect
            w, h = size
            if w == 0 or h == 0: continue
            
            if w < h:
                w, h = h, w
                angle += 90
            
            aspect_ratio = w / h
            if 2.0 <= aspect_ratio <= 5.5:
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                
                # 严格的四点排序，确保不会翻转
                rect_pts = np.zeros((4, 2), dtype="float32")
                s = box.sum(axis=1)
                rect_pts[0] = box[np.argmin(s)]       
                rect_pts[2] = box[np.argmax(s)]       
                diff = np.diff(box, axis=1)
                rect_pts[1] = box[np.argmin(diff)]    
                rect_pts[3] = box[np.argmax(diff)]    
                
                # 强制拉平为 400x120 绝对坐标系
                dst_w, dst_h = 400, 120
                dst_pts = np.array([
                    [0, 0],
                    [dst_w - 1, 0],
                    [dst_w - 1, dst_h - 1],
                    [0, dst_h - 1]
                ], dtype="float32")
                
                M = cv2.getPerspectiveTransform(rect_pts, dst_pts)
                warped_plate = cv2.warpPerspective(img, M, (dst_w, dst_h))
                return warped_plate
                
        h_img, w_img = img.shape[:2]
        return img[int(h_img*0.4):int(h_img*0.95), int(w_img*0.1):int(w_img*0.9)]

    def segment_characters(self, plate_img):
        """🌟 终极物理约束切割算法 (彻底解决多切、重叠)"""
        if plate_img is None: return []
            
        plate_img = cv2.resize(plate_img, (400, 120))
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 1. 暴力切除所有边框、铆钉区域
        binary[0:15, :] = 0    
        binary[105:, :] = 0    
        binary[:, 0:12] = 0    
        binary[:, 388:] = 0    
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.dilate(binary, kernel, iterations=1)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_boxes = []
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # 初步过滤：排除明显不是字符的微小噪点
            if h > 35 and w > 5 and h < 110 and w < 100:
                raw_boxes.append((x, y, w, h))
                
        raw_boxes = sorted(raw_boxes, key=lambda b: b[0])
        
        # 2. 🌟 核心融合算法：物理极限约束融合
        merged_boxes = []
        for box in raw_boxes:
            if not merged_boxes:
                merged_boxes.append(box)
                continue
                
            lx, ly, lw, lh = merged_boxes[-1]
            x, y, w, h = box
            
            gap = x - (lx + lw) # 前一个框右边缘 与 当前框左边缘 的距离
            
            # 如果存在重叠(gap<0)，或者缝隙极小(gap<=12)，说明它们是同一个被污染断裂的字
            if gap <= 12:
                min_x = min(lx, x)
                min_y = min(ly, y)
                max_x = max(lx + lw, x + w)
                max_y = max(ly + lh, y + h)
                merged_w = max_x - min_x
                merged_h = max_y - min_y
                
                # 【物理极限约束】：在 400x120 下，一个单字符宽度绝不可能超过 65。
                # 超过 65 说明算法正在错误地试图把相邻的两个独立字符合并，必须拦截！
                if merged_w <= 65:
                    merged_boxes[-1] = (min_x, min_y, merged_w, merged_h)
                else:
                    merged_boxes.append(box)
            else:
                merged_boxes.append(box)
                
        # 3. 严格二次过滤
        final_boxes = []
        for box in merged_boxes:
            x, y, w, h = box
            # 排除车牌中间的小圆点 (通常 w 和 h 都很小)
            # 真实字符的高度绝对大于 40，宽度绝对大于 8 (数字1)
            if h > 40 and w > 8:
                final_boxes.append(box)
                
        # 4. 🌟 强制 7 字符机制
        # 如果经过上述融合还有超过 7 个块，说明存在极强污渍。
        # 我们只保留面积最大的 7 个块 (字符面积必然大于噪点)
        if len(final_boxes) > 7:
            final_boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
            final_boxes = final_boxes[:7]
            final_boxes.sort(key=lambda b: b[0]) # 再次按 X 从左到右排序
            
        char_images = []
        for box in final_boxes:
            x, y, w, h = box
            x1, y1 = max(0, x-2), max(0, y-2)
            x2, y2 = min(400, x+w+2), min(120, y+h+2)
            char_crop = gray[y1:y2, x1:x2]
            
            _, char_bin = cv2.threshold(char_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 防反转检测
            corners = [char_bin[0,0], char_bin[0,-1], char_bin[-1,0], char_bin[-1,-1]]
            if sum(corners) > 255 * 2: 
                char_bin = cv2.bitwise_not(char_bin)
                
            char_images.append(char_bin)
            
        return char_images

class LPRPipeline:
    def __init__(self, use_synthetic_pollution=False):
        self.enhancer = ImageEnhancer()
        self.segmenter = PlateSegmenter()
        self.use_synthetic_pollution = use_synthetic_pollution

    def process_image(self, img_path):
        if isinstance(img_path, str):
            img = cv2.imread(img_path)
        else:
            img = img_path 
            
        if img is None: return None, []

        if self.use_synthetic_pollution:
            img = ArtificialPolluter.add_synthetic_fog(img, severity=0.6)
            img = ArtificialPolluter.add_synthetic_dirt(img, num_spots=10)

        enhanced_img = self.enhancer.dehaze(img)
        plate_img = self.segmenter.locate_plate(enhanced_img)
        chars = self.segmenter.segment_characters(plate_img)
        
        return plate_img, chars