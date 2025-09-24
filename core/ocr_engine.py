# # core/ocr_engine.py
# import cv2
# import numpy as np
# import logging
# from typing import List, Tuple
# from rapidfuzz import process as rf_process, fuzz
# from symspellpy import SymSpell, Verbosity
# import os
# import re

# logger = logging.getLogger("ocr_engine")

# # PaddleOCR import attempt
# try:
#     from paddleocr import PaddleOCR
#     PADDLE_AVAILABLE = True
# except Exception:
#     PADDLE_AVAILABLE = False

# # YOLO import attempt (ultralytics)
# try:
#     from ultralytics import YOLO
#     YOLO_AVAILABLE = True
# except Exception:
#     YOLO_AVAILABLE = False

# # pytesseract fallback
# try:
#     import pytesseract
#     TESSER_AVAILABLE = True
# except Exception:
#     TESSER_AVAILABLE = False

# # Small utilities
# LANG_MAP = {
#     "eng": "en",
#     "hin": "hi",
#     "kan": "kn"
# }


# class OCRResult:
#     def __init__(self, text: str, boxes: List[Tuple[int, int, int, int]] = None):
#         self.text = text
#         self.boxes = boxes or []


# class OCREngine:
#     def __init__(self, cfg: dict):
#         self.cfg = cfg
#         self.engine = cfg.get("engine", "paddle")
#         self.lang = cfg.get("language", "eng")
#         self.paddle = None
#         if PADDLE_AVAILABLE and self.engine == "paddle":
#             try:
#                 lang = LANG_MAP.get(self.lang, "en")
#                 self.paddle = PaddleOCR(use_angle_cls=True, lang=lang)
#                 logger.info("PaddleOCR initialized with lang=%s", lang)
#             except Exception as e:
#                 logger.warning("Failed to init PaddleOCR: %s", e)
#                 self.paddle = None
#         # YOLO detector (for board/text region detection)
#         self.yolo = None
#         if YOLO_AVAILABLE and cfg.get("use_yolo", True):
#             try:
#                 self.yolo = YOLO("yolov8n.pt")
#                 logger.info("YOLO initialized for region detection")
#             except Exception as e:
#                 logger.warning("YOLO init failed: %s", e)
#                 self.yolo = None

#         # symspell for suggestion/cleanup (optional)
#         try:
#             self.symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
#         except Exception:
#             self.symspell = None

#     @staticmethod
#     def enhance_image(img: np.ndarray) -> np.ndarray:
#         """Enhance image to improve OCR robustness: CLAHE, denoise, unsharp, multi-scale."""
#         if img is None:
#             return img
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        
#         # Deblurring
#         blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
#         unsharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         enhanced = clahe.apply(unsharp)
        
#         enhanced = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

#         h, w = enhanced.shape[:2]
#         if max(h, w) < 1200:
#             enhanced = cv2.resize(enhanced, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        
#         return enhanced

#     def detect_board_box(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
#         """Try YOLO detection for board-like region or fallback to contour-based detection."""
#         h, w = frame.shape[:2]
#         if self.yolo is not None:
#             try:
#                 results = self.yolo(frame, imgsz=640, conf=0.4) # Increased confidence
#                 best, best_area = None, 0
#                 for r in results:
#                     for box in r.boxes:
#                         x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#                         area = (x2 - x1) * (y2 - y1)
#                         if area > best_area:
#                             best_area = area
#                             best = (x1, y1, x2, y2)
#                 if best and best_area > 0.03 * w * h: # Stricter area threshold
#                     x1, y1, x2, y2 = best
#                     return frame[y1:y2, x1:x2], (x1, y1, x2, y2)
#             except Exception as e:
#                 logger.debug("YOLO detection error: %s", e)
#         # fallback
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if contours:
#             largest = max(contours, key=cv2.contourArea)
#             x, y, wc, hc = cv2.boundingRect(largest)
#             if wc * hc > 0.05 * frame.shape[0] * frame.shape[1]: # Stricter area threshold
#                 return frame[y:y + hc, x:x + wc], (x, y, x + wc, y + hc)
#         return frame, (0, 0, w, h)

#     def ocr_paddle(self, img: np.ndarray) -> OCRResult:
#         result_texts = []
#         boxes = []
#         try:
#             res = self.paddle.ocr(img, cls=True)
#             for line in res[0]:
#                 box = line[0]
#                 txt, conf = line[1][0], float(line[1][1])
#                 if txt.strip() and conf >= 0.85 and len(txt.strip()) > 2: # Stricter confidence and length
#                     result_texts.append(txt.strip())
#                     x_coords = [int(pt[0]) for pt in box]
#                     y_coords = [int(pt[1]) for pt in box]
#                     boxes.append((min(x_coords), min(y_coords), max(x_coords), max(y_coords)))
#         except Exception as e:
#             logger.debug("PaddleOCR error: %s", e)
#         return OCRResult("\n".join(result_texts), boxes)

#     def ocr_tesseract(self, img: np.ndarray) -> OCRResult:
#         try:
#             config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
#             lang = self.lang if self.lang in ["eng", "hin", "kan"] else "eng"
#             text = pytesseract.image_to_string(img, lang=lang, config=config)
#             return OCRResult(text.strip(), [])
#         except Exception as e:
#             logger.debug("Tesseract error: %s", e)
#             return OCRResult("", [])

#     def cleanup_text(self, text: str) -> str:
#         t = " ".join(text.split())
#         t = re.sub(r'[^A-Za-z0-9\s.,?!]', '', t) # Remove special characters
#         if self.symspell and t and len(t.split()) > 1:
#             suggestions = self.symspell.lookup_compound(t, max_edit_distance=2)
#             if suggestions:
#                 t = suggestions[0].term
#         return t

#     def extract_text(self, frame: np.ndarray) -> OCRResult:
#         crop, box = self.detect_board_box(frame)
#         enhanced = self.enhance_image(crop)
#         final_text, boxes = "", []
#         if self.paddle is not None and self.engine == "paddle":
#             res = self.ocr_paddle(enhanced)
#             final_text = res.text
#             boxes = res.boxes
#         if (not final_text.strip()) and TESSER_AVAILABLE:
#             res = self.ocr_tesseract(enhanced)
#             final_text = res.text
#         final_text = self.cleanup_text(final_text)
#         return OCRResult(final_text, boxes)

# --------------------------------------------------------------------------
# Improved version 0.1

# # core/ocr_engine.py
# import cv2
# import numpy as np
# import logging
# from typing import List, Tuple
# from rapidfuzz import process as rf_process, fuzz
# from symspellpy import SymSpell, Verbosity
# import os
# import re

# logger = logging.getLogger("ocr_engine")

# # PaddleOCR import attempt
# try:
#     from paddleocr import PaddleOCR
#     PADDLE_AVAILABLE = True
# except Exception:
#     PADDLE_AVAILABLE = False

# # YOLO import attempt (ultralytics)
# try:
#     from ultralytics import YOLO
#     YOLO_AVAILABLE = True
# except Exception:
#     YOLO_AVAILABLE = False

# # pytesseract fallback
# try:
#     import pytesseract
#     TESSER_AVAILABLE = True
# except Exception:
#     TESSER_AVAILABLE = False

# # Small utilities
# LANG_MAP = {
#     "eng": "en",
#     "hin": "hi",
#     "kan": "kn"
# }


# class OCRResult:
#     def __init__(self, text: str, boxes: List[Tuple[int, int, int, int]] = None):
#         self.text = text
#         self.boxes = boxes or []


# class OCREngine:
#     def __init__(self, cfg: dict):
#         self.cfg = cfg
#         self.engine = cfg.get("engine", "paddle")
#         self.lang = cfg.get("language", "eng")
#         self.paddle = None
#         if PADDLE_AVAILABLE and self.engine == "paddle":
#             try:
#                 lang = LANG_MAP.get(self.lang, "en")
#                 self.paddle = PaddleOCR(use_angle_cls=True, lang=lang)
#                 logger.info("PaddleOCR initialized with lang=%s", lang)
#             except Exception as e:
#                 logger.warning("Failed to init PaddleOCR: %s", e)
#                 self.paddle = None
#         # YOLO detector (for board/text region detection)
#         self.yolo = None
#         if YOLO_AVAILABLE and cfg.get("use_yolo", True):
#             try:
#                 self.yolo = YOLO("yolov8n.pt")
#                 logger.info("YOLO initialized for region detection")
#             except Exception as e:
#                 logger.warning("YOLO init failed: %s", e)
#                 self.yolo = None

#         # symspell for suggestion/cleanup (optional)
#         try:
#             self.symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
#         except Exception:
#             self.symspell = None

#     @staticmethod
#     def enhance_image(img: np.ndarray) -> np.ndarray:
#         """Enhance image to improve OCR robustness: CLAHE, denoise, unsharp, multi-scale."""
#         if img is None:
#             return img
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        
#         # Deblurring
#         blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
#         unsharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         enhanced = clahe.apply(unsharp)
        
#         enhanced = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

#         h, w = enhanced.shape[:2]
#         if max(h, w) < 1200:
#             enhanced = cv2.resize(enhanced, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        
#         return enhanced

#     def detect_board_box(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
#         """Try YOLO detection for board-like region or fallback to contour-based detection."""
#         h, w = frame.shape[:2]
#         if self.yolo is not None:
#             try:
#                 results = self.yolo(frame, imgsz=640, conf=0.4) # Increased confidence
#                 best, best_area = None, 0
#                 for r in results:
#                     for box in r.boxes:
#                         x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#                         area = (x2 - x1) * (y2 - y1)
#                         if area > best_area:
#                             best_area = area
#                             best = (x1, y1, x2, y2)
#                 if best and best_area > 0.03 * w * h: # Stricter area threshold
#                     x1, y1, x2, y2 = best
#                     return frame[y1:y2, x1:x2], (x1, y1, x2, y2)
#             except Exception as e:
#                 logger.debug("YOLO detection error: %s", e)
#         # fallback
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if contours:
#             largest = max(contours, key=cv2.contourArea)
#             x, y, wc, hc = cv2.boundingRect(largest)
#             if wc * hc > 0.05 * frame.shape[0] * frame.shape[1]: # Stricter area threshold
#                 return frame[y:y + hc, x:x + wc], (x, y, x + wc, y + hc)
#         return frame, (0, 0, w, h)
        
#     def _consolidate_results(self, ocr_output: list) -> OCRResult:
#         if not ocr_output or not ocr_output[0]:
#             return OCRResult("", [])

#         lines = []
#         for line in ocr_output[0]:
#             box_coords = line[0]
#             text, confidence = line[1]
            
#             # More stringent filtering for individual words
#             if confidence < 0.90 or len(text) <= 1:
#                 continue

#             x_coords = [p[0] for p in box_coords]
#             y_coords = [p[1] for p in box_coords]
#             box = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
#             center_y = (box[1] + box[3]) / 2
            
#             # Find a line to append this word to
#             found_line = False
#             for line_group in lines:
#                 avg_y = line_group['avg_y']
#                 box_height = box[3] - box[1]
#                 # If vertically aligned and close enough, add to the line
#                 if abs(center_y - avg_y) < box_height * 0.7:
#                     line_group['words'].append({'text': text, 'box': box})
#                     # Update average y-position of the line
#                     line_group['avg_y'] = (avg_y * (len(line_group['words']) - 1) + center_y) / len(line_group['words'])
#                     found_line = True
#                     break
            
#             if not found_line:
#                 lines.append({'words': [{'text': text, 'box': box}], 'avg_y': center_y})

#         # Sort words within each line by their x-position
#         for line_group in lines:
#             line_group['words'].sort(key=lambda w: w['box'][0])
            
#         # Join words to form final text lines and get overall boxes
#         final_texts = [" ".join([w['text'] for w in line_group['words']]) for line_group in lines]
#         final_boxes = []
#         for line_group in lines:
#             x1 = min(w['box'][0] for w in line_group['words'])
#             y1 = min(w['box'][1] for w in line_group['words'])
#             x2 = max(w['box'][2] for w in line_group['words'])
#             y2 = max(w['box'][3] for w in line_group['words'])
#             final_boxes.append((x1, y1, x2, y2))
            
#         return OCRResult("\n".join(final_texts), final_boxes)


#     def ocr_paddle(self, img: np.ndarray) -> OCRResult:
#         try:
#             res = self.paddle.ocr(img, cls=True)
#             return self._consolidate_results(res)
#         except Exception as e:
#             logger.debug("PaddleOCR error: %s", e)
#             return OCRResult("", [])

#     def ocr_tesseract(self, img: np.ndarray) -> OCRResult:
#         try:
#             config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
#             lang = self.lang if self.lang in ["eng", "hin", "kan"] else "eng"
#             text = pytesseract.image_to_string(img, lang=lang, config=config)
#             return OCRResult(text.strip(), [])
#         except Exception as e:
#             logger.debug("Tesseract error: %s", e)
#             return OCRResult("", [])

#     def cleanup_text(self, text: str) -> str:
#         t = " ".join(text.split())
#         t = re.sub(r'[^A-Za-z0-9\s.,?!()\-_=+*/<>\[\]{}]', '', t)
        
#         if self.symspell and t and len(t.split()) > 1:
#             suggestions = self.symspell.lookup_compound(t, max_edit_distance=2)
#             if suggestions:
#                 t = suggestions[0].term
#         return t

#     def extract_text(self, frame: np.ndarray) -> OCRResult:
#         crop, box = self.detect_board_box(frame)
#         enhanced = self.enhance_image(crop)
        
#         res = OCRResult("", [])
#         if self.paddle is not None and self.engine == "paddle":
#             res = self.ocr_paddle(enhanced)
            
#         if not res.text.strip() and TESSER_AVAILABLE:
#             res = self.ocr_tesseract(enhanced)
        
#         res.text = self.cleanup_text(res.text)
#         return res


# ----------------------------------------------------------------------------------------------
# Ip ver 0.2
# Perfectly Working the text

# core/ocr_engine.py
import cv2
import numpy as np
import logging
import re
from typing import List, Tuple
from symspellpy import SymSpell

logger = logging.getLogger("ocr_engine")

# --- Import third-party libraries safely ---
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
try:
    import pytesseract
    TESSER_AVAILABLE = True
except ImportError:
    TESSER_AVAILABLE = False

LANG_MAP = {"eng": "en", "hin": "hi", "kan": "kn"}

class OCRResult:
    def __init__(self, text: str, boxes: List[Tuple[int, int, int, int]] = None):
        self.text = text
        self.boxes = boxes or []

class OCREngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.engine = cfg.get("engine", "paddle")
        self.lang = cfg.get("language", "eng")
        self.paddle = None
        if PADDLE_AVAILABLE and self.engine == "paddle":
            try:
                lang = LANG_MAP.get(self.lang, "en")
                self.paddle = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
                logger.info("PaddleOCR initialized with lang=%s", lang)
            except Exception as e:
                logger.warning("Failed to init PaddleOCR: %s", e)
                self.paddle = None
        
        self.yolo = None
        if YOLO_AVAILABLE and cfg.get("use_yolo", False): # Defaulting YOLO to False as it's unreliable
            try:
                self.yolo = YOLO("yolov8n.pt")
                logger.info("YOLO initialized for region detection")
            except Exception as e:
                logger.warning("YOLO init failed: %s", e)
                self.yolo = None
        
        self.symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

    @staticmethod
    def enhance_image(img: np.ndarray) -> np.ndarray:
        """
        NEW: Enhance image using adaptive thresholding for cleaner text isolation.
        This is much more effective at removing noise than the previous method.
        """
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply a Gaussian blur to reduce noise before thresholding
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use adaptive thresholding to handle different lighting conditions
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 4)
        
        # Use a morphological operation to remove small noise specks
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Invert back so text is black on a white background (if needed by OCR)
        # PaddleOCR handles both, but this is a clean standard form.
        return 255 - opening

    def detect_board_box(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        NEW: More robust board detection. It now specifically looks for large 
        quadrilaterals and returns None if no plausible board is found.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_candidate = None
        max_area = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter out very small contours and contours that are too large (likely the whole frame)
            if area < (w * h * 0.05) or area > (w * h * 0.9):
                continue
            
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            # Check if the contour is a quadrilateral and has a large area
            if len(approx) == 4 and area > max_area:
                x, y, wc, hc = cv2.boundingRect(approx)
                aspect_ratio = float(wc) / hc
                # Check for a plausible aspect ratio (e.g., not a very thin line)
                if 0.5 < aspect_ratio < 5.0:
                    max_area = area
                    best_candidate = (x, y, wc, hc)

        if best_candidate:
            x, y, wc, hc = best_candidate
            # Add some padding around the detected box
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + wc + padding)
            y2 = min(h, y + hc + padding)
            return frame[y1:y2, x1:x2], (x1, y1, x2, y2)
            
        # If no good contour is found, return None. This is crucial.
        return None, None

    def _is_plausible_text(self, text: str) -> bool:
        """
        NEW: A validation step to check if the OCR output is likely real text or just noise.
        """
        if not text:
            return False

        words = text.split()
        if not words:
            return False

        # 1. Check character type ratio (at least 70% should be letters)
        alpha_chars = sum(c.isalpha() for c in text)
        total_chars = len(text)
        if (alpha_chars / total_chars) < 0.7:
            logger.debug(f"Plausibility Fail (Alpha Ratio): {text}")
            return False

        # 2. Check average word length (avoid lots of single-letter noise)
        avg_word_len = sum(len(word) for word in words) / len(words)
        if avg_word_len < 2.5:
            logger.debug(f"Plausibility Fail (Avg Word Length): {text}")
            return False
            
        return True

    def _consolidate_results(self, ocr_output: list) -> OCRResult:
        if not ocr_output or not ocr_output[0]:
            return OCRResult("", [])

        lines = []
        for line in ocr_output[0]:
            text, confidence = line[1]
            if confidence > 0.85: # Keep confidence check reasonable
                lines.append(text)
        
        full_text = " ".join(lines).strip()
        
        # NEW: Validate the final text block for plausibility
        if not self._is_plausible_text(full_text):
            return OCRResult("", [])
            
        return OCRResult(full_text, [])

    def ocr_paddle(self, img: np.ndarray) -> OCRResult:
        try:
            res = self.paddle.ocr(img, cls=True)
            return self._consolidate_results(res)
        except Exception as e:
            logger.debug("PaddleOCR error: %s", e)
            return OCRResult("", [])

    def cleanup_text(self, text: str) -> str:
        t = " ".join(text.split())
        t = re.sub(r'[^A-Za-z0-9\s.,?!()\-_=+*/<>\[\]{}]', '', t)
        
        if self.symspell and t and len(t.split()) > 1:
            suggestions = self.symspell.lookup_compound(t, max_edit_distance=2)
            if suggestions:
                t = suggestions[0].term
        return t

    def extract_text(self, frame: np.ndarray) -> OCRResult:
        # MODIFIED: The main control flow is now safer
        
        # 1. Try to detect a board.
        crop, box = self.detect_board_box(frame)
        
        # 2. If no board is found, stop immediately. Prevents OCR on noisy backgrounds.
        if crop is None:
            return OCRResult("", [])
        
        # 3. Enhance ONLY the cropped region.
        enhanced = self.enhance_image(crop)
        
        res = OCRResult("", [])
        if self.paddle:
            res = self.ocr_paddle(enhanced)
            
        if not res.text.strip() and TESSER_AVAILABLE:
            # Tesseract fallback (less common now with plausibility checks)
            text = pytesseract.image_to_string(enhanced)
            if self._is_plausible_text(text):
                res.text = text
        
        res.text = self.cleanup_text(res.text)
        return res



