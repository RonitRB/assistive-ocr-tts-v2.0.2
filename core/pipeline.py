# # core/pipeline.py
# import threading
# import time
# import queue
# import cv2
# import logging
# from typing import Dict, Any
# from .ocr_engine import OCREngine, OCRResult
# from .tts_engine import TTSEngine
# from rapidfuzz import fuzz

# logger = logging.getLogger("pipeline")


# class AssistivePipeline:
#     def __init__(self, config):
#         self.config = config
#         self.cfg = config.data
#         self.ocr = OCREngine(self.cfg["ocr"])
#         self.tts = TTSEngine(self.cfg["tts"])
#         self.frame_q = queue.Queue(maxsize=1) # Reduced queue size for responsiveness
#         self.text_q = queue.Queue()
#         self.history = []
#         self.lock = threading.Lock()
#         self.running = False
#         self.threads = []
#         self.last_text = ""
#         self.last_audio = None

#     def start(self):
#         if self.running:
#             return
#         self.running = True
#         tcap = threading.Thread(target=self._capture_loop, name="capture", daemon=True)
#         tproc = threading.Thread(target=self._process_loop, name="process", daemon=True)
#         ttts = threading.Thread(target=self._tts_loop, name="tts", daemon=True)
#         self.threads = [tcap, tproc, ttts]
#         for t in self.threads:
#             t.start()
#         logger.info("Pipeline started")

#     def stop(self):
#         self.running = False
#         try:
#             self.frame_q.put(None, timeout=1)
#         except queue.Full:
#             pass
#         try:
#             self.text_q.put(None, timeout=1)
#         except queue.Full:
#             pass
#         logger.info("Stopping pipeline - waiting for threads to exit")
#         for t in self.threads:
#             t.join(timeout=2)
#         logger.info("Pipeline stopped")

#     def _get_capture_source(self):
#         cam_cfg = self.cfg["camera"]
#         if cam_cfg.get("source_type") == "gstreamer":
#             return (
#                 "nvarguscamerasrc ! video/x-raw(memory:NVMM), "
#                 f"width={self.cfg['camera'].get('width', 1920)}, "
#                 f"height={self.cfg['camera'].get('height', 1080)}, "
#                 "framerate=30/1 ! nvvidconv ! videoconvert ! appsink"
#             )
#         return int(cam_cfg.get("camera_id", 0))

#     def _capture_loop(self):
#         cap_source = self._get_capture_source()
#         cap = cv2.VideoCapture(cap_source)

#         # resolution
#         if isinstance(cap_source, int) and cap.isOpened():
#             res = self.cfg["camera"].get("resolution", "1080p")
#             if res == "1080p":
#                 cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#                 cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
#             else:
#                 cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#                 cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#         if not cap.isOpened():
#             logger.error("Camera open failed for source: %s", cap_source)
#             self.running = False
#             return

#         while self.running:
#             ret, frame = cap.read()
#             if not ret:
#                 time.sleep(0.01)
#                 continue
#             try:
#                 if self.frame_q.full():
#                     _ = self.frame_q.get_nowait()
#                 self.frame_q.put_nowait(frame)
#             except queue.Full:
#                 pass
#         cap.release()

#     def _process_loop(self):
#         last_valid_text = ""
#         last_time = 0
#         while self.running:
#             try:
#                 frame = self.frame_q.get(timeout=1)
#                 if frame is None:
#                     break

#                 ocr_res: OCRResult = self.ocr.extract_text(frame)
#                 text = (ocr_res.text or "").strip()

#                 # Skip if invalid/short
#                 if not text or len(text) < 4:
#                     continue

#                 # Avoid repeats using fuzzy matching
#                 now = time.time()
#                 if fuzz.ratio(text, last_valid_text) > 85 and (now - last_time < 5):
#                     continue

#                 # Save to history
#                 with self.lock:
#                     ts = time.time()
#                     self.history.append({"ts": ts, "text": text})
#                     if len(self.history) > self.cfg["app"].get("max_history", 50):
#                         self.history.pop(0)

#                 last_valid_text = text
#                 last_time = now
#                 self.last_text = text
#                 self.text_q.put(text)
#             except queue.Empty:
#                 continue
#             except Exception as e:
#                 logger.error(f"Error in processing loop: {e}")
#                 continue

#     def _tts_loop(self):
#         while self.running:
#             try:
#                 text = self.text_q.get(timeout=1)
#                 if text is None:
#                     break
#                 self.tts.speak(
#                     text,
#                     voice=self.cfg["tts"].get("voice"),
#                     speed=self.cfg["tts"].get("speed", 1.0),
#                     volume=self.cfg["tts"].get("volume", 0.9),
#                 )
#             except queue.Empty:
#                 continue
#             except Exception as e:
#                 logger.error(f"Error in TTS loop: {e}")
#                 continue

#     def get_status(self) -> Dict[str, Any]:
#         return {
#             "running": self.running,
#             "last_text": self.last_text,
#             "history_count": len(self.history),
#         }

#     def get_history(self):
#         with self.lock:
#             return list(self.history)[::-1]


# core/pipeline.py
import threading
import time
import queue
import cv2
import logging
from typing import Dict, Any
from .ocr_engine import OCREngine, OCRResult
from .tts_engine import TTSEngine
from rapidfuzz import fuzz

logger = logging.getLogger("pipeline")


class AssistivePipeline:
    def __init__(self, config):
        self.config = config
        self.cfg = config.data
        self.ocr = OCREngine(self.cfg["ocr"])
        self.tts = TTSEngine(self.cfg["tts"])
        self.frame_q = queue.Queue(maxsize=1) # Reduced queue size for responsiveness
        self.text_q = queue.Queue()
        self.history = []
        self.lock = threading.Lock()
        self.running = False
        self.threads = []
        self.last_text = ""
        self.last_audio = None

    def start(self):
        if self.running:
            return
        self.running = True
        tcap = threading.Thread(target=self._capture_loop, name="capture", daemon=True)
        tproc = threading.Thread(target=self._process_loop, name="process", daemon=True)
        ttts = threading.Thread(target=self._tts_loop, name="tts", daemon=True)
        self.threads = [tcap, tproc, ttts]
        for t in self.threads:
            t.start()
        logger.info("Pipeline started")

    def stop(self):
        self.running = False
        try:
            self.frame_q.put(None, timeout=1)
        except queue.Full:
            pass
        try:
            self.text_q.put(None, timeout=1)
        except queue.Full:
            pass
        logger.info("Stopping pipeline - waiting for threads to exit")
        for t in self.threads:
            t.join(timeout=2)
        logger.info("Pipeline stopped")

    def _get_capture_source(self):
        cam_cfg = self.cfg["camera"]
        if cam_cfg.get("source_type") == "gstreamer":
            return (
                "nvarguscamerasrc ! video/x-raw(memory:NVMM), "
                f"width={self.cfg['camera'].get('width', 1920)}, "
                f"height={self.cfg['camera'].get('height', 1080)}, "
                "framerate=30/1 ! nvvidconv ! videoconvert ! appsink"
            )
        return int(cam_cfg.get("camera_id", 0))

    def _capture_loop(self):
        cap_source = self._get_capture_source()
        cap = cv2.VideoCapture(cap_source)

        # resolution
        if isinstance(cap_source, int) and cap.isOpened():
            res = self.cfg["camera"].get("resolution", "1080p")
            if res == "1080p":
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            else:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            logger.error("Camera open failed for source: %s", cap_source)
            self.running = False
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            try:
                if self.frame_q.full():
                    _ = self.frame_q.get_nowait()
                self.frame_q.put_nowait(frame)
            except queue.Full:
                pass
        cap.release()

    def _process_loop(self):
        last_valid_text = ""
        last_time = 0
        while self.running:
            try:
                frame = self.frame_q.get(timeout=1)
                if frame is None:
                    break

                ocr_res: OCRResult = self.ocr.extract_text(frame)
                text = (ocr_res.text or "").strip()

                # Increase minimum length since we expect sentences now
                if not text or len(text) < 8:
                    continue

                # Lower the ratio slightly to catch more duplicates with minor OCR errors
                now = time.time()
                if fuzz.ratio(text, last_valid_text) > 80 and (now - last_time < 5):
                    continue

                # Save to history
                with self.lock:
                    ts = time.time()
                    self.history.append({"ts": ts, "text": text})
                    if len(self.history) > self.cfg["app"].get("max_history", 50):
                        self.history.pop(0)

                last_valid_text = text
                last_time = now
                self.last_text = text
                self.text_q.put(text)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                continue

    def _tts_loop(self):
        while self.running:
            try:
                text = self.text_q.get(timeout=1)
                if text is None:
                    break
                self.tts.speak(
                    text,
                    voice=self.cfg["tts"].get("voice"),
                    speed=self.cfg["tts"].get("speed", 1.0),
                    volume=self.cfg["tts"].get("volume", 0.9),
                )
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in TTS loop: {e}")
                continue

    def get_status(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "last_text": self.last_text,
            "history_count": len(self.history),
        }

    def get_history(self):
        with self.lock:
            return list(self.history)[::-1]