# core/config.py
import json
import os
from typing import Dict

DEFAULT_CONFIG = {
    "camera": {
        "source_type": "gstreamer",   # options: opencv, gstreamer, arducam
        "camera_id": 0,
        "resolution": "1080p"      # 720p or 1080p
    },
    "ocr": {
        "engine": "paddle",        # paddle or tesseract
        "language": "eng",         # eng / hin / kan
        "capture_interval": 1,     # seconds between frames
        "use_yolo": True,
        "onnx_sr_model": None      # path to a super-resolution onnx model (optional)
    },
    "tts": {
        "engine": "coqui",         # coqui or espeak
        "coqui_model": "tts_models/en/vctk/vits",
        "voice": "p335",        # speaker / voice for coqui
        "speed": 1.0,
        "volume": 0.9
    },
    "app": {
        "high_contrast": False,
        "font_size": "16px",
        "max_history": 50
    }
}

class Config:
    def __init__(self, filepath: str = "config.json"):
        self.filepath = filepath
        if not os.path.exists(self.filepath):
            self.data = DEFAULT_CONFIG.copy()
            self.save()
        else:
            self.load()

    def load(self):
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except Exception:
            self.data = DEFAULT_CONFIG.copy()
            self.save()

    def save(self):
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def update(self, patch: Dict):
        # shallow update for simplicity; UI will send structured JSON
        self.data.update(patch)
        self.save()