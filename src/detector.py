from pathlib import Path
import torch
import os

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = Path(os.getenv("MODEL_PATH", ROOT / "models" / "mask_detector.pt"))


class MaskDetector:
    def __init__(self, weights=DEFAULT_WEIGHTS, conf=0.25, imgsz=640):
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO(str(weights))
        self.conf = conf
        self.imgsz = imgsz
        self.last_speed = {}
        self.warmup()

    def warmup(self):
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.model.predict(dummy, imgsz=self.imgsz, device=self.device, verbose=False)

    def predict(self, image_bgr):
        r = self.model.predict(image_bgr, conf=self.conf, imgsz=self.imgsz,device=self.device, verbose=False)[0]
        self.last_speed = {k: round(v, 1) for k, v in r.speed.items()}
        detections = []
        for box, cls, score in zip(r.boxes.xyxy.cpu().numpy(),
                                   r.boxes.cls.cpu().numpy(),
                                   r.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = (round(float(v), 1) for v in box)
            detections.append({
                "class": self.model.names[int(cls)],
                "confidence": round(float(score), 3),
                "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            })
        return detections

    @property
    def device_info(self):
        predictor = self.model.predictor
        return {
            "inference_device": str(predictor.device) if predictor else "not loaded",
            "cuda_available": torch.cuda.is_available(),
        }