# Re-using your provided file logic but ensuring imports work in the module structure
from ultralytics import YOLO
import numpy as np
import time
import os

# Ensure model path is correct relative to where we run the app
MODEL_PATH = "yolo11s.pt"

class YOLOAnalyzer:
    def __init__(self):
        try:
            self.model = YOLO(MODEL_PATH)
        except Exception as e:
            print(f"[WARN] YOLO model not found at {MODEL_PATH}. Download yolo11s.pt. Error: {e}")
            self.model = None

        self.PHONE_CLASS_NAMES = {"mobile phone", "cell phone", "phone"}
        self.PERSON_CLASS_NAMES = {"person"}

    def detect(self, frame):
        if self.model is None:
            return {"phone_detected": False, "phone_boxes": [], "person_count": 0, "inference_ms": 0}

        t0 = time.time()
        try:
            # Run inference
            results = self.model(frame, conf=0.35, verbose=False)[0]
        except Exception:
            return {"phone_detected": False, "phone_boxes": [], "person_count": 0, "inference_ms": 0}

        phone_boxes = []
        person_count = 0

        if results.boxes:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id].lower()
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()

                if label in self.PHONE_CLASS_NAMES:
                    phone_boxes.append({"label": label, "conf": conf, "xyxy": xyxy})
                elif label in self.PERSON_CLASS_NAMES:
                    person_count += 1

        inference_ms = (time.time() - t0) * 1000.0
        
        return {
            "phone_detected": len(phone_boxes) > 0,
            "phone_boxes": phone_boxes,
            "person_count": person_count,
            "inference_ms": inference_ms
        }