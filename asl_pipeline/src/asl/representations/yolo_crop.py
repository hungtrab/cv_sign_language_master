from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .base import BaseRepresentation, RepresentationOutput
from ..utils.visualization import draw_bbox


class YOLOCropRepresentation(BaseRepresentation):
    """YOLO hand detector → crop → resized image for classifier."""
    name = "yolo_crop"
    output_type = "image"

    def __init__(self, *, model_path: str = "weights/yolo_hand.pt",
                 image_size: int = 224, pad: int = 20, conf: float = 0.5,
                 device: str = ""):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.image_size = image_size
        self.pad = pad
        self.conf = conf
        self.device = device

    def process(self, frame_rgb: np.ndarray) -> Optional[RepresentationOutput]:
        results = self.model.predict(
            source=frame_rgb,
            verbose=False,
            conf=self.conf,
            device=self.device or None,
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            return None

        boxes = results.boxes.xyxy.detach().cpu().numpy()
        confs = results.boxes.conf.detach().cpu().numpy()
        best = int(confs.argmax())
        x1, y1, x2, y2 = (float(v) for v in boxes[best])
        det_conf = float(confs[best])

        h, w = frame_rgb.shape[:2]
        x1 = int(max(0, x1 - self.pad))
        y1 = int(max(0, y1 - self.pad))
        x2 = int(min(w, x2 + self.pad))
        y2 = int(min(h, y2 + self.pad))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame_rgb[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop, (self.image_size, self.image_size))

        return RepresentationOutput(
            data=crop_resized,
            output_type="image",
            bbox=(x1, y1, x2, y2),
            confidence=det_conf,
        )

    def visualize(self, frame: np.ndarray, output: RepresentationOutput) -> np.ndarray:
        if output.bbox:
            draw_bbox(frame, output.bbox, color=(255, 165, 0))
        return frame
