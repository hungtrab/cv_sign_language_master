"""Inference-time wrapper around an Ultralytics YOLO checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class Detection:
    bbox_xyxy: tuple[float, float, float, float]   # absolute pixel coords
    confidence: float
    class_id: int = 0
    class_name: str = "hand"

    def crop(self, image: np.ndarray, *, pad: int = 8) -> np.ndarray:
        """Crop the image to this bbox plus a small ``pad`` margin."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = self.bbox_xyxy
        x1 = int(max(0, x1 - pad))
        y1 = int(max(0, y1 - pad))
        x2 = int(min(w, x2 + pad))
        y2 = int(min(h, y2 + pad))
        if x2 <= x1 or y2 <= y1:
            return np.zeros((1, 1, 3), dtype=image.dtype)
        return image[y1:y2, x1:x2]


class HandDetector:
    """Lazy-loaded YOLOv8 detector. Returns the highest-confidence box."""

    def __init__(self, weights: str | Path, *, conf_threshold: float = 0.5, device: str | int = "cpu"):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.class_names = dict(self.model.names)
        if self.class_names != {0: "hand"}:
            raise ValueError(
                "Expected a single-class hand detector checkpoint, "
                f"but {weights} contains classes: {self.class_names}"
            )
        self.conf_threshold = float(conf_threshold)
        self.device = device

    def detect(self, image: np.ndarray, *, top_k: int = 1) -> List[Detection]:
        """Run YOLO on a single BGR image and return up to ``top_k`` detections."""
        results = self.model.predict(
            source=image[..., ::-1],  # YOLO expects RGB
            verbose=False,
            conf=self.conf_threshold,
            device=self.device,
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            return []

        boxes = results.boxes.xyxy.detach().cpu().numpy()
        confs = results.boxes.conf.detach().cpu().numpy()
        cls = results.boxes.cls.detach().cpu().numpy().astype(int)
        order = np.argsort(-confs)[:top_k]

        out: List[Detection] = []
        for i in order:
            x1, y1, x2, y2 = (float(v) for v in boxes[i])
            out.append(Detection(
                bbox_xyxy=(x1, y1, x2, y2),
                confidence=float(confs[i]),
                class_id=int(cls[i]),
                class_name=self.class_names[int(cls[i])],
            ))
        return out
