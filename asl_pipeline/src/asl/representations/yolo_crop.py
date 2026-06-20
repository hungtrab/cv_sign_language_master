from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BaseRepresentation, RepresentationOutput


class YOLOCropRepresentation(BaseRepresentation):
    """YOLO hand detector → crop. Requires a hand-trained YOLO .pt model."""
    name = "yolo_crop"
    output_type = "image"

    def __init__(self, *, model_path: str = "weights/yolo11_hand_detector.pt",
                 image_size: int = 224, pad: int = 32, conf: float = 0.5):
        raise NotImplementedError(
            "YOLOCropRepresentation requires a hand-trained YOLO model. "
            f"Expected weights at: {model_path}. "
            "Train or download a YOLO hand detector first."
        )

    def process(self, frame_rgb: np.ndarray) -> Optional[RepresentationOutput]:
        raise NotImplementedError
