from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BaseRepresentation, RepresentationOutput


class YOLOHandPoseRepresentation(BaseRepresentation):
    """YOLO hand-pose → 21 keypoints → normalized landmark vector.

    Requires a YOLO model trained on hand keypoints (not body keypoints).
    """
    name = "yolo_hand_pose"
    output_type = "features"

    def __init__(self, *, model_path: str = "weights/yolo11_hand_pose.pt"):
        raise NotImplementedError(
            "YOLOHandPoseRepresentation requires a hand-keypoint-trained YOLO model. "
            f"Expected weights at: {model_path}. "
            "Standard body-pose YOLO models output body joints, not finger-level keypoints."
        )

    def process(self, frame_rgb: np.ndarray) -> Optional[RepresentationOutput]:
        raise NotImplementedError
