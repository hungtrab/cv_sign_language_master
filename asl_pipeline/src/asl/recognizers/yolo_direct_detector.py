from __future__ import annotations

from .base import BaseRecognizer, Prediction
from ..representations.base import RepresentationOutput


class YOLODirectDetector(BaseRecognizer):
    """YOLO direct A-Z detection — single-stage detector trained on ASL letters.

    Output: bbox + A-Z class + confidence.
    Evaluation: mAP@50, mAP@50:95, classification accuracy from top detection.
    """
    name = "yolo_direct_detector"
    input_type = "image"

    def __init__(self, *, model_path: str = "weights/yolo11_asl_letters.pt",
                 num_classes: int = 26, conf: float = 0.5):
        raise NotImplementedError(
            "YOLODirectDetector requires a YOLO model trained on ASL A-Z letter classes. "
            f"Expected weights at: {model_path}. "
            "Dataset: Roboflow American Sign Language Letters Object Detection."
        )

    def predict(self, rep_output: RepresentationOutput) -> Prediction:
        raise NotImplementedError
