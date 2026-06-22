from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BaseRepresentation, RepresentationOutput
from .enhancement import ENHANCEMENT_METHODS

ENHANCE_FNS = ENHANCEMENT_METHODS


class EnhancedCropRepresentation(BaseRepresentation):
    """RGB → Enhancement → MediaPipe/YOLO Crop → image for classifier."""
    name = "enhanced_crop"
    output_type = "image"

    def __init__(self, *, enhancement: str = "clahe", cropper: str = "mediapipe",
                 image_size: int = 224, pad: int = 20,
                 model_path: str = "weights/yolo_hand.pt", **kwargs):
        self.enhancement = enhancement
        self.enhance_fn = ENHANCE_FNS[enhancement]

        if cropper == "mediapipe":
            from .mediapipe_crop import MediaPipeCropRepresentation
            self._cropper = MediaPipeCropRepresentation(image_size=image_size, pad=pad)
        elif cropper == "yolo":
            from .yolo_crop import YOLOCropRepresentation
            self._cropper = YOLOCropRepresentation(
                model_path=model_path, image_size=image_size, pad=pad)
        else:
            raise ValueError(f"Unknown cropper: {cropper}")

        self.name = f"enhanced_{enhancement}_{cropper}_crop"

    def process(self, frame_rgb: np.ndarray) -> Optional[RepresentationOutput]:
        enhanced = self.enhance_fn(frame_rgb)
        result = self._cropper.process(enhanced)
        if result is not None:
            result.metadata["enhancement"] = self.enhancement
        return result

    def visualize(self, frame: np.ndarray, output: RepresentationOutput) -> np.ndarray:
        return self._cropper.visualize(frame, output)

    def close(self) -> None:
        self._cropper.close()


class EnhancedLandmarksRepresentation(BaseRepresentation):
    """RGB → Enhancement → Pose Estimator → 42-dim features."""
    name = "enhanced_landmarks"
    output_type = "features"

    def __init__(self, *, enhancement: str = "clahe", pose_backend: str = "mediapipe", **kwargs):
        self.enhancement = enhancement
        self.enhance_fn = ENHANCE_FNS[enhancement]

        if pose_backend == "mediapipe":
            from .mediapipe_landmarks import MediaPipeLandmarksRepresentation
            self._landmarker = MediaPipeLandmarksRepresentation()
        elif pose_backend == "mmpose":
            from .mmpose_landmarks import MMPoseLandmarksRepresentation
            self._landmarker = MMPoseLandmarksRepresentation(**kwargs)
        else:
            raise ValueError(f"Unknown pose_backend: {pose_backend}")

        self.name = f"enhanced_{enhancement}_{pose_backend}_landmarks"

    def process(self, frame_rgb: np.ndarray) -> Optional[RepresentationOutput]:
        enhanced = self.enhance_fn(frame_rgb)
        result = self._landmarker.process(enhanced)
        if result is not None:
            result.metadata["enhancement"] = self.enhancement
        return result

    def visualize(self, frame: np.ndarray, output: RepresentationOutput) -> np.ndarray:
        return self._landmarker.visualize(frame, output)

    def close(self) -> None:
        if hasattr(self._landmarker, "close"):
            self._landmarker.close()
