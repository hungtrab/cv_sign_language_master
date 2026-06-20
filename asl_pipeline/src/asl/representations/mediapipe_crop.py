from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .base import BaseRepresentation, RepresentationOutput
from ..utils.visualization import draw_bbox

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_CACHE = Path("weights/hand_landmarker.task")


def _ensure_model() -> str:
    if MODEL_CACHE.exists():
        return str(MODEL_CACHE)
    MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading HandLandmarker model -> {MODEL_CACHE}")
    urllib.request.urlretrieve(MODEL_URL, str(MODEL_CACHE))
    return str(MODEL_CACHE)


class MediaPipeCropRepresentation(BaseRepresentation):
    name = "mediapipe_crop"
    output_type = "image"

    def __init__(self, *, image_size: int = 224, pad: int = 20,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5):
        self.image_size = image_size
        self.pad = pad

        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker, HandLandmarkerOptions, RunningMode,
        )

        model_path = _ensure_model()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._mp = mp

    def process(self, frame_rgb: np.ndarray) -> Optional[RepresentationOutput]:
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return None

        landmarks = result.hand_landmarks[0]
        h, w = frame_rgb.shape[:2]

        points = []
        for lm in landmarks:
            px = min(int(lm.x * w), w - 1)
            py = min(int(lm.y * h), h - 1)
            points.append([px, py])

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x1 = max(0, min(xs) - self.pad)
        y1 = max(0, min(ys) - self.pad)
        x2 = min(w, max(xs) + self.pad)
        y2 = min(h, max(ys) + self.pad)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame_rgb[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop, (self.image_size, self.image_size))

        return RepresentationOutput(
            data=crop_resized,
            output_type="image",
            bbox=(x1, y1, x2, y2),
            landmarks=points,
            confidence=1.0,
        )

    def visualize(self, frame: np.ndarray, output: RepresentationOutput) -> np.ndarray:
        if output.bbox:
            draw_bbox(frame, output.bbox)
        if output.landmarks:
            for x, y in output.landmarks:
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        return frame

    def close(self) -> None:
        self._landmarker.close()
