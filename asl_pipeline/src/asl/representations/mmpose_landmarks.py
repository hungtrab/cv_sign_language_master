"""MMPose RTMPose-Hand → 21 keypoints → normalized 42-dim feature vector.

Uses RTMPose-M trained on Hand5 dataset (FreiHAND + OneHand10K + etc).
Checkpoint: rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256
"""

from __future__ import annotations

import copy
import itertools
import os
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .base import BaseRepresentation, RepresentationOutput

CONFIG_URL = "https://raw.githubusercontent.com/open-mmlab/mmpose/main/configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py"
CKPT_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth"
CONFIG_PATH = Path("weights/rtmpose_hand_config.py")
CKPT_PATH = Path("weights/rtmpose_hand.pth")


def _ensure_files():
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        print(f"Downloading RTMPose-Hand config -> {CONFIG_PATH}")
        urllib.request.urlretrieve(CONFIG_URL, str(CONFIG_PATH))
    if not CKPT_PATH.exists():
        print(f"Downloading RTMPose-Hand checkpoint -> {CKPT_PATH}")
        urllib.request.urlretrieve(CKPT_URL, str(CKPT_PATH))


def normalize_landmarks(points: list[list[int]]) -> list[float]:
    temp = copy.deepcopy(points)
    base_x, base_y = temp[0][0], temp[0][1]
    for point in temp:
        point[0] -= base_x
        point[1] -= base_y
    flat = list(itertools.chain.from_iterable(temp))
    max_value = max(map(abs, flat)) or 1
    return [v / max_value for v in flat]


class MMPoseLandmarksRepresentation(BaseRepresentation):
    """MMPose RTMPose-Hand → 21 keypoints → 42-dim normalized features."""
    name = "mmpose_landmarks"
    output_type = "features"

    def __init__(self, *, device: str = "cpu"):
        _ensure_files()
        try:
            from mmpose.apis import init_model, inference_topdown
            from mmpose.utils import register_all_modules
            register_all_modules()
            self._model = init_model(str(CONFIG_PATH), str(CKPT_PATH), device=device)
            self._inference_fn = inference_topdown
            self._backend = "mmpose"
        except ImportError:
            raise ImportError(
                "MMPose not installed. Run: pip install mmpose mmcv mmdet mmengine"
            )

    def process(self, frame_rgb: np.ndarray) -> Optional[RepresentationOutput]:
        h, w = frame_rgb.shape[:2]
        # Full-image bounding box as input (no separate hand detector)
        bboxes = [{"bbox": [0, 0, w, h]}]

        results = self._inference_fn(self._model, frame_rgb, bboxes)
        if not results:
            return None

        keypoints = results[0].pred_instances.keypoints[0]  # (21, 2)
        if len(keypoints) != 21:
            return None

        points = [[int(x), int(y)] for x, y in keypoints]
        features = normalize_landmarks(points)

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        bbox = (min(xs) - 10, min(ys) - 10, max(xs) + 10, max(ys) + 10)

        return RepresentationOutput(
            data=features,
            output_type="features",
            bbox=bbox,
            landmarks=points,
            confidence=1.0,
        )

    def visualize(self, frame: np.ndarray, output: RepresentationOutput) -> np.ndarray:
        if output.landmarks:
            for x, y in output.landmarks:
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        return frame
