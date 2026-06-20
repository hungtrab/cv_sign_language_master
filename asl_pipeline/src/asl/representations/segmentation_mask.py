from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BaseRepresentation, RepresentationOutput


class SegmentationMaskRepresentation(BaseRepresentation):
    """Hand segmentation → binary mask → masked crop.

    Requires a trained hand segmentation model (YOLO11-seg, SAM, U-Net, or DeepLabV3+).
    """
    name = "segmentation_mask"
    output_type = "image"

    def __init__(self, *, segmenter: str = "sam_or_yolo11seg",
                 model_path: str = "weights/hand_seg.pt",
                 image_size: int = 224):
        raise NotImplementedError(
            "SegmentationMaskRepresentation requires a trained hand segmentation model. "
            f"Segmenter: {segmenter}, expected weights: {model_path}. "
            "Options: YOLO11-seg, SAM/SAM2 with box prompt, U-Net, DeepLabV3+."
        )

    def process(self, frame_rgb: np.ndarray) -> Optional[RepresentationOutput]:
        raise NotImplementedError
