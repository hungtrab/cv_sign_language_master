from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .base import BaseRepresentation, RepresentationOutput


class RawImageRepresentation(BaseRepresentation):
    name = "raw_image"
    output_type = "image"

    def __init__(self, *, image_size: int = 224):
        self.image_size = image_size

    def process(self, frame_rgb: np.ndarray) -> Optional[RepresentationOutput]:
        resized = cv2.resize(frame_rgb, (self.image_size, self.image_size))
        return RepresentationOutput(
            data=resized,
            output_type="image",
        )
