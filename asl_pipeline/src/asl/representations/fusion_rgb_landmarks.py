from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BaseRepresentation, RepresentationOutput


class FusionRGBLandmarksRepresentation(BaseRepresentation):
    """Dual-branch: RGB hand crop + normalized landmark vector.

    Returns both a cropped image and a landmark feature vector for fusion.
    """
    name = "fusion_rgb_landmarks"
    output_type = "fusion"

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "FusionRGBLandmarksRepresentation is an advanced pipeline. "
            "It requires a trained fusion classifier that accepts both RGB features "
            "and landmark features. See docs/modular_pipelines.md for architecture."
        )

    def process(self, frame_rgb: np.ndarray) -> Optional[RepresentationOutput]:
        raise NotImplementedError
