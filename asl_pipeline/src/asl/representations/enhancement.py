from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .base import BaseRepresentation, RepresentationOutput


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0,
                tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def apply_gamma(img: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, table)


def apply_sharpening(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)


def apply_denoising(img: np.ndarray, h: float = 10.0) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)


def apply_zero_dce(img: np.ndarray) -> np.ndarray:
    from .zero_dce import apply_zero_dce as _apply
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    return _apply(img, device=device)


ENHANCEMENT_METHODS = {
    "clahe": apply_clahe,
    "gamma": apply_gamma,
    "sharpening": apply_sharpening,
    "denoising": apply_denoising,
    "zero_dce": apply_zero_dce,
}


class EnhancementRepresentation(BaseRepresentation):
    name = "enhancement"
    output_type = "image"

    def __init__(self, *, method: str = "clahe", image_size: int = 224,
                 clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8),
                 gamma: float = 1.5):
        self.method = method
        self.image_size = image_size
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.gamma = gamma

        if method not in ENHANCEMENT_METHODS and method != "clahe_gamma":
            raise ValueError(f"Unknown enhancement method: {method}. "
                             f"Available: {list(ENHANCEMENT_METHODS.keys()) + ['clahe_gamma']}")

    def process(self, frame_rgb: np.ndarray) -> Optional[RepresentationOutput]:
        if self.method == "clahe":
            enhanced = apply_clahe(frame_rgb, self.clip_limit, self.tile_grid_size)
        elif self.method == "gamma":
            enhanced = apply_gamma(frame_rgb, self.gamma)
        elif self.method == "sharpening":
            enhanced = apply_sharpening(frame_rgb)
        elif self.method == "denoising":
            enhanced = apply_denoising(frame_rgb)
        elif self.method == "clahe_gamma":
            enhanced = apply_clahe(frame_rgb, self.clip_limit, self.tile_grid_size)
            enhanced = apply_gamma(enhanced, self.gamma)
        else:
            enhanced = frame_rgb

        resized = cv2.resize(enhanced, (self.image_size, self.image_size))
        return RepresentationOutput(
            data=resized,
            output_type="image",
            metadata={"enhancement": self.method},
        )
