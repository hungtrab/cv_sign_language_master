from __future__ import annotations

import cv2
import numpy as np


def apply_low_light(img: np.ndarray, factor: float = 0.3) -> np.ndarray:
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def apply_overexposure(img: np.ndarray, factor: float = 2.0) -> np.ndarray:
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def apply_motion_blur(img: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    return cv2.filter2D(img, -1, kernel)


def apply_gaussian_noise(img: np.ndarray, std: float = 25.0) -> np.ndarray:
    noise = np.random.default_rng(42).normal(0, std, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_rotation(img: np.ndarray, degrees: float = 15.0) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def apply_scale_shift(img: np.ndarray, scale: float = 0.8) -> np.ndarray:
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h))
    pad_top = (h - new_h) // 2
    pad_left = (w - new_w) // 2
    result = np.zeros_like(img)
    result[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return result


def apply_partial_occlusion(img: np.ndarray, fraction: float = 0.2) -> np.ndarray:
    h, w = img.shape[:2]
    bh, bw = int(h * fraction), int(w * fraction)
    result = img.copy()
    y = (h - bh) // 2
    x = (w - bw) // 2
    result[y:y + bh, x:x + bw] = 0
    return result


def apply_crop_shift(img: np.ndarray, shift_x: int = 20, shift_y: int = 20) -> np.ndarray:
    h, w = img.shape[:2]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


ALL_CORRUPTIONS = {
    "low_light": apply_low_light,
    "overexposure": apply_overexposure,
    "motion_blur": apply_motion_blur,
    "gaussian_noise": apply_gaussian_noise,
    "rotation": apply_rotation,
    "scale_shift": apply_scale_shift,
    "crop_shift": apply_crop_shift,
    "partial_occlusion": apply_partial_occlusion,
}
