"""OpenCV drawing helpers (bbox, labels, text-buffer overlay)."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


def draw_bbox(
    frame: np.ndarray,
    bbox_xyxy: Sequence[float],
    *,
    color: tuple[int, int, int] = (0, 200, 255),
    thickness: int = 2,
) -> None:
    """Draw a single bounding box in-place on ``frame``."""
    x1, y1, x2, y2 = (int(v) for v in bbox_xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def draw_label(
    frame: np.ndarray,
    text: str,
    *,
    origin: tuple[int, int],
    color: tuple[int, int, int] = (40, 220, 40),
    bg: tuple[int, int, int] = (0, 0, 0),
    font_scale: float = 0.7,
    thickness: int = 2,
) -> None:
    """Draw text with a filled background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    cv2.rectangle(frame, (x, y - h - baseline - 4), (x + w + 4, y + 4), bg, -1)
    cv2.putText(frame, text, (x + 2, y - 2), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_text_buffer(
    frame: np.ndarray,
    buffer: str,
    *,
    color: tuple[int, int, int] = (40, 220, 40),
    margin: int = 12,
) -> None:
    """Render the running 'spelt' buffer at the bottom-left corner."""
    h, _ = frame.shape[:2]
    draw_label(frame, f"> {buffer}", origin=(margin, h - margin),
               color=color, bg=(0, 0, 0), font_scale=0.8, thickness=2)
