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
    h, _ = frame.shape[:2]
    draw_label(frame, f"> {buffer}", origin=(margin, h - margin),
               color=color, bg=(0, 0, 0), font_scale=0.8, thickness=2)


def draw_hud(
    frame: np.ndarray,
    *,
    pipeline_name: str,
    letter: str | None = None,
    confidence: float = 0.0,
    fps: float = 0.0,
    elapsed_ms: float = 0.0,
) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 60), (20, 20, 20), -1)
    cv2.putText(frame, f"Pipeline: {pipeline_name}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 220), 1)
    if letter:
        cv2.putText(frame, f"Letter: {letter} ({confidence * 100:.0f}%)",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.0f}  {elapsed_ms:.0f}ms",
                (w - 200, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
