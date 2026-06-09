"""Two-stage pipeline: YOLOv8 hand detector -> CNN letter classifier.

Runtime overhead per frame (single hand, 224×224 crop, CPU):

* YOLOv8-nano: ~10–15 ms
* ResNet18 / MobileNetV2: ~10–15 ms

Total well under the 33 ms budget for 30 FPS.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import cv2
import numpy as np

from ..classification.inference import LetterClassifier, Prediction
from ..detection.detector import Detection, HandDetector


@dataclass
class FrameOutput:
    detections: List[Detection]
    prediction: Optional[Prediction]
    accepted_letter: Optional[str]
    text_buffer: str
    fps: float = 0.0
    elapsed_ms: float = 0.0


class _FpsMeter:
    def __init__(self, window: int = 30):
        self.times: Deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        now = time.monotonic()
        self.times.append(now)
        if len(self.times) < 2:
            return 0.0
        dt = self.times[-1] - self.times[0]
        return (len(self.times) - 1) / dt if dt > 0 else 0.0


class TwoStagePipeline:
    """Stateful pipeline with per-frame smoothing of the predicted letter."""

    def __init__(
        self,
        detector: HandDetector,
        classifier: LetterClassifier,
        *,
        cls_conf_threshold: float = 0.7,
        cls_margin_threshold: float = 0.15,
        smoothing_window: int = 5,
        letter_repeat_cooldown_ms: int = 600,
        require_neutral_between_letters: bool = True,
        neutral_frames_to_reset: int = 3,
    ):
        self.detector = detector
        self.classifier = classifier
        self.cls_conf_threshold = float(cls_conf_threshold)
        self.cls_margin_threshold = float(cls_margin_threshold)
        self.smoothing: Deque[str] = deque(maxlen=int(smoothing_window))
        self.cooldown_ms = int(letter_repeat_cooldown_ms)
        self.require_neutral_between_letters = bool(require_neutral_between_letters)
        self.neutral_frames_to_reset = int(neutral_frames_to_reset)
        self._armed: bool = True
        self._neutral_frames: int = 0
        self._last_emitted_at: float = 0.0
        self._last_letter: Optional[str] = None
        self.text_buffer: str = ""
        self._fps = _FpsMeter()

    # ---------------------------------------------------------------- buffer ops

    def reset_text_buffer(self) -> None:
        self.text_buffer = ""

    def save_text_buffer(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.text_buffer)

    # ---------------------------------------------------------------- core

    def _observe_neutral(self) -> None:
        """Hand absent or classifier said 'nothing': count toward re-arming."""
        self.smoothing.clear()
        self._neutral_frames += 1
        if self._neutral_frames >= self.neutral_frames_to_reset:
            self._armed = True
            self._last_letter = None

    def _observe_uncertain(self) -> None:
        """Hand present but classifier confidence/margin too low.

        Drop the smoothing buffer so we don't accumulate noisy frames, but do
        NOT bump neutral_frames — the user is still holding a gesture, so
        ``_armed`` must not flip back to True from a few uncertain frames.
        """
        self.smoothing.clear()

    def _try_emit(self, letter: str) -> Optional[str]:
        """Decide whether to commit ``letter`` to the running buffer."""
        if letter == "nothing":
            self._observe_neutral()
            return None

        self._neutral_frames = 0
        self.smoothing.append(letter)
        if len(self.smoothing) < self.smoothing.maxlen:
            return None
        # majority vote
        counts: dict[str, int] = {}
        for label in self.smoothing:
            counts[label] = counts.get(label, 0) + 1
        majority, count = max(counts.items(), key=lambda kv: kv[1])
        if count <= len(self.smoothing) // 2:
            return None

        if self.require_neutral_between_letters and not self._armed:
            return None

        now = time.monotonic() * 1000
        if majority == self._last_letter and (now - self._last_emitted_at) < self.cooldown_ms:
            return None

        self._last_letter = majority
        self._last_emitted_at = now
        self._armed = False

        if majority == "space":
            self.text_buffer += " "
            return " "
        if majority == "del":
            self.text_buffer = self.text_buffer[:-1]
            return "<DEL>"
        self.text_buffer += majority
        return majority

    def process(self, bgr_frame: np.ndarray) -> FrameOutput:
        t0 = time.monotonic()
        detections = self.detector.detect(bgr_frame, top_k=1)
        prediction: Optional[Prediction] = None
        emitted: Optional[str] = None

        if detections:
            crop_bgr = detections[0].crop(bgr_frame)
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            prediction = self.classifier.predict(crop_rgb)
            second_conf = prediction.top5[1][1] if len(prediction.top5) > 1 else 0.0
            margin = prediction.confidence - second_conf
            if prediction.confidence >= self.cls_conf_threshold and margin >= self.cls_margin_threshold:
                emitted = self._try_emit(prediction.label)
            else:
                self._observe_uncertain()
        else:
            self._observe_neutral()

        elapsed = (time.monotonic() - t0) * 1000
        return FrameOutput(
            detections=detections,
            prediction=prediction,
            accepted_letter=emitted,
            text_buffer=self.text_buffer,
            fps=self._fps.tick(),
            elapsed_ms=elapsed,
        )
