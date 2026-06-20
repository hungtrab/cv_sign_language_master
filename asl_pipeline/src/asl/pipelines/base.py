from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..recognizers.base import BaseRecognizer, Prediction
from ..representations.base import BaseRepresentation, RepresentationOutput
from ..utils.smoothing import PredictionSmoother


@dataclass
class FrameOutput:
    prediction: Optional[Prediction]
    accepted_letter: Optional[str]
    text_buffer: str
    fps: float
    elapsed_ms: float
    representation_output: Optional[RepresentationOutput]


class Pipeline:
    def __init__(
        self,
        representation: BaseRepresentation,
        recognizer: BaseRecognizer,
        smoother: Optional[PredictionSmoother] = None,
    ):
        if representation.output_type != recognizer.input_type:
            raise ValueError(
                f"Type mismatch: representation outputs '{representation.output_type}' "
                f"but recognizer expects '{recognizer.input_type}'"
            )
        self.representation = representation
        self.recognizer = recognizer
        self.smoother = smoother
        self._fps_times: deque[float] = deque(maxlen=30)

    @property
    def name(self) -> str:
        return f"{self.representation.name}+{self.recognizer.name}"

    def _tick_fps(self) -> float:
        now = time.monotonic()
        self._fps_times.append(now)
        if len(self._fps_times) < 2:
            return 0.0
        dt = self._fps_times[-1] - self._fps_times[0]
        return (len(self._fps_times) - 1) / dt if dt > 0 else 0.0

    def predict_frame(self, frame_rgb: np.ndarray) -> FrameOutput:
        t0 = time.monotonic()
        rep_output = self.representation.process(frame_rgb)

        prediction: Optional[Prediction] = None
        accepted: Optional[str] = None

        if rep_output is not None:
            prediction = self.recognizer.predict(rep_output)
            if self.smoother and prediction:
                accepted = self.smoother.update(prediction.label, prediction.confidence)
        else:
            if self.smoother:
                self.smoother.observe_neutral()

        elapsed = (time.monotonic() - t0) * 1000
        text_buffer = self.smoother.text_buffer if self.smoother else ""

        return FrameOutput(
            prediction=prediction,
            accepted_letter=accepted,
            text_buffer=text_buffer,
            fps=self._tick_fps(),
            elapsed_ms=elapsed,
            representation_output=rep_output,
        )

    def predict_image(self, image_rgb: np.ndarray) -> Optional[Prediction]:
        rep_output = self.representation.process(image_rgb)
        if rep_output is None:
            return None
        return self.recognizer.predict(rep_output)

    def reset(self) -> None:
        if self.smoother:
            self.smoother.reset()

    def close(self) -> None:
        self.representation.close()
