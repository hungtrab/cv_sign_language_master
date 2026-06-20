from __future__ import annotations

import time
from collections import deque
from typing import Optional


class PredictionSmoother:
    """Majority-vote smoothing + debounce + neutral-gate for real-time demos."""

    def __init__(
        self,
        *,
        smoothing_window: int = 7,
        commit_frames: int = 10,
        cooldown_ms: int = 600,
        require_neutral: bool = True,
        neutral_frames_to_reset: int = 3,
    ):
        self.history: deque[str] = deque(maxlen=smoothing_window)
        self.commit_frames = commit_frames
        self.cooldown_ms = cooldown_ms
        self.require_neutral = require_neutral
        self.neutral_frames_to_reset = neutral_frames_to_reset
        self._armed: bool = True
        self._neutral_frames: int = 0
        self._last_emitted_at: float = 0.0
        self._last_letter: Optional[str] = None
        self._stable_count: int = 0
        self._last_smoothed: Optional[str] = None
        self.text_buffer: str = ""

    def reset(self) -> None:
        self.text_buffer = ""
        self.history.clear()
        self._armed = True
        self._neutral_frames = 0
        self._last_letter = None
        self._stable_count = 0
        self._last_smoothed = None

    def observe_neutral(self) -> None:
        self.history.clear()
        self._stable_count = 0
        self._last_smoothed = None
        self._neutral_frames += 1
        if self._neutral_frames >= self.neutral_frames_to_reset:
            self._armed = True
            self._last_letter = None

    def observe_uncertain(self) -> None:
        self.history.clear()
        self._stable_count = 0
        self._last_smoothed = None

    def update(self, label: str | None, confidence: float = 1.0) -> Optional[str]:
        if label is None:
            self.observe_neutral()
            return None

        self._neutral_frames = 0
        self.history.append(label)

        # Majority vote
        counts: dict[str, int] = {}
        for l in self.history:
            counts[l] = counts.get(l, 0) + 1
        majority, count = max(counts.items(), key=lambda kv: kv[1])
        if count <= len(self.history) // 2:
            return None

        smoothed = majority

        # Commit stability check
        if smoothed == self._last_smoothed:
            self._stable_count += 1
        else:
            self._last_smoothed = smoothed
            self._stable_count = 1

        if self._stable_count < self.commit_frames:
            return None

        # Neutral gate
        if self.require_neutral and not self._armed:
            return None

        # Cooldown
        now = time.monotonic() * 1000
        if smoothed == self._last_letter and (now - self._last_emitted_at) < self.cooldown_ms:
            return None

        # Commit
        self._last_letter = smoothed
        self._last_emitted_at = now
        self._armed = False
        self._stable_count = 0

        if smoothed == "space":
            self.text_buffer += " "
            return " "
        if smoothed == "del":
            self.text_buffer = self.text_buffer[:-1]
            return "<DEL>"
        self.text_buffer += smoothed
        return smoothed
