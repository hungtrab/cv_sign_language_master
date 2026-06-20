from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

import numpy as np


@dataclass
class RepresentationOutput:
    data: Any
    output_type: str
    bbox: Optional[tuple[float, float, float, float]] = None
    landmarks: Optional[List[list[float]]] = None
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)


class BaseRepresentation(ABC):
    name: str = "base"
    output_type: str = "image"

    @abstractmethod
    def process(self, frame_rgb: np.ndarray) -> Optional[RepresentationOutput]:
        ...

    def visualize(self, frame: np.ndarray, output: RepresentationOutput) -> np.ndarray:
        return frame

    def close(self) -> None:
        pass
