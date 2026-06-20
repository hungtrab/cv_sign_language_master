from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from ..representations.base import RepresentationOutput


@dataclass
class Prediction:
    label: str
    confidence: float
    class_id: int
    top_k: List[tuple[str, float]]


class BaseRecognizer(ABC):
    name: str = "base"
    input_type: str = "image"
    class_names: List[str] = []

    @abstractmethod
    def predict(self, rep_output: RepresentationOutput) -> Prediction:
        ...

    def predict_batch(self, rep_outputs: List[RepresentationOutput]) -> List[Prediction]:
        return [self.predict(r) for r in rep_outputs]
