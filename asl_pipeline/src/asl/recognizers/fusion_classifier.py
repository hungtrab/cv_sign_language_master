from __future__ import annotations

from .base import BaseRecognizer, Prediction
from ..representations.base import RepresentationOutput


class FusionClassifier(BaseRecognizer):
    """Dual-branch fusion: RGB features + landmark features → A-Z.

    Loss: L_total = L_fusion + λ1 * L_rgb + λ2 * L_landmark
    Default: λ1 = 0.3, λ2 = 0.3
    """
    name = "fusion_classifier"
    input_type = "fusion"

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "FusionClassifier requires training a dual-head model. "
            "See docs/modular_pipelines.md §P7 for architecture and loss design."
        )

    def predict(self, rep_output: RepresentationOutput) -> Prediction:
        raise NotImplementedError
