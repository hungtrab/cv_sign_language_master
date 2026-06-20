from __future__ import annotations

from ..recognizers.base import Prediction

AZ_SET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def apply_confidence_threshold(prediction: Prediction | None,
                                threshold: float = 0.5) -> Prediction | None:
    if prediction is None:
        return None
    if prediction.confidence < threshold:
        return Prediction(
            label="Unknown",
            confidence=prediction.confidence,
            class_id=-1,
            top_k=prediction.top_k,
        )
    if prediction.label.upper() not in AZ_SET:
        return Prediction(
            label="Unknown",
            confidence=prediction.confidence,
            class_id=-1,
            top_k=prediction.top_k,
        )
    return prediction
