from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from .base import BaseRecognizer, Prediction
from ..representations.base import RepresentationOutput
from ..utils.logging import get_logger

log = get_logger("asl.hf_classifier")

AZ_SET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class HFImageClassifier(BaseRecognizer):
    name = "hf_classifier"
    input_type = "image"

    def __init__(self, *, model_name: Optional[str] = None):
        if model_name is None:
            from ..utils.download import load_recognizer_with_fallback
            info = load_recognizer_with_fallback()
            if info["arch"] == "hf_transformers":
                self.processor = info["processor"]
                self.model = info["model"]
                self.source = info["source"]
            else:
                raise RuntimeError(
                    "Fallback returned a non-HF model. Use ResNet18ASLRecognizer instead."
                )
        else:
            self._load_model(model_name)

        self.model.eval()
        labels = self.model.config.id2label
        self.class_names = [labels[i] for i in sorted(labels.keys())]
        self._az_indices = {
            i for i, name in enumerate(self.class_names)
            if name.upper() in AZ_SET
        }

    def _load_model(self, model_name: str) -> None:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
        except Exception:
            from transformers import SiglipForImageClassification
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = SiglipForImageClassification.from_pretrained(model_name)
        self.source = model_name

    @torch.no_grad()
    def predict(self, rep_output: RepresentationOutput) -> Prediction:
        img_rgb = rep_output.data
        pil = Image.fromarray(img_rgb)
        inputs = self.processor(images=pil, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits.squeeze(0)
        probs = logits.softmax(-1).cpu().numpy()

        top_idx = probs.argsort()[::-1][:5]
        top_k = [(self.class_names[i].upper(), float(probs[i])) for i in top_idx]

        best = top_idx[0]
        label = self.class_names[best].upper()
        if label not in AZ_SET:
            label = "Unknown"

        return Prediction(
            label=label,
            confidence=float(probs[best]),
            class_id=int(best),
            top_k=top_k,
        )
