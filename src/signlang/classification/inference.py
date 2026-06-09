"""Inference-time wrapper around a trained classifier checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image

from ..data.transforms import build_inference_transform


@dataclass
class Prediction:
    label: str
    confidence: float
    class_id: int
    top5: List[tuple[str, float]]      # [(label, confidence), ...]


class LetterClassifier:
    """Loads a checkpoint produced by :func:`signlang.classification.train`."""

    def __init__(self, weights: str | Path, *, device: str | None = None):
        ckpt = torch.load(str(weights), map_location="cpu")
        self.arch: str = ckpt["arch"]
        self.class_names: List[str] = ckpt["class_names"]
        self.num_classes: int = int(ckpt["num_classes"])
        self.image_size: int = int(ckpt["image_size"])

        self.model = self._build_arch(self.arch, self.num_classes)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self.model.to(self.device)
        self._transform = build_inference_transform(image_size=self.image_size)

    @staticmethod
    def _build_arch(arch: str, num_classes: int):
        # Re-use the same factory the trainer uses so we don't drift.
        from ..utils.config import Config
        from .models import build_classifier
        cfg = Config({
            "model": {"arch": arch, "num_classes": num_classes,
                      "pretrained": False, "freeze_backbone": False},
        })
        return build_classifier(cfg)

    def _preprocess(self, image_rgb: np.ndarray) -> torch.Tensor:
        # image_rgb: HxWx3 uint8. Use the same torchvision v2 pipeline as
        # build_eval_transform so train- and inference-time preprocessing match.
        pil = Image.fromarray(image_rgb)
        tensor = self._transform(pil).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def predict(self, image_rgb: np.ndarray, *, top_k: int = 5) -> Prediction:
        x = self._preprocess(image_rgb)
        logits = self.model(x).squeeze(0)
        probs = logits.softmax(-1).cpu().numpy()
        top_idx = probs.argsort()[::-1][:top_k]
        top5 = [(self.class_names[i], float(probs[i])) for i in top_idx]
        best = top_idx[0]
        return Prediction(
            label=self.class_names[best],
            confidence=float(probs[best]),
            class_id=int(best),
            top5=top5,
        )

    def predict_batch(self, images_rgb: Sequence[np.ndarray]) -> List[Prediction]:
        return [self.predict(img) for img in images_rgb]
