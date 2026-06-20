from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from .base import BaseRecognizer, Prediction
from ..representations.base import RepresentationOutput
from ..data.transforms import build_inference_transform

AZ_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

BACKBONE_REGISTRY = {
    "resnet18": ("torchvision.models", "resnet18", "fc"),
    "resnet50": ("torchvision.models", "resnet50", "fc"),
    "mobilenet_v3_small": ("torchvision.models", "mobilenet_v3_small", "classifier"),
    "efficientnet_b0": ("torchvision.models", "efficientnet_b0", "classifier"),
}


class TorchvisionClassifier(BaseRecognizer):
    """Generic torchvision backbone classifier for A-Z."""
    name = "torchvision_classifier"
    input_type = "image"

    def __init__(self, *, backbone: str = "resnet18", num_classes: int = 26,
                 weights_path: Optional[str] = None, image_size: int = 224,
                 device: Optional[str] = None):
        self.class_names = AZ_CLASSES[:num_classes]
        self.image_size = image_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = self._build_model(backbone, num_classes)

        if weights_path and Path(weights_path).exists():
            sd = torch.load(weights_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(sd, strict=False)

        self.model.eval()
        self.model.to(self.device)
        self._transform = build_inference_transform(image_size=image_size)

    def _build_model(self, backbone: str, num_classes: int):
        import torchvision.models as models

        if backbone == "resnet18":
            model = models.resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        elif backbone == "resnet50":
            model = models.resnet50(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        elif backbone == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(weights=None)
            model.classifier[-1] = torch.nn.Linear(
                model.classifier[-1].in_features, num_classes)
        elif backbone == "efficientnet_b0":
            model = models.efficientnet_b0(weights=None)
            model.classifier[-1] = torch.nn.Linear(
                model.classifier[-1].in_features, num_classes)
        else:
            raise ValueError(f"Unknown backbone: {backbone}. "
                             f"Available: {list(BACKBONE_REGISTRY.keys())}")
        return model

    @torch.no_grad()
    def predict(self, rep_output: RepresentationOutput) -> Prediction:
        img_rgb = rep_output.data
        pil = Image.fromarray(img_rgb)
        tensor = self._transform(pil).unsqueeze(0).to(self.device)
        logits = self.model(tensor).squeeze(0)
        probs = logits.softmax(-1).cpu().numpy()
        top_idx = probs.argsort()[::-1][:5]
        top_k = [(self.class_names[i], float(probs[i])) for i in top_idx
                  if i < len(self.class_names)]
        best = top_idx[0]
        label = self.class_names[best] if best < len(self.class_names) else "Unknown"
        return Prediction(
            label=label,
            confidence=float(probs[best]),
            class_id=int(best),
            top_k=top_k,
        )
