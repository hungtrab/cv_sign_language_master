from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torchvision.models import resnet18

from .base import BaseRecognizer, Prediction
from ..representations.base import RepresentationOutput
from ..data.transforms import build_inference_transform
from ..utils.logging import get_logger

log = get_logger("asl.resnet18")

AZ_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _clean_state_dict(sd: dict) -> dict:
    cleaned = {}
    for k, v in sd.items():
        for prefix in ("module.", "model.", "backbone."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        cleaned[k] = v
    return cleaned


class ResNet18ASLRecognizer(BaseRecognizer):
    name = "resnet18_asl"
    input_type = "image"

    def __init__(self, *, weights_path: Optional[str] = None,
                 class_mapping_path: Optional[str] = None,
                 num_classes: int = 26, image_size: int = 224,
                 device: Optional[str] = None):
        self.image_size = image_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        if weights_path is None:
            from ..utils.download import download_model_option1
            info = download_model_option1()
            weights_path = str(info["weights_path"])
            if info.get("class_mapping"):
                self._build_label_map(info["class_mapping"])
            else:
                self.class_names = AZ_CLASSES
        else:
            self.class_names = AZ_CLASSES

        if class_mapping_path:
            with open(class_mapping_path) as f:
                self._build_label_map(json.load(f))

        if not hasattr(self, "class_names"):
            self.class_names = AZ_CLASSES
        num_classes = len(self.class_names)

        self.model = resnet18(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

        sd = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        if "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        elif "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        elif "state_dict" in sd:
            sd = sd["state_dict"]
        sd = _clean_state_dict(sd)

        try:
            self.model.load_state_dict(sd, strict=True)
        except RuntimeError:
            log.warning("Strict load failed, trying non-strict")
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            if missing:
                log.warning(f"Missing keys: {missing}")
            if unexpected:
                log.warning(f"Unexpected keys: {unexpected}")

        self.model.eval()
        self.model.to(self.device)
        self._transform = build_inference_transform(image_size=image_size)

    def _build_label_map(self, mapping: dict) -> None:
        if not isinstance(mapping, dict):
            self.class_names = AZ_CLASSES
            return

        # Handle nested format: {"class_to_idx": {...}, "idx_to_class": {...}}
        if "idx_to_class" in mapping:
            mapping = mapping["idx_to_class"]
        elif "class_to_idx" in mapping:
            mapping = {str(v): k for k, v in mapping["class_to_idx"].items()}

        idx_to_label = {}
        for k, v in mapping.items():
            try:
                idx_to_label[int(k)] = str(v).upper()
            except (ValueError, TypeError):
                try:
                    idx_to_label[int(v)] = str(k).upper()
                except (ValueError, TypeError):
                    continue
        if not idx_to_label:
            self.class_names = AZ_CLASSES
            return
        max_idx = max(idx_to_label.keys()) + 1
        self.class_names = [idx_to_label.get(i, f"UNK{i}") for i in range(max_idx)]

    @torch.no_grad()
    def predict(self, rep_output: RepresentationOutput) -> Prediction:
        img_rgb = rep_output.data
        pil = Image.fromarray(img_rgb)
        tensor = self._transform(pil).unsqueeze(0).to(self.device)
        logits = self.model(tensor).squeeze(0)
        probs = logits.softmax(-1).cpu().numpy()
        top_idx = probs.argsort()[::-1][:5]
        top_k = [(self.class_names[i], float(probs[i])) for i in top_idx]
        best = top_idx[0]
        return Prediction(
            label=self.class_names[best],
            confidence=float(probs[best]),
            class_id=int(best),
            top_k=top_k,
        )
