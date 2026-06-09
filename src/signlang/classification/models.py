"""CNN model factory for the second-stage letter classifier.

Supported architectures (from ``cfg.model.arch``):

* ``resnet18`` — torchvision pretrained ResNet18 with the FC head replaced.
* ``mobilenet_v2`` — torchvision pretrained MobileNetV2 with classifier head replaced.

Both load ImageNet weights when ``cfg.model.pretrained`` is true.
"""

from __future__ import annotations

import torch
from torch import nn


def _replace_resnet_head(model: nn.Module, num_classes: int) -> nn.Module:
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def _replace_mobilenet_head(model: nn.Module, num_classes: int, *, dropout: float = 0.2) -> nn.Module:
    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def build_classifier(cfg) -> nn.Module:
    """Construct a classifier described by ``cfg.model``."""
    arch = cfg.model.arch
    num_classes = int(cfg.model.num_classes)
    pretrained = bool(cfg.model.pretrained)

    if arch == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model = _replace_resnet_head(model, num_classes)

    elif arch == "mobilenet_v2":
        from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = mobilenet_v2(weights=weights)
        model = _replace_mobilenet_head(model, num_classes)

    else:
        raise ValueError(f"Unknown architecture '{arch}' (expected 'resnet18' or 'mobilenet_v2').")

    if bool(cfg.model.freeze_backbone):
        for name, p in model.named_parameters():
            if not name.startswith("fc") and not name.startswith("classifier"):
                p.requires_grad = False

    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
