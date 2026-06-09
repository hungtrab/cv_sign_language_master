"""Image transforms for training and inference.

Uses torchvision v2 transforms because they handle both PIL and torch
Tensors interchangeably.
"""

from __future__ import annotations

import torch
from torchvision.transforms import v2 as T


# ImageNet stats — a sensible default for pretrained backbones.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(cfg) -> T.Compose:
    aug = cfg.data.augment
    image_size = int(cfg.data.image_size)
    transforms = [
        T.Resize((image_size, image_size), antialias=True),
    ]
    if bool(aug.get("horizontal_flip", False)):
        transforms.append(T.RandomHorizontalFlip(0.5))
    if float(aug.get("rotation_deg", 0)) > 0:
        transforms.append(T.RandomRotation(degrees=float(aug.rotation_deg)))
    if float(aug.get("color_jitter", 0)) > 0:
        cj = float(aug.color_jitter)
        transforms.append(T.ColorJitter(brightness=cj, contrast=cj, saturation=cj))
    transforms += [
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return T.Compose(transforms)


def build_eval_transform(cfg) -> T.Compose:
    image_size = int(cfg.data.image_size)
    return T.Compose([
        T.Resize((image_size, image_size), antialias=True),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_inference_transform(*, image_size: int) -> T.Compose:
    """Same as ``build_eval_transform`` but only needs ``image_size``.

    Used at inference time when we have a checkpoint but no training cfg.
    """
    return T.Compose([
        T.Resize((image_size, image_size), antialias=True),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
