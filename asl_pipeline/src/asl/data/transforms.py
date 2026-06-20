from __future__ import annotations

import torch
from torchvision.transforms import v2 as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(image_size: int = 224, *, rotation_deg: float = 15,
                           color_jitter: float = 0.2) -> T.Compose:
    transforms = [
        T.Resize((image_size, image_size), antialias=True),
    ]
    if rotation_deg > 0:
        transforms.append(T.RandomRotation(degrees=rotation_deg))
    if color_jitter > 0:
        transforms.append(T.ColorJitter(brightness=color_jitter, contrast=color_jitter,
                                        saturation=color_jitter))
    transforms += [
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return T.Compose(transforms)


def build_eval_transform(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size), antialias=True),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_inference_transform(image_size: int = 224) -> T.Compose:
    return build_eval_transform(image_size)
