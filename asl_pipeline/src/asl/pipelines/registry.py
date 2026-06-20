from __future__ import annotations

from typing import Any, Dict, Optional, Type

from ..representations.base import BaseRepresentation
from ..recognizers.base import BaseRecognizer
from ..utils.smoothing import PredictionSmoother
from .base import Pipeline

_REGISTRY: Dict[str, dict] = {}


def register(
    name: str,
    repr_cls: Type[BaseRepresentation],
    recog_cls: Type[BaseRecognizer],
    **kwargs: Any,
) -> None:
    _REGISTRY[name] = {
        "repr_cls": repr_cls,
        "recog_cls": recog_cls,
        "kwargs": kwargs,
    }


def list_pipelines() -> list[str]:
    return list(_REGISTRY.keys())


def build_pipeline(
    name: str,
    *,
    smoother: Optional[PredictionSmoother] = None,
    repr_kwargs: Optional[dict] = None,
    recog_kwargs: Optional[dict] = None,
) -> Pipeline:
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys()) or "(none registered)"
        raise ValueError(f"Unknown pipeline '{name}'. Available: {available}")

    entry = _REGISTRY[name]
    rk = {**entry["kwargs"].get("repr_kwargs", {}), **(repr_kwargs or {})}
    ck = {**entry["kwargs"].get("recog_kwargs", {}), **(recog_kwargs or {})}

    representation = entry["repr_cls"](**rk)
    recognizer = entry["recog_cls"](**ck)

    return Pipeline(representation, recognizer, smoother=smoother)


def register_defaults() -> None:
    from ..representations.mediapipe_crop import MediaPipeCropRepresentation
    from ..representations.mediapipe_landmarks import MediaPipeLandmarksRepresentation
    from ..representations.raw_image import RawImageRepresentation
    from ..representations.enhancement import EnhancementRepresentation
    from ..recognizers.resnet18_asl import ResNet18ASLRecognizer
    from ..recognizers.hf_image_classifier import HFImageClassifier
    from ..recognizers.landmark_mlp import LandmarkMLPRecognizer
    from ..recognizers.torchvision_classifier import TorchvisionClassifier

    register("mediapipe_resnet18", MediaPipeCropRepresentation, ResNet18ASLRecognizer)
    register("mediapipe_crop_resnet18", MediaPipeCropRepresentation, ResNet18ASLRecognizer)
    register("mediapipe_crop_vit", MediaPipeCropRepresentation, HFImageClassifier)
    register("raw_hf", RawImageRepresentation, HFImageClassifier)
    register("raw_siglip", RawImageRepresentation, HFImageClassifier)
    register("raw_resnet18", RawImageRepresentation, TorchvisionClassifier)
    register("landmark_mlp", MediaPipeLandmarksRepresentation, LandmarkMLPRecognizer)
    register("mediapipe_landmarks_mlp", MediaPipeLandmarksRepresentation, LandmarkMLPRecognizer)
    register("enhancement_clahe_resnet18", EnhancementRepresentation, ResNet18ASLRecognizer,
             repr_kwargs={"method": "clahe"})
    register("enhancement_gamma_resnet18", EnhancementRepresentation, ResNet18ASLRecognizer,
             repr_kwargs={"method": "gamma"})
