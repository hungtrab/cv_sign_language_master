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
    from ..recognizers.landmark_sklearn import LandmarkSVMRecognizer, LandmarkRFRecognizer

    # === P1: Raw image → classifier (classifier backbone ablation) ===
    register("raw_hf", RawImageRepresentation, HFImageClassifier)
    register("raw_siglip", RawImageRepresentation, HFImageClassifier)
    register("raw_resnet18", RawImageRepresentation, ResNet18ASLRecognizer)

    # === P2: Hand crop → classifier (detector swap + classifier swap) ===
    register("mediapipe_resnet18", MediaPipeCropRepresentation, ResNet18ASLRecognizer)
    register("mediapipe_crop_resnet18", MediaPipeCropRepresentation, ResNet18ASLRecognizer)
    register("mediapipe_crop_vit", MediaPipeCropRepresentation, HFImageClassifier)

    # YOLO hand crop → classifier
    from ..representations.yolo_crop import YOLOCropRepresentation
    YOLO_WEIGHTS = "weights/yolo_hand.pt"
    register("yolo_crop_resnet18", YOLOCropRepresentation, ResNet18ASLRecognizer,
             repr_kwargs={"model_path": YOLO_WEIGHTS})
    register("yolo_crop_vit", YOLOCropRepresentation, HFImageClassifier,
             repr_kwargs={"model_path": YOLO_WEIGHTS})

    # === Enhanced chains: RGB → Enhancement → Crop/Landmarks → Classifier ===
    from ..representations.enhanced_chain import EnhancedCropRepresentation, EnhancedLandmarksRepresentation

    # CLAHE → MediaPipe crop → ResNet18
    register("clahe_mediapipe_crop_resnet18", EnhancedCropRepresentation, ResNet18ASLRecognizer,
             repr_kwargs={"enhancement": "clahe", "cropper": "mediapipe"})
    # CLAHE → MediaPipe crop → ViT/SigLIP
    register("clahe_mediapipe_crop_vit", EnhancedCropRepresentation, HFImageClassifier,
             repr_kwargs={"enhancement": "clahe", "cropper": "mediapipe"})
    # CLAHE → MediaPipe landmarks → MLP
    register("clahe_mediapipe_landmarks_mlp", EnhancedLandmarksRepresentation, LandmarkMLPRecognizer,
             repr_kwargs={"enhancement": "clahe"})
    # CLAHE → YOLO crop → ResNet18
    register("clahe_yolo_crop_resnet18", EnhancedCropRepresentation, ResNet18ASLRecognizer,
             repr_kwargs={"enhancement": "clahe", "cropper": "yolo", "model_path": YOLO_WEIGHTS})
    # Gamma → MediaPipe crop → ResNet18
    register("gamma_mediapipe_crop_resnet18", EnhancedCropRepresentation, ResNet18ASLRecognizer,
             repr_kwargs={"enhancement": "gamma", "cropper": "mediapipe"})

    # === P3: Landmarks → classifier (landmark classifier swap) ===
    register("landmark_mlp", MediaPipeLandmarksRepresentation, LandmarkMLPRecognizer)
    register("mediapipe_landmarks_mlp", MediaPipeLandmarksRepresentation, LandmarkMLPRecognizer)
    register("mediapipe_landmarks_svm", MediaPipeLandmarksRepresentation, LandmarkSVMRecognizer)
    register("mediapipe_landmarks_rf", MediaPipeLandmarksRepresentation, LandmarkRFRecognizer)

    # === P6: Enhancement → classifier (enhancement swap) ===
    register("no_enhance_resnet18", RawImageRepresentation, ResNet18ASLRecognizer)
    register("enhancement_clahe_resnet18", EnhancementRepresentation, ResNet18ASLRecognizer,
             repr_kwargs={"method": "clahe"})
    register("enhancement_gamma_resnet18", EnhancementRepresentation, ResNet18ASLRecognizer,
             repr_kwargs={"method": "gamma"})
    register("enhancement_sharpen_resnet18", EnhancementRepresentation, ResNet18ASLRecognizer,
             repr_kwargs={"method": "sharpening"})
    register("enhancement_denoise_resnet18", EnhancementRepresentation, ResNet18ASLRecognizer,
             repr_kwargs={"method": "denoising"})
    register("enhancement_clahe_vit", EnhancementRepresentation, HFImageClassifier,
             repr_kwargs={"method": "clahe"})
    register("enhancement_gamma_vit", EnhancementRepresentation, HFImageClassifier,
             repr_kwargs={"method": "gamma"})
