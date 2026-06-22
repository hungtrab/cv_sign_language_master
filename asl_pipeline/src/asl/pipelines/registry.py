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
    from ..representations.mediapipe_landmarks import MediaPipeLandmarksRepresentation
    from ..representations.mmpose_landmarks import MMPoseLandmarksRepresentation
    from ..representations.yolo_crop import YOLOCropRepresentation
    from ..representations.enhanced_chain import EnhancedLandmarksRepresentation, EnhancedCropRepresentation
    from ..recognizers.landmark_mlp import LandmarkMLPRecognizer
    from ..recognizers.landmark_xgboost import LandmarkXGBoostRecognizer
    from ..recognizers.resnet18_asl import ResNet18ASLRecognizer

    YOLO_WEIGHTS = "weights/yolo_hand.pt"
    ENHANCEMENTS = ["raw", "clahe", "gamma", "sharpening", "zero_dce"]

    for enh in ENHANCEMENTS:
        prefix = enh

        # --- MediaPipe Landmarks → MLP ---
        if enh == "raw":
            register(f"{prefix}_mp_mlp",
                     MediaPipeLandmarksRepresentation, LandmarkMLPRecognizer)
        else:
            register(f"{prefix}_mp_mlp",
                     EnhancedLandmarksRepresentation, LandmarkMLPRecognizer,
                     repr_kwargs={"enhancement": enh, "pose_backend": "mediapipe"})

        # --- MediaPipe Landmarks → XGBoost ---
        if enh == "raw":
            register(f"{prefix}_mp_xgb",
                     MediaPipeLandmarksRepresentation, LandmarkXGBoostRecognizer)
        else:
            register(f"{prefix}_mp_xgb",
                     EnhancedLandmarksRepresentation, LandmarkXGBoostRecognizer,
                     repr_kwargs={"enhancement": enh, "pose_backend": "mediapipe"})

        # --- MMPose Landmarks → MLP ---
        if enh == "raw":
            register(f"{prefix}_mmpose_mlp",
                     MMPoseLandmarksRepresentation, LandmarkMLPRecognizer)
        else:
            register(f"{prefix}_mmpose_mlp",
                     EnhancedLandmarksRepresentation, LandmarkMLPRecognizer,
                     repr_kwargs={"enhancement": enh, "pose_backend": "mmpose"})

        # --- MMPose Landmarks → XGBoost ---
        if enh == "raw":
            register(f"{prefix}_mmpose_xgb",
                     MMPoseLandmarksRepresentation, LandmarkXGBoostRecognizer)
        else:
            register(f"{prefix}_mmpose_xgb",
                     EnhancedLandmarksRepresentation, LandmarkXGBoostRecognizer,
                     repr_kwargs={"enhancement": enh, "pose_backend": "mmpose"})

        # --- YOLOv11 Crop → ResNet18 ---
        if enh == "raw":
            register(f"{prefix}_yolo_resnet18",
                     YOLOCropRepresentation, ResNet18ASLRecognizer,
                     repr_kwargs={"model_path": YOLO_WEIGHTS})
        else:
            register(f"{prefix}_yolo_resnet18",
                     EnhancedCropRepresentation, ResNet18ASLRecognizer,
                     repr_kwargs={"enhancement": enh, "cropper": "yolo",
                                  "model_path": YOLO_WEIGHTS})
