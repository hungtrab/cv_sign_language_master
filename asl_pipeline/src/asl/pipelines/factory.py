from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..utils.config import load_config, Config
from ..utils.smoothing import PredictionSmoother
from .base import Pipeline

REPR_MAP = {
    "raw_image": "asl.representations.raw_image.RawImageRepresentation",
    "mediapipe_crop": "asl.representations.mediapipe_crop.MediaPipeCropRepresentation",
    "mediapipe_landmarks": "asl.representations.mediapipe_landmarks.MediaPipeLandmarksRepresentation",
    "enhancement": "asl.representations.enhancement.EnhancementRepresentation",
    "yolo_crop": "asl.representations.yolo_crop.YOLOCropRepresentation",
    "yolo_hand_pose": "asl.representations.yolo_hand_pose.YOLOHandPoseRepresentation",
    "hand_segmentation": "asl.representations.segmentation_mask.SegmentationMaskRepresentation",
    "fusion_rgb_landmarks": "asl.representations.fusion_rgb_landmarks.FusionRGBLandmarksRepresentation",
}

RECOG_MAP = {
    "resnet18_asl": "asl.recognizers.resnet18_asl.ResNet18ASLRecognizer",
    "hf_image_classifier": "asl.recognizers.hf_image_classifier.HFImageClassifier",
    "landmark_mlp": "asl.recognizers.landmark_mlp.LandmarkMLPRecognizer",
    "torchvision_classifier": "asl.recognizers.torchvision_classifier.TorchvisionClassifier",
    "fusion_classifier": "asl.recognizers.fusion_classifier.FusionClassifier",
    "yolo_direct_detector": "asl.recognizers.yolo_direct_detector.YOLODirectDetector",
}


def _import_class(dotted_path: str):
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def build_pipeline_from_config(
    pipeline_name: str,
    config_path: str = "configs/pipelines.yaml",
    *,
    smoother: Optional[PredictionSmoother] = None,
) -> Pipeline:
    cfg = load_config(config_path)
    pipelines_cfg = cfg.get("pipelines", {})

    if pipeline_name not in pipelines_cfg:
        available = ", ".join(pipelines_cfg.keys()) if pipelines_cfg else "(none)"
        raise ValueError(f"Pipeline '{pipeline_name}' not in {config_path}. Available: {available}")

    pcfg = Config(pipelines_cfg[pipeline_name])

    # Build representation
    repr_cfg = pcfg.get("representation", {})
    repr_type = repr_cfg.get("type", "raw_image") if isinstance(repr_cfg, dict) else "raw_image"

    if repr_type == "none":
        from ..representations.raw_image import RawImageRepresentation
        representation = RawImageRepresentation()
    elif repr_type in REPR_MAP:
        repr_cls = _import_class(REPR_MAP[repr_type])
        repr_kwargs = {k: v for k, v in (repr_cfg if isinstance(repr_cfg, dict) else {}).items()
                       if k != "type"}
        representation = repr_cls(**repr_kwargs)
    else:
        raise ValueError(f"Unknown representation type: {repr_type}")

    # Build recognizer
    recog_cfg = pcfg.get("recognizer", {})
    recog_type = recog_cfg.get("type", "torchvision_classifier") if isinstance(recog_cfg, dict) else "torchvision_classifier"

    if recog_type in RECOG_MAP:
        recog_cls = _import_class(RECOG_MAP[recog_type])
        recog_kwargs = {k: v for k, v in (recog_cfg if isinstance(recog_cfg, dict) else {}).items()
                        if k != "type"}
        recognizer = recog_cls(**recog_kwargs)
    else:
        raise ValueError(f"Unknown recognizer type: {recog_type}")

    # Build smoother from postprocess config
    if smoother is None:
        post_cfg = pcfg.get("postprocess", {})
        if isinstance(post_cfg, dict):
            smooth_cfg = post_cfg.get("smoothing", None)
            if smooth_cfg and isinstance(smooth_cfg, dict) and smooth_cfg.get("type") == "majority_vote":
                smoother = PredictionSmoother(
                    smoothing_window=int(smooth_cfg.get("window", 7)),
                )

    return Pipeline(representation, recognizer, smoother=smoother)
