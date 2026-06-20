"""Predict ASL letter from a single image.

Usage:
    python predict_image.py --image path/to/image.jpg
    python predict_image.py --image path.jpg --pipeline raw_hf --output result.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from asl.pipelines.registry import build_pipeline, list_pipelines, register_defaults
from asl.utils.visualization import draw_bbox, draw_label


def main():
    parser = argparse.ArgumentParser(description="Predict ASL letter from image")
    parser.add_argument("--image", default=None, help="Path to input image")
    parser.add_argument("--pipeline", default="mediapipe_resnet18")
    parser.add_argument("--output", default=None, help="Save annotated image to this path")
    parser.add_argument("--list-pipelines", action="store_true")
    args = parser.parse_args()

    register_defaults()

    if args.list_pipelines:
        for name in list_pipelines():
            print(f"  - {name}")
        return

    if not args.image:
        parser.error("--image is required (unless using --list-pipelines)")

    img_path = Path(args.image)
    if not img_path.exists():
        raise SystemExit(f"Image not found: {img_path}")

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise SystemExit(f"Could not read image: {img_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    print(f"Pipeline: {args.pipeline}")
    pipeline = build_pipeline(args.pipeline)
    prediction = pipeline.predict_image(img_rgb)

    if prediction is None:
        print("No hand detected in image.")
        return

    print(f"Prediction: {prediction.label}")
    print(f"Confidence: {prediction.confidence:.4f}")
    print(f"Top-5:")
    for label, conf in prediction.top_k:
        print(f"  {label}: {conf:.4f}")

    if args.output:
        annotated = img_bgr.copy()
        draw_label(annotated,
                   f"{prediction.label} ({prediction.confidence * 100:.0f}%)",
                   origin=(10, 30), color=(0, 255, 0), font_scale=1.0, thickness=2)
        cv2.imwrite(args.output, annotated)
        print(f"Saved annotated image -> {args.output}")

    pipeline.close()


if __name__ == "__main__":
    main()
