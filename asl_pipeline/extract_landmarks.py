"""Extract MediaPipe hand landmarks from an image dataset.

Per next_steps.md Task 5:
  Reads ImageFolder → runs MediaPipe → saves CSV with landmarks + label.

Usage:
    python extract_landmarks.py --dataset data/asl_alphabet/train --output data/landmarks.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from asl.representations.mediapipe_landmarks import (
    MediaPipeLandmarksRepresentation,
    normalize_landmarks,
)
from asl.utils.logging import get_logger

log = get_logger("asl.extract")

AZ_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def iter_images(root: Path, class_names: list[str]):
    for cls_name in class_names:
        d = root / cls_name
        if not d.exists():
            continue
        for img_path in sorted(d.glob("*")):
            try:
                img = np.array(Image.open(img_path).convert("RGB"))
            except Exception:
                continue
            yield img, cls_name, str(img_path)


def main():
    parser = argparse.ArgumentParser(description="Extract hand landmarks from images")
    parser.add_argument("--dataset", required=True, help="Path to ImageFolder root")
    parser.add_argument("--output", default="data/landmarks.csv")
    parser.add_argument("--class-names", default="configs/asl_classes_az.txt")
    args = parser.parse_args()

    root = Path(args.dataset)
    if not root.exists():
        raise SystemExit(f"Dataset not found: {root}")

    class_names = AZ_CLASSES
    if Path(args.class_names).exists():
        from asl.utils.config import load_class_names
        class_names = load_class_names(args.class_names)

    extractor = MediaPipeLandmarksRepresentation()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    header = ["label", "image_path", "hand_detected"] + [f"f{i}" for i in range(42)]

    total = 0
    detected = 0

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for img, cls_name, img_path in tqdm(iter_images(root, class_names), desc="Extracting"):
            total += 1
            result = extractor.process(img)

            if result is None:
                writer.writerow([cls_name, img_path, False] + [0.0] * 42)
                continue

            detected += 1
            features = result.data
            writer.writerow([cls_name, img_path, True] + [round(f, 6) for f in features])

    extractor.close()
    log.info(f"Extracted {detected}/{total} images with hand detected ({detected/max(1,total)*100:.1f}%)")
    log.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
