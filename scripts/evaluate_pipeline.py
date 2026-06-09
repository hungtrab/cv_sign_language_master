"""Evaluate the two-stage pipeline on the held-out ASL test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm

from signlang.classification.inference import LetterClassifier
from signlang.detection.detector import HandDetector
from signlang.evaluation import accuracy, confusion_matrix, macro_f1
from signlang.pipeline.two_stage import TwoStagePipeline
from signlang.utils import get_logger, load_config


def _iter_labelled_images(test_root: Path, class_names: List[str]):
    """Yields (rgb_image, class_id) pairs from an ImageFolder-style tree."""
    for cls_idx, name in enumerate(class_names):
        d = test_root / name
        if not d.exists():
            continue
        for img_path in d.glob("*"):
            try:
                img = np.array(Image.open(img_path).convert("RGB"))
            except Exception:
                continue
            yield img, cls_idx, str(img_path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--test-dir", default="data/asl_alphabet/asl_alphabet_test", type=str)
    p.add_argument("--output", default=None, type=str)
    p.add_argument(
        "--skip-detector", action="store_true",
        help="Skip YOLO and feed the full image straight into the classifier. "
             "Use this when the test set is already cropped (e.g. Kaggle ASL).",
    )
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    log = get_logger("signlang.eval")

    classifier = LetterClassifier(cfg.paths.classifier_weights)
    pipeline = None
    if not args.skip_detector:
        detector = HandDetector(cfg.paths.detector_weights,
                                conf_threshold=float(cfg.inference.detection_conf_threshold))
        pipeline = TwoStagePipeline(detector, classifier,
                                    cls_conf_threshold=0.0,    # don't gate during eval
                                    smoothing_window=1)

    class_names = classifier.class_names
    preds, gts = [], []
    test_root = Path(args.test_dir)
    if not test_root.exists():
        raise FileNotFoundError(f"Test dir not found: {test_root}")

    log.info(f"running pipeline over {test_root} (skip_detector={args.skip_detector})")
    for img, cls_idx, _ in tqdm(_iter_labelled_images(test_root, class_names)):
        if args.skip_detector:
            pred = classifier.predict(img)
            preds.append(pred.class_id)
        else:
            # Convert to BGR for the pipeline (which speaks OpenCV).
            bgr = img[..., ::-1].copy()
            out = pipeline.process(bgr)
            preds.append(out.prediction.class_id if out.prediction is not None else -1)
        gts.append(cls_idx)

    preds_np = np.array(preds); gts_np = np.array(gts)
    has_pred = preds_np >= 0
    if not args.skip_detector:
        log.info(f"detection rate: {has_pred.mean() * 100:.1f}%")

    metrics = {
        "accuracy": accuracy(preds_np[has_pred], gts_np[has_pred]),
        "macro_f1": macro_f1(preds_np[has_pred], gts_np[has_pred], num_classes=len(class_names)),
    }
    if not args.skip_detector:
        metrics["detection_rate"] = float(has_pred.mean())
    log.info(json.dumps(metrics, indent=2))

    if args.output:
        Path(args.output).write_text(json.dumps(metrics, indent=2))
        cm = confusion_matrix(preds_np[has_pred], gts_np[has_pred], num_classes=len(class_names))
        np.savetxt(Path(args.output).with_suffix(".cm.csv"), cm, fmt="%d", delimiter=",")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
