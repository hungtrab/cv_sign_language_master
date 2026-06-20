"""Evaluate a pipeline on an ImageFolder-style test set.

Outputs per eval doc §8.1:
  - metrics JSON (accuracy, top3, macro-F1, per-class, latency, FPS, detection rate)
  - predictions CSV (image_path, true_label, pred_label, confidence, top3, latency, hand_detected)
  - confusion matrix PNG
  - failure cases directory
  - environment JSON
  - run config YAML

Usage:
    python evaluate.py --dataset data/asl_alphabet/test --pipeline mediapipe_resnet18
    python evaluate.py --dataset data/test --pipeline raw_hf --confidence-threshold 0.6
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from asl.pipelines.registry import build_pipeline, register_defaults
from asl.evaluation.metrics import (
    accuracy, macro_f1, per_class_f1, per_class_precision, per_class_recall,
    weighted_f1, confusion_matrix,
)
from asl.evaluation.confusion import plot_confusion_matrix, save_predictions_csv
from asl.postprocess.threshold import apply_confidence_threshold
from asl.utils.config import load_class_names
from asl.utils.logging import get_logger

log = get_logger("asl.evaluate")

AZ_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def iter_test_images(test_root: Path, class_names: list[str]):
    for cls_idx, name in enumerate(class_names):
        d = test_root / name
        if not d.exists():
            continue
        for img_path in sorted(d.glob("*")):
            try:
                img = np.array(Image.open(img_path).convert("RGB"))
            except Exception:
                continue
            yield img, cls_idx, name, str(img_path)


def save_environment(output_dir: Path):
    env = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "opencv_version": cv2.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "platform": platform.platform(),
    }
    try:
        import mediapipe as mp
        env["mediapipe_version"] = mp.__version__
    except ImportError:
        pass
    try:
        import transformers
        env["transformers_version"] = transformers.__version__
    except ImportError:
        pass
    with open(output_dir / "environment.json", "w") as f:
        json.dump(env, f, indent=2)


def save_failure_case(failure_dir: Path, img_path: str, true_label: str,
                       pred_label: str, confidence: float, top3: list,
                       hand_detected: bool):
    failure_dir.mkdir(parents=True, exist_ok=True)
    src = Path(img_path)
    if src.exists():
        import shutil
        dest = failure_dir / f"{true_label}_as_{pred_label}_{src.name}"
        shutil.copy2(src, dest)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ASL pipeline")
    parser.add_argument("--dataset", required=True, help="Path to ImageFolder test set")
    parser.add_argument("--pipeline", default="mediapipe_resnet18")
    parser.add_argument("--output", default="outputs/", help="Output root directory")
    parser.add_argument("--class-names", default="configs/asl_classes_az.txt")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--save-failures", action="store_true", default=True)
    args = parser.parse_args()

    register_defaults()

    test_root = Path(args.dataset)
    if not test_root.exists():
        raise SystemExit(f"Test directory not found: {test_root}")

    class_names = AZ_CLASSES
    if Path(args.class_names).exists():
        class_names = load_class_names(args.class_names)

    log.info(f"Pipeline: {args.pipeline}")
    log.info(f"Test set: {test_root}")
    log.info(f"Classes: {len(class_names)}")
    log.info(f"Confidence threshold: {args.confidence_threshold}")

    pipeline = build_pipeline(args.pipeline)

    # Per-prediction records
    records = []
    latencies = []
    no_detection = 0
    unknown_count = 0
    total = 0

    for img, cls_idx, cls_name, img_path in tqdm(
        iter_test_images(test_root, class_names), desc="Evaluating"
    ):
        total += 1
        t0 = time.monotonic()
        prediction = pipeline.predict_image(img)
        latency_ms = (time.monotonic() - t0) * 1000
        latencies.append(latency_ms)

        hand_detected = prediction is not None

        if prediction is None:
            no_detection += 1
            records.append({
                "image_path": img_path,
                "true_label": cls_name,
                "pred_label": "Unknown",
                "confidence": 0.0,
                "top3_labels": "",
                "top3_scores": "",
                "latency_ms": round(latency_ms, 2),
                "hand_detected": False,
                "representation_status": "no_detection",
                "pipeline": args.pipeline,
            })
            continue

        prediction = apply_confidence_threshold(prediction, args.confidence_threshold)

        top3_labels = [l for l, _ in prediction.top_k[:3]]
        top3_scores = [round(s, 4) for _, s in prediction.top_k[:3]]

        is_unknown = prediction.label == "Unknown"
        if is_unknown:
            unknown_count += 1

        records.append({
            "image_path": img_path,
            "true_label": cls_name,
            "pred_label": prediction.label,
            "confidence": round(prediction.confidence, 4),
            "top3_labels": ";".join(top3_labels),
            "top3_scores": ";".join(str(s) for s in top3_scores),
            "latency_ms": round(latency_ms, 2),
            "hand_detected": True,
            "representation_status": "unknown" if is_unknown else "ok",
            "pipeline": args.pipeline,
        })

        # Save failure case
        if args.save_failures and prediction.label != cls_name:
            prefix = args.pipeline.replace("/", "_")
            failure_dir = Path(args.output) / "failure_cases" / prefix
            save_failure_case(failure_dir, img_path, cls_name, prediction.label,
                              prediction.confidence, top3_labels, hand_detected)

    # Compute metrics
    pred_labels = [r["pred_label"] for r in records if r["hand_detected"]]
    true_labels = [r["true_label"] for r in records if r["hand_detected"]]

    # For numeric metrics, map to class indices
    valid_preds = []
    valid_gts = []
    for r in records:
        if not r["hand_detected"]:
            continue
        if r["pred_label"] in class_names:
            valid_preds.append(class_names.index(r["pred_label"]))
            valid_gts.append(class_names.index(r["true_label"]))

    preds_np = np.array(valid_preds)
    gt_np = np.array(valid_gts)
    latencies_np = np.array(latencies)

    # Top-3 accuracy
    top3_correct = 0
    for r in records:
        if not r["hand_detected"]:
            continue
        top3 = r["top3_labels"].split(";") if r["top3_labels"] else []
        if r["true_label"] in top3:
            top3_correct += 1
    top3_acc = top3_correct / max(1, total - no_detection)

    # Confidence stats
    confidences = [r["confidence"] for r in records if r["hand_detected"]]
    correct_confs = [r["confidence"] for r in records
                     if r["hand_detected"] and r["pred_label"] == r["true_label"]]
    wrong_confs = [r["confidence"] for r in records
                   if r["hand_detected"] and r["pred_label"] != r["true_label"]]

    metrics = {
        "pipeline": args.pipeline,
        "dataset": str(test_root),
        "num_samples": total,
        "num_classes": len(class_names),
        "confidence_threshold": args.confidence_threshold,
        "smoothing": "none_for_image_eval",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "input_resolution": "224x224",
        "top1_accuracy": round(accuracy(preds_np, gt_np), 4) if len(preds_np) > 0 else 0.0,
        "top3_accuracy": round(top3_acc, 4),
        "macro_f1": round(macro_f1(preds_np, gt_np, num_classes=len(class_names)), 4) if len(preds_np) > 0 else 0.0,
        "weighted_f1": round(weighted_f1(preds_np, gt_np, num_classes=len(class_names)), 4) if len(preds_np) > 0 else 0.0,
        "per_class_precision": {name: round(p, 4) for name, p in
                                zip(class_names, per_class_precision(preds_np, gt_np, num_classes=len(class_names)))} if len(preds_np) > 0 else {},
        "per_class_recall": {name: round(r, 4) for name, r in
                             zip(class_names, per_class_recall(preds_np, gt_np, num_classes=len(class_names)))} if len(preds_np) > 0 else {},
        "per_class_f1": {name: round(f, 4) for name, f in
                         zip(class_names, per_class_f1(preds_np, gt_np, num_classes=len(class_names)))} if len(preds_np) > 0 else {},
        "mean_confidence": round(float(np.mean(confidences)), 4) if confidences else 0.0,
        "mean_confidence_correct": round(float(np.mean(correct_confs)), 4) if correct_confs else 0.0,
        "mean_confidence_incorrect": round(float(np.mean(wrong_confs)), 4) if wrong_confs else 0.0,
        "unknown_rate": round(unknown_count / max(1, total), 4),
        "hand_detection_failure_rate": round(no_detection / max(1, total), 4),
        "accuracy_with_unknown_as_wrong": round(
            sum(1 for r in records if r["pred_label"] == r["true_label"]) / max(1, total), 4),
        "accuracy_without_unknown": round(
            sum(1 for r in records if r["pred_label"] == r["true_label"] and r["pred_label"] != "Unknown")
            / max(1, sum(1 for r in records if r["pred_label"] != "Unknown")), 4)
            if any(r["pred_label"] != "Unknown" for r in records) else 0.0,
        "mean_latency_ms": round(float(latencies_np.mean()), 2),
        "median_latency_ms": round(float(np.median(latencies_np)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies_np, 95)), 2),
        "fps": round(1000.0 / float(latencies_np.mean()), 1) if latencies_np.mean() > 0 else 0.0,
    }

    log.info(json.dumps(metrics, indent=2))

    # Save outputs
    prefix = args.pipeline.replace("/", "_")
    metrics_dir = Path(args.output) / "metrics"
    figures_dir = Path(args.output) / "figures"
    preds_dir = Path(args.output) / "predictions"
    runs_dir = Path(args.output) / "runs" / prefix
    for d in [metrics_dir, figures_dir, preds_dir, runs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Metrics JSON
    with open(metrics_dir / f"{prefix}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Predictions CSV (full schema from eval doc §8.1)
    import csv
    csv_path = preds_dir / f"{prefix}_predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_path", "true_label", "pred_label", "confidence",
            "top3_labels", "top3_scores", "latency_ms", "hand_detected",
            "representation_status", "pipeline",
        ])
        writer.writeheader()
        writer.writerows(records)

    # Confusion matrix
    if len(preds_np) > 0:
        cm = confusion_matrix(preds_np, gt_np, num_classes=len(class_names))
        plot_confusion_matrix(cm, class_names,
                              figures_dir / f"{prefix}_confusion_matrix.png",
                              title=f"Confusion Matrix — {args.pipeline}")
        np.savetxt(metrics_dir / f"{prefix}_confusion_matrix.csv", cm, fmt="%d", delimiter=",")

    # Per-class F1 bar chart
    if metrics.get("per_class_f1"):
        _plot_per_class_f1(metrics["per_class_f1"], class_names,
                           figures_dir / f"{prefix}_per_class_f1.png", args.pipeline)

    # Run config
    run_config = {
        "pipeline": args.pipeline,
        "dataset_path": str(test_root),
        "confidence_threshold": args.confidence_threshold,
        "input_size": "224x224",
        "smoothing_window": "none",
        "device": metrics["device"],
        "random_seed": 42,
    }
    with open(runs_dir / "config.yaml", "w") as f:
        import yaml
        yaml.dump(run_config, f)

    # Environment
    save_environment(runs_dir)

    # Failure summary CSV
    if args.save_failures:
        failure_records = [r for r in records if r["pred_label"] != r["true_label"]]
        if failure_records:
            tables_dir = Path(args.output) / "tables"
            tables_dir.mkdir(parents=True, exist_ok=True)
            with open(tables_dir / f"{prefix}_failure_summary.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "image_path", "true_label", "pred_label", "confidence",
                    "top3_labels", "failure_category", "notes",
                ])
                writer.writeheader()
                for r in failure_records:
                    category = "hand_not_detected" if not r["hand_detected"] else \
                               "low_confidence" if r["confidence"] < args.confidence_threshold else \
                               "model_confident_but_wrong" if r["confidence"] > 0.8 else \
                               "similar_hand_shape"
                    writer.writerow({
                        "image_path": r["image_path"],
                        "true_label": r["true_label"],
                        "pred_label": r["pred_label"],
                        "confidence": r["confidence"],
                        "top3_labels": r["top3_labels"],
                        "failure_category": category,
                        "notes": "",
                    })

    log.info(f"Results saved to {args.output}/")
    pipeline.close()


def _plot_per_class_f1(per_class_f1_dict: dict, class_names: list,
                        output_path: Path, pipeline_name: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f1_values = [per_class_f1_dict.get(c, 0.0) for c in class_names]
    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(class_names, f1_values, color="steelblue")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Class")
    ax.set_ylabel("F1 Score")
    ax.set_title(f"Per-class F1 — {pipeline_name}")
    ax.axhline(y=np.mean(f1_values), color="red", linestyle="--", label=f"Macro F1: {np.mean(f1_values):.3f}")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
