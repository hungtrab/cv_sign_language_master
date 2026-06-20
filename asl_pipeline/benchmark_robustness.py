"""Benchmark pipeline robustness under image corruptions.

Outputs per eval doc §8.2:
  - robustness_summary.csv (pipeline, corruption, severity, accuracy, macro_f1, drops, etc.)
  - robustness_accuracy_bar_chart.png
  - robustness_accuracy_drop_chart.png

Usage:
    python benchmark_robustness.py --dataset data/asl_alphabet/test --pipeline mediapipe_resnet18
    python benchmark_robustness.py --dataset data/test --pipeline raw_hf --corruptions low_light,motion_blur
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from asl.pipelines.registry import build_pipeline, register_defaults
from asl.evaluation.metrics import accuracy, macro_f1
from asl.evaluation.robustness import ALL_CORRUPTIONS
from asl.postprocess.threshold import apply_confidence_threshold
from asl.utils.config import load_class_names
from asl.utils.logging import get_logger

log = get_logger("asl.robustness")

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
            yield img, cls_idx, name


def evaluate_on_images(pipeline, images, class_names, conf_threshold: float = 0.5):
    preds, gts = [], []
    no_det = 0
    unknown = 0
    confidences = []
    latencies = []
    for img, cls_idx, _ in images:
        t0 = time.monotonic()
        pred = pipeline.predict_image(img)
        latency = (time.monotonic() - t0) * 1000
        latencies.append(latency)
        if pred is None:
            no_det += 1
            continue
        pred = apply_confidence_threshold(pred, conf_threshold)
        confidences.append(pred.confidence)
        if pred.label == "Unknown":
            unknown += 1
            continue
        pred_idx = class_names.index(pred.label) if pred.label in class_names else -1
        if pred_idx >= 0:
            preds.append(pred_idx)
            gts.append(cls_idx)

    total = len(images) if isinstance(images, list) else sum(1 for _ in images)
    if not preds:
        return {
            "accuracy": 0.0, "macro_f1": 0.0, "detected": 0,
            "no_detection": no_det, "unknown": unknown,
            "mean_confidence": 0.0, "unknown_rate": 0.0,
            "hand_detection_failure_rate": 0.0,
            "mean_latency_ms": 0.0, "fps": 0.0,
            "num_samples": 0,
        }
    preds_np = np.array(preds)
    gts_np = np.array(gts)
    latencies_np = np.array(latencies)
    return {
        "accuracy": round(accuracy(preds_np, gts_np), 4),
        "macro_f1": round(macro_f1(preds_np, gts_np, num_classes=len(class_names)), 4),
        "detected": len(preds),
        "no_detection": no_det,
        "unknown": unknown,
        "mean_confidence": round(float(np.mean(confidences)), 4) if confidences else 0.0,
        "unknown_rate": round(unknown / max(1, len(latencies)), 4),
        "hand_detection_failure_rate": round(no_det / max(1, len(latencies)), 4),
        "mean_latency_ms": round(float(latencies_np.mean()), 2),
        "fps": round(1000.0 / float(latencies_np.mean()), 1) if latencies_np.mean() > 0 else 0.0,
        "num_samples": len(latencies),
    }


def main():
    parser = argparse.ArgumentParser(description="Robustness benchmark")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pipeline", default="mediapipe_resnet18")
    parser.add_argument("--corruptions", default="all",
                        help="Comma-separated corruption names, or 'all'")
    parser.add_argument("--output", default="outputs/")
    parser.add_argument("--class-names", default="configs/asl_classes_az.txt")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    args = parser.parse_args()

    register_defaults()

    test_root = Path(args.dataset)
    if not test_root.exists():
        raise SystemExit(f"Test dir not found: {test_root}")

    class_names = AZ_CLASSES
    if Path(args.class_names).exists():
        class_names = load_class_names(args.class_names)

    if args.corruptions == "all":
        corruptions = list(ALL_CORRUPTIONS.keys())
    else:
        corruptions = [c.strip() for c in args.corruptions.split(",")]

    pipeline = build_pipeline(args.pipeline)
    log.info(f"Pipeline: {args.pipeline}")
    log.info(f"Corruptions: {corruptions}")

    all_images = list(iter_test_images(test_root, class_names))
    log.info(f"Loaded {len(all_images)} test images")

    results = {}

    # Clean baseline
    log.info("Evaluating: clean")
    results["clean"] = evaluate_on_images(pipeline, all_images, class_names,
                                           args.confidence_threshold)
    results["clean"]["severity"] = "none"
    clean_acc = results["clean"]["accuracy"]
    log.info(f"  clean: acc={clean_acc}")

    # Corrupted
    for corruption_name in corruptions:
        if corruption_name not in ALL_CORRUPTIONS:
            log.warning(f"Unknown corruption: {corruption_name}, skipping")
            continue
        corrupt_fn = ALL_CORRUPTIONS[corruption_name]
        log.info(f"Evaluating: {corruption_name}")
        corrupted_images = [(corrupt_fn(img), idx, name) for img, idx, name in all_images]
        r = evaluate_on_images(pipeline, corrupted_images, class_names,
                                args.confidence_threshold)
        r["severity"] = "default"
        r["accuracy_drop"] = round(clean_acc - r["accuracy"], 4)
        r["relative_accuracy_drop"] = round(
            (clean_acc - r["accuracy"]) / max(clean_acc, 1e-6), 4)
        results[corruption_name] = r
        log.info(f"  {corruption_name}: acc={r['accuracy']}, drop={r['accuracy_drop']}")

    # Save outputs
    prefix = args.pipeline.replace("/", "_")
    metrics_dir = Path(args.output) / "metrics"
    figures_dir = Path(args.output) / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(metrics_dir / f"{prefix}_robustness.json", "w") as f:
        json.dump(results, f, indent=2)

    # CSV with all required columns (eval doc §7.3) — append mode so multiple pipelines accumulate
    csv_path = metrics_dir / "robustness_summary.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
            "pipeline", "corruption", "severity", "accuracy", "macro_f1",
            "accuracy_drop", "relative_accuracy_drop", "mean_confidence",
            "unknown_rate", "hand_detection_failure_rate",
            "mean_latency_ms", "fps", "num_samples",
        ])
        for name, m in results.items():
            writer.writerow([
                args.pipeline, name, m.get("severity", ""),
                m["accuracy"], m["macro_f1"],
                m.get("accuracy_drop", 0.0), m.get("relative_accuracy_drop", 0.0),
                m["mean_confidence"], m["unknown_rate"],
                m["hand_detection_failure_rate"],
                m["mean_latency_ms"], m["fps"], m["num_samples"],
            ])

    # Bar chart: accuracy per corruption
    _plot_accuracy_bar(results, figures_dir / f"{prefix}_robustness_accuracy_bar_chart.png",
                       args.pipeline)

    # Accuracy drop chart
    _plot_accuracy_drop(results, figures_dir / f"{prefix}_robustness_accuracy_drop_chart.png",
                         args.pipeline)

    log.info(f"Results saved to {args.output}/")

    # Print summary table
    print(f"\n{'Corruption':<20} {'Accuracy':>10} {'Macro-F1':>10} {'Drop':>10} {'FPS':>8}")
    print("-" * 60)
    for name, m in results.items():
        drop = m.get("accuracy_drop", 0.0)
        print(f"{name:<20} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f} {drop:>10.4f} {m['fps']:>8.1f}")

    pipeline.close()


def _plot_accuracy_bar(results: dict, output_path: Path, pipeline_name: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["green" if n == "clean" else "steelblue" for n in names]
    ax.bar(names, accs, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Robustness — {pipeline_name}")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


def _plot_accuracy_drop(results: dict, output_path: Path, pipeline_name: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    corrupted = {k: v for k, v in results.items() if k != "clean"}
    if not corrupted:
        return

    names = list(corrupted.keys())
    drops = [corrupted[n].get("accuracy_drop", 0.0) for n in names]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["red" if d > 0.1 else "orange" if d > 0.05 else "green" for d in drops]
    ax.bar(names, drops, color=colors)
    ax.set_ylabel("Accuracy Drop")
    ax.set_title(f"Accuracy Drop Under Corruption — {pipeline_name}")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
