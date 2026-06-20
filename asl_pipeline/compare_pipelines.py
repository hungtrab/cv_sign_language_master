"""Compare evaluation results across pipelines.

Reads metrics JSON files from outputs/metrics/ and generates comparison tables and charts.

Outputs per eval doc §8.4:
  - pipeline_accuracy_comparison.png
  - pipeline_latency_comparison.png
  - pipeline_robustness_comparison.png
  - pipeline_summary.csv

Usage:
    python compare_pipelines.py --metrics-dir outputs/metrics --output-dir outputs/figures
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from asl.utils.logging import get_logger

log = get_logger("asl.compare")


def load_metrics(metrics_dir: Path) -> dict:
    results = {}
    for f in sorted(metrics_dir.glob("*_metrics.json")):
        with open(f) as fh:
            data = json.load(fh)
            name = data.get("pipeline", f.stem.replace("_metrics", ""))
            results[name] = data
    return results


def load_robustness(metrics_dir: Path) -> dict:
    results = {}
    for f in sorted(metrics_dir.glob("*_robustness.json")):
        with open(f) as fh:
            data = json.load(fh)
            pipeline = f.stem.replace("_robustness", "")
            results[pipeline] = data
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare pipeline results")
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    parser.add_argument("--output-dir", default="outputs/figures")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    output_dir = Path(args.output_dir)
    tables_dir = output_dir.parent / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Load all metrics
    metrics = load_metrics(metrics_dir)
    if not metrics:
        log.warning(f"No *_metrics.json files found in {metrics_dir}")
        return

    log.info(f"Found {len(metrics)} pipeline results")

    # Summary CSV (eval doc §8.4)
    csv_path = tables_dir / "pipeline_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pipeline", "clean_accuracy", "macro_f1",
            "worst_corruption_accuracy", "average_corrupted_accuracy",
            "average_accuracy_drop", "fps", "mean_latency_ms",
            "hand_detection_failure_rate", "notes",
        ])
        robustness = load_robustness(metrics_dir)
        for name, m in metrics.items():
            rob = robustness.get(name, {})
            corrupted_accs = [v["accuracy"] for k, v in rob.items()
                              if k != "clean" and isinstance(v, dict) and "accuracy" in v]
            drops = [v.get("accuracy_drop", 0) for k, v in rob.items()
                     if k != "clean" and isinstance(v, dict)]
            writer.writerow([
                name,
                m.get("top1_accuracy", 0),
                m.get("macro_f1", 0),
                min(corrupted_accs) if corrupted_accs else "",
                round(sum(corrupted_accs) / len(corrupted_accs), 4) if corrupted_accs else "",
                round(sum(drops) / len(drops), 4) if drops else "",
                m.get("fps", 0),
                m.get("mean_latency_ms", 0),
                m.get("hand_detection_failure_rate", 0),
                "",
            ])
    log.info(f"Summary table -> {csv_path}")

    # Accuracy comparison bar chart
    _plot_comparison(
        metrics,
        key="top1_accuracy",
        ylabel="Top-1 Accuracy",
        title="Pipeline Accuracy Comparison",
        output_path=output_dir / "pipeline_accuracy_comparison.png",
    )

    # Latency comparison bar chart
    _plot_comparison(
        metrics,
        key="mean_latency_ms",
        ylabel="Mean Latency (ms)",
        title="Pipeline Latency Comparison",
        output_path=output_dir / "pipeline_latency_comparison.png",
    )

    # Robustness comparison (average corrupted accuracy)
    if robustness:
        rob_summary = {}
        for name, rob in robustness.items():
            corrupted_accs = [v["accuracy"] for k, v in rob.items()
                              if k != "clean" and isinstance(v, dict) and "accuracy" in v]
            if corrupted_accs:
                rob_summary[name] = {
                    "clean": rob.get("clean", {}).get("accuracy", 0),
                    "avg_corrupted": round(sum(corrupted_accs) / len(corrupted_accs), 4),
                }
        if rob_summary:
            _plot_robustness_comparison(rob_summary,
                                         output_dir / "pipeline_robustness_comparison.png")

    # Print summary
    print(f"\n{'Pipeline':<30} {'Accuracy':>10} {'F1':>10} {'FPS':>8} {'Latency':>10}")
    print("-" * 70)
    for name, m in metrics.items():
        print(f"{name:<30} {m.get('top1_accuracy', 0):>10.4f} "
              f"{m.get('macro_f1', 0):>10.4f} {m.get('fps', 0):>8.1f} "
              f"{m.get('mean_latency_ms', 0):>10.1f}ms")

    log.info("Done.")


def _plot_comparison(metrics: dict, key: str, ylabel: str, title: str, output_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(metrics.keys())
    values = [metrics[n].get(key, 0) for n in names]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 2), 5))
    ax.bar(names, values, color="steelblue")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    log.info(f"Chart -> {output_path}")


def _plot_robustness_comparison(rob_summary: dict, output_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(rob_summary.keys())
    clean = [rob_summary[n]["clean"] for n in names]
    corrupted = [rob_summary[n]["avg_corrupted"] for n in names]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 2), 5))
    ax.bar(x - width / 2, clean, width, label="Clean", color="green")
    ax.bar(x + width / 2, corrupted, width, label="Avg Corrupted", color="orange")
    ax.set_ylabel("Accuracy")
    ax.set_title("Pipeline Robustness Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    log.info(f"Chart -> {output_path}")


if __name__ == "__main__":
    main()
