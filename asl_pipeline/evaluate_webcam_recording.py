"""Evaluate a pipeline on a recorded webcam video.

Per eval doc §8.3:
  - frame_accuracy, majority_vote_accuracy, label_switch_rate
  - unknown_rate, mean_confidence, mean_latency_ms, fps
  - stable_prediction_delay

Usage:
    python evaluate_webcam_recording.py --video data/webcam_videos/A_01.mp4 --label A --pipeline mediapipe_resnet18
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from asl.pipelines.registry import build_pipeline, register_defaults
from asl.postprocess.threshold import apply_confidence_threshold
from asl.utils.logging import get_logger
from asl.utils.smoothing import PredictionSmoother

log = get_logger("asl.webcam_eval")


def main():
    parser = argparse.ArgumentParser(description="Evaluate pipeline on webcam recording")
    parser.add_argument("--video", required=True, help="Path to recorded video")
    parser.add_argument("--label", required=True, help="Ground-truth label for this video")
    parser.add_argument("--pipeline", default="mediapipe_resnet18")
    parser.add_argument("--output", default="outputs/")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--smoothing-window", type=int, default=7)
    args = parser.parse_args()

    register_defaults()

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    smoother = PredictionSmoother(smoothing_window=args.smoothing_window)
    pipeline = build_pipeline(args.pipeline, smoother=smoother)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")

    records = []
    frame_idx = 0
    prev_raw_pred = None
    label_switches = 0
    unknown_frames = 0
    correct_raw = 0
    correct_smoothed = 0
    latencies = []
    confidences = []
    stable_delay_frames = []
    current_stable_count = 0
    found_stable = False

    true_label = args.label.upper()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t0 = time.monotonic()
        out = pipeline.predict_frame(frame_rgb)
        latency = (time.monotonic() - t0) * 1000
        latencies.append(latency)

        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        raw_pred = None
        confidence = 0.0
        hand_detected = False

        if out.prediction:
            hand_detected = True
            pred = apply_confidence_threshold(out.prediction, args.confidence_threshold)
            raw_pred = pred.label
            confidence = pred.confidence
            confidences.append(confidence)

            if raw_pred == true_label:
                correct_raw += 1
            if raw_pred == "Unknown":
                unknown_frames += 1

            if prev_raw_pred is not None and raw_pred != prev_raw_pred:
                label_switches += 1
            prev_raw_pred = raw_pred
        else:
            unknown_frames += 1

        # Track stable prediction delay
        smoothed = out.accepted_letter
        if smoothed == true_label and not found_stable:
            found_stable = True
            stable_delay_frames.append(frame_idx)

        if smoothed == true_label:
            correct_smoothed += 1

        records.append({
            "frame_index": frame_idx,
            "timestamp_ms": round(timestamp_ms, 1),
            "true_label": true_label,
            "raw_pred": raw_pred or "Unknown",
            "smoothed_pred": smoothed or "",
            "confidence": round(confidence, 4),
            "hand_detected": hand_detected,
            "latency_ms": round(latency, 2),
        })

        frame_idx += 1

    cap.release()

    if frame_idx == 0:
        log.warning("No frames processed")
        return

    # Compute metrics
    metrics = {
        "video": str(video_path),
        "true_label": true_label,
        "pipeline": args.pipeline,
        "total_frames": frame_idx,
        "frame_accuracy": round(correct_raw / frame_idx, 4),
        "majority_vote_accuracy": round(correct_smoothed / max(1, frame_idx), 4),
        "label_switch_rate": round(label_switches / max(1, frame_idx), 4),
        "unknown_rate": round(unknown_frames / frame_idx, 4),
        "mean_confidence": round(float(np.mean(confidences)), 4) if confidences else 0.0,
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "fps": round(1000.0 / float(np.mean(latencies)), 1) if np.mean(latencies) > 0 else 0.0,
        "stable_prediction_delay": stable_delay_frames[0] if stable_delay_frames else -1,
    }

    log.info(json.dumps(metrics, indent=2))

    # Save outputs
    prefix = args.pipeline.replace("/", "_")
    video_name = video_path.stem
    metrics_dir = Path(args.output) / "metrics"
    preds_dir = Path(args.output) / "predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_dir / f"webcam_{video_name}_{prefix}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    csv_path = preds_dir / f"webcam_{video_name}_{prefix}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame_index", "timestamp_ms", "true_label", "raw_pred",
            "smoothed_pred", "confidence", "hand_detected", "latency_ms",
        ])
        writer.writeheader()
        writer.writerows(records)

    # Prediction timeline chart
    _plot_timeline(records, true_label,
                   Path(args.output) / "figures" / f"webcam_{video_name}_{prefix}_timeline.png",
                   args.pipeline)

    log.info(f"Results saved to {args.output}/")
    pipeline.close()


def _plot_timeline(records: list, true_label: str, output_path: Path, pipeline_name: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = [r["frame_index"] for r in records]
    raw_preds = [r["raw_pred"] for r in records]
    confs = [r["confidence"] for r in records]

    all_labels = sorted(set(raw_preds))
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    raw_indices = [label_to_idx[p] for p in raw_preds]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    ax1.scatter(frames, raw_indices, c=["green" if p == true_label else "red" for p in raw_preds],
                s=4, alpha=0.6)
    ax1.set_yticks(range(len(all_labels)))
    ax1.set_yticklabels(all_labels, fontsize=7)
    ax1.set_ylabel("Predicted Label")
    ax1.set_title(f"Webcam Prediction Timeline — {pipeline_name} (true: {true_label})")

    ax2.plot(frames, confs, color="steelblue", linewidth=0.5)
    ax2.set_ylabel("Confidence")
    ax2.set_xlabel("Frame")
    ax2.set_ylim(0, 1.05)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
