"""Visualize each pipeline step (representation -> recognizer input -> prediction).

For a professor/report audience: don't just show final numbers, show what the
image actually looks like after every transformation, side by side with the
predicted label/confidence and whether it was correct.

Usage:
    # Per-pipeline gallery: 1 correct + 1 wrong case each, picked from
    # outputs/predictions/<pipeline>_predictions.csv
    python visualize_steps.py --pipelines raw_resnet18 mediapipe_crop_resnet18 \
        mediapipe_landmarks_svm enhancement_clahe_resnet18 mediapipe_crop_vit

    # Compare several pipelines on one specific image
    python visualize_steps.py --image data/asl_alphabet/test_split/A/A1.jpg \
        --pipelines raw_resnet18 mediapipe_crop_resnet18 mediapipe_landmarks_svm
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from asl.pipelines.registry import build_pipeline, register_defaults
from asl.representations.base import RepresentationOutput
from asl.recognizers.base import Prediction

DEFAULT_PIPELINES = [
    "raw_resnet18",
    "mediapipe_crop_resnet18",
    "mediapipe_landmarks_svm",
    "enhancement_clahe_resnet18",
    "mediapipe_crop_vit",
]

PRED_DIR = Path("outputs/predictions")
OUT_DIR = Path("outputs/visualizations")


def load_rgb(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def pick_cases(pipeline_name: str, seed: int = 0) -> tuple[Optional[dict], Optional[dict]]:
    """Return (correct_row, wrong_row) sampled from this pipeline's predictions.csv."""
    csv_path = PRED_DIR / f"{pipeline_name}_predictions.csv"
    if not csv_path.exists():
        return None, None
    rng = random.Random(seed)
    correct, wrong = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("pred_label") in ("", "Unknown"):
                continue
            if row["pred_label"] == row["true_label"]:
                correct.append(row)
            else:
                wrong.append(row)
    return (rng.choice(correct) if correct else None,
            rng.choice(wrong) if wrong else None)


def representation_display(rep_output: RepresentationOutput, raw_rgb: np.ndarray) -> tuple[np.ndarray, str]:
    """Return an image array to display for the 'recognizer input' panel + a caption."""
    if rep_output.output_type == "image":
        return rep_output.data, "input ảnh cho recognizer"
    if rep_output.output_type == "features":
        # render the 42-dim normalized landmark vector as a bar plot image
        fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
        feats = np.asarray(rep_output.data)
        ax.bar(range(len(feats)), feats, color="steelblue", width=1.0)
        ax.set_title("landmark features (42-d)", fontsize=8)
        ax.tick_params(labelsize=6)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return buf, "landmark vector (input cho MLP/SVM/RF)"
    return raw_rgb, "input"


def draw_overlay(representation, raw_rgb: np.ndarray, rep_output: RepresentationOutput) -> np.ndarray:
    overlay = raw_rgb.copy()
    try:
        overlay = representation.visualize(overlay, rep_output)
    except Exception:
        pass
    return overlay


def render_row(fig, axes_row, pipeline_name: str, image_path: Path, true_label: Optional[str]):
    """Build one row: [original | overlay | recognizer input | top-3 bar] for one pipeline+image."""
    register_defaults()
    pipeline = build_pipeline(pipeline_name)
    raw_rgb = load_rgb(image_path)

    rep_output = pipeline.representation.process(raw_rgb)
    ax_raw, ax_overlay, ax_input, ax_bar = axes_row

    ax_raw.imshow(raw_rgb)
    ax_raw.set_title(f"{pipeline_name}\nảnh gốc", fontsize=9)
    ax_raw.axis("off")

    if rep_output is None:
        ax_overlay.imshow(raw_rgb)
        ax_overlay.set_title("không phát hiện tay", fontsize=9, color="red")
        ax_overlay.axis("off")
        ax_input.axis("off")
        ax_bar.axis("off")
        ax_bar.text(0.5, 0.5, "Unknown\n(no hand detected)", ha="center", va="center",
                     color="red", fontsize=10, transform=ax_bar.transAxes)
        pipeline.close()
        return

    overlay = draw_overlay(pipeline.representation, raw_rgb, rep_output)
    ax_overlay.imshow(overlay)
    ax_overlay.set_title("representation (bbox/landmarks)", fontsize=9)
    ax_overlay.axis("off")

    input_img, caption = representation_display(rep_output, raw_rgb)
    ax_input.imshow(input_img)
    ax_input.set_title(caption, fontsize=9)
    ax_input.axis("off")

    prediction: Prediction = pipeline.recognizer.predict(rep_output)
    labels = [l for l, _ in prediction.top_k[:3]]
    scores = [s for _, s in prediction.top_k[:3]]
    is_correct = true_label is not None and prediction.label == true_label
    colors = ["seagreen" if l == prediction.label else "lightgray" for l in labels]
    if true_label is not None and prediction.label != true_label:
        colors = ["crimson" if l == prediction.label else "lightgray" for l in labels]

    ax_bar.barh(labels[::-1], [s * 100 for s in scores[::-1]], color=colors[::-1])
    ax_bar.set_xlim(0, 100)
    title = f"pred={prediction.label} ({prediction.confidence * 100:.1f}%)"
    if true_label is not None:
        title += f"\ntrue={true_label} [{'ĐÚNG' if is_correct else 'SAI'}]"
    ax_bar.set_title(title, fontsize=9,
                      color=("seagreen" if is_correct else "crimson") if true_label else "black")
    ax_bar.tick_params(labelsize=8)

    pipeline.close()


def make_figure(rows: list[tuple[str, Path, Optional[str]]], out_path: Path, suptitle: str):
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.4 * n))
    if n == 1:
        axes = axes.reshape(1, 4)
    for i, (pipeline_name, image_path, true_label) in enumerate(rows):
        print(f"  -> {pipeline_name} :: {image_path}")
        render_row(fig, axes[i], pipeline_name, image_path, true_label)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def cmd_per_pipeline_gallery(pipelines: list[str], seed: int):
    for name in pipelines:
        correct_row, wrong_row = pick_cases(name, seed=seed)
        for case_label, row in (("correct", correct_row), ("wrong", wrong_row)):
            if row is None:
                print(f"[{name}] no {case_label} example found in predictions.csv, skipping")
                continue
            image_path = Path(row["image_path"])
            true_label = row["true_label"]
            out_path = OUT_DIR / f"steps_{name}_{case_label}_{true_label}.png"
            suptitle = (f"{name} — {case_label.upper()} case — true={true_label} "
                        f"pred={row['pred_label']} conf={float(row['confidence']):.2f}")
            make_figure([(name, image_path, true_label)], out_path, suptitle)


def cmd_compare_single_image(image: Path, pipelines: list[str], true_label: Optional[str]):
    rows = [(name, image, true_label) for name in pipelines]
    out_path = OUT_DIR / f"compare_{image.stem}.png"
    make_figure(rows, out_path, f"So sánh {len(pipelines)} pipeline trên ảnh: {image.name}")


def main():
    parser = argparse.ArgumentParser(description="Visualize per-step pipeline outputs")
    parser.add_argument("--pipelines", nargs="+", default=DEFAULT_PIPELINES)
    parser.add_argument("--image", default=None,
                         help="If set, compare all --pipelines on this one image instead of "
                              "sampling correct/wrong cases per pipeline")
    parser.add_argument("--true-label", default=None,
                         help="True label for --image, inferred from parent dir name if omitted")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.image:
        image_path = Path(args.image)
        true_label = args.true_label or image_path.parent.name
        cmd_compare_single_image(image_path, args.pipelines, true_label)
    else:
        cmd_per_pipeline_gallery(args.pipelines, args.seed)


if __name__ == "__main__":
    main()
