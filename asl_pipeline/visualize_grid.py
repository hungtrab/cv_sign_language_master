"""Report-style qualitative grids, matching the look the professor liked
(rows = sample images, columns = methods, each cell = image + small overlay
text — no extra side-panels/bar charts cluttering the figure).

Grid 1 — Representation module: shows what each representation does to the
same set of raw images (raw resize / mediapipe crop / enhancement variants).

Grid 2 — Recognizer module (full pipeline): shows the final prediction of
several complete pipelines on the same images, with pred/confidence overlaid
and colored green (correct) / red (wrong), the same way the reference slides
overlay PSNR/SSIM or bbox+confidence on each cell.

Usage:
    python visualize_grid.py --grid representation --n-samples 5
    python visualize_grid.py --grid recognizer --n-samples 5 \
        --pipelines raw_resnet18 mediapipe_crop_resnet18 mediapipe_crop_vit \
                    enhancement_clahe_resnet18 raw_siglip
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

OUT_DIR = Path("outputs/visualizations")
PRED_DIR = Path("outputs/predictions")

REPR_COLUMNS = [
    ("raw_image", {}, "Raw"),
    ("mediapipe_crop", {}, "MediaPipe Crop"),
    ("mediapipe_landmarks", {}, "MediaPipe\nLandmarks"),
    ("enhancement", {"method": "clahe"}, "CLAHE"),
    ("enhancement", {"method": "gamma"}, "Gamma Corr."),
    ("enhancement", {"method": "sharpening"}, "Sharpening"),
    ("enhancement", {"method": "denoising"}, "Denoising"),
]

REPR_CLASS_MAP = {
    "raw_image": "asl.representations.raw_image.RawImageRepresentation",
    "mediapipe_crop": "asl.representations.mediapipe_crop.MediaPipeCropRepresentation",
    "mediapipe_landmarks": "asl.representations.mediapipe_landmarks.MediaPipeLandmarksRepresentation",
    "enhancement": "asl.representations.enhancement.EnhancementRepresentation",
}

# All 13 unique pipelines (duplicates already removed from the registry/outputs)
DEFAULT_RECOGNIZER_PIPELINES = [
    "raw_resnet18",
    "raw_siglip",
    "mediapipe_crop_resnet18",
    "mediapipe_crop_vit",
    "mediapipe_landmarks_mlp",
    "mediapipe_landmarks_svm",
    "mediapipe_landmarks_rf",
    "enhancement_clahe_resnet18",
    "enhancement_gamma_resnet18",
    "enhancement_sharpen_resnet18",
    "enhancement_denoise_resnet18",
    "enhancement_clahe_vit",
    "enhancement_gamma_vit",
]


def _import_class(dotted: str):
    import importlib
    module_path, class_name = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


def load_rgb(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def sample_images(n: int, seed: int) -> list[Path]:
    """Pick n images, one per distinct class, from the test set."""
    test_root = Path("data/asl_alphabet/test_split")
    classes = sorted(p.name for p in test_root.iterdir() if p.is_dir())
    rng = random.Random(seed)
    chosen_classes = rng.sample(classes, min(n, len(classes)))
    paths = []
    for c in chosen_classes:
        imgs = sorted((test_root / c).glob("*.jpg"))
        paths.append(rng.choice(imgs))
    return paths


def put_overlay_text(ax, text: str, color: str = "white", bg: str = "black"):
    ax.text(0.02, 0.96, text, transform=ax.transAxes, fontsize=7, color=color,
             va="top", ha="left", family="monospace",
             bbox=dict(facecolor=bg, alpha=0.65, pad=2, edgecolor="none"))


def build_representation_grid(image_paths: list[Path], out_path: Path):
    register_defaults()
    n_rows, n_cols = len(image_paths), len(REPR_COLUMNS) + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.1 * n_cols, 2.1 * n_rows))

    reps = []
    for repr_type, kwargs, _ in REPR_COLUMNS:
        cls = _import_class(REPR_CLASS_MAP[repr_type])
        reps.append(cls(**kwargs))

    for col, (_, _, label) in enumerate([("", {}, "Input")] + REPR_COLUMNS):
        axes[0, col].set_title(label, fontsize=10, fontweight="bold")

    for row, img_path in enumerate(image_paths):
        raw_rgb = load_rgb(img_path)
        axes[row, 0].imshow(cv2.resize(raw_rgb, (224, 224)))
        axes[row, 0].axis("off")
        put_overlay_text(axes[row, 0], img_path.parent.name, color="yellow")

        for col, rep in enumerate(reps, start=1):
            out = rep.process(raw_rgb)
            ax = axes[row, col]
            if out is None:
                ax.imshow(np.zeros((224, 224, 3), dtype=np.uint8))
                ax.axis("off")
                put_overlay_text(ax, "no hand\ndetected", color="red")
                continue
            if out.output_type == "image":
                ax.imshow(out.data)
                ax.axis("off")
            else:
                ax.imshow(np.zeros((224, 224, 3), dtype=np.uint8))
                ax.axis("off")
                put_overlay_text(ax, "(feature vector,\nnot an image)", color="cyan")

    for rep in reps:
        rep.close()

    fig.suptitle("Representation Module — cùng input, các phép biến đổi khác nhau", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def _wrap_title(name: str, width: int = 12) -> str:
    parts = name.split("_")
    lines, cur = [], ""
    for p in parts:
        if cur and len(cur) + 1 + len(p) > width:
            lines.append(cur)
            cur = p
        else:
            cur = f"{cur}_{p}" if cur else p
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def build_recognizer_grid(image_paths: list[Path], pipelines: list[str], out_path: Path):
    register_defaults()
    n_rows, n_cols = len(image_paths), len(pipelines) + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.7 * n_cols, 1.9 * n_rows))

    axes[0, 0].set_title("Input\n(true label)", fontsize=9, fontweight="bold")
    for col, name in enumerate(pipelines, start=1):
        axes[0, col].set_title(_wrap_title(name), fontsize=7.5, fontweight="bold")

    built_pipelines = {name: build_pipeline(name) for name in pipelines}

    for row, img_path in enumerate(image_paths):
        true_label = img_path.parent.name
        raw_rgb = load_rgb(img_path)
        axes[row, 0].imshow(cv2.resize(raw_rgb, (224, 224)))
        axes[row, 0].axis("off")
        put_overlay_text(axes[row, 0], f"true={true_label}", color="yellow")

        for col, name in enumerate(pipelines, start=1):
            pipeline = built_pipelines[name]
            ax = axes[row, col]
            rep_output = pipeline.representation.process(raw_rgb)
            if rep_output is None:
                ax.imshow(cv2.resize(raw_rgb, (224, 224)))
                ax.axis("off")
                put_overlay_text(ax, "Unknown\n(no hand detected)", color="red")
                continue
            prediction = pipeline.recognizer.predict(rep_output)
            display_img = rep_output.data if rep_output.output_type == "image" else raw_rgb
            ax.imshow(cv2.resize(np.asarray(display_img), (224, 224)))
            ax.axis("off")
            is_correct = prediction.label == true_label
            color = "lime" if is_correct else "red"
            put_overlay_text(ax, f"pred={prediction.label} ({prediction.confidence*100:.0f}%)", color=color)

    for pipeline in built_pipelines.values():
        pipeline.close()

    fig.suptitle("Recognizer Module — cùng input, kết quả của từng pipeline hoàn chỉnh", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", choices=["representation", "recognizer"], required=True)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pipelines", nargs="+", default=DEFAULT_RECOGNIZER_PIPELINES)
    args = parser.parse_args()

    image_paths = sample_images(args.n_samples, args.seed)
    print("Sampled images:", [str(p) for p in image_paths])

    if args.grid == "representation":
        build_representation_grid(image_paths, OUT_DIR / "grid_representation.png")
    else:
        build_recognizer_grid(image_paths, args.pipelines, OUT_DIR / "grid_recognizer.png")


if __name__ == "__main__":
    main()
