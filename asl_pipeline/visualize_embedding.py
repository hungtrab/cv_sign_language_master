"""UMAP / t-SNE embedding plots, the style used in the reference slide
(side-by-side UMAP | t-SNE scatter, colored by a category) — applied here to:

  - landmark   : 42-d MediaPipe landmark vectors, colored by class (A-Z)
  - cnn        : 512-d ResNet18 avgpool embeddings, colored by class (A-Z)
  - repshift   : 512-d ResNet18 avgpool embeddings of the SAME images under
                 different representations (raw/crop/clahe/gamma), colored by
                 representation — the direct analog of the reference slide's
                 "does preprocessing shift the feature space" plot.

Usage:
    python visualize_embedding.py --mode landmark --per-class 30
    python visualize_embedding.py --mode cnn --per-class 30
    python visualize_embedding.py --mode repshift --per-class 15
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.manifold import TSNE
import umap

from asl.pipelines.registry import build_pipeline, register_defaults
from asl.representations.mediapipe_landmarks import MediaPipeLandmarksRepresentation
from asl.representations.mediapipe_crop import MediaPipeCropRepresentation
from asl.representations.enhancement import EnhancementRepresentation
from asl.representations.raw_image import RawImageRepresentation

TEST_ROOT = Path("data/asl_alphabet/test_split")
OUT_DIR = Path("outputs/visualizations")


def load_rgb(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path))
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def sample_per_class(per_class: int, seed: int = 0) -> list[Path]:
    rng = random.Random(seed)
    classes = sorted(p.name for p in TEST_ROOT.iterdir() if p.is_dir())
    paths = []
    for c in classes:
        imgs = sorted((TEST_ROOT / c).glob("*.jpg"))
        paths.extend(rng.sample(imgs, min(per_class, len(imgs))))
    return paths


def class_color_map(labels: list[str]) -> dict:
    classes = sorted(set(labels))
    cmap = plt.get_cmap("tab20")
    cmap2 = plt.get_cmap("tab20b")
    colors = [cmap(i / 20) for i in range(20)] + [cmap2(i / 20) for i in range(20)]
    return {c: colors[i % len(colors)] for i, c in enumerate(classes)}


def plot_umap_tsne(feats: np.ndarray, labels: list[str], title: str, out_path: Path,
                    legend_title: str = "Class"):
    print(f"Running UMAP + t-SNE on {feats.shape[0]} points x {feats.shape[1]} dims...")
    umap_emb = umap.UMAP(n_components=2, random_state=0).fit_transform(feats)
    tsne_emb = TSNE(n_components=2, random_state=0, init="pca",
                     perplexity=min(30, max(5, feats.shape[0] // 20))).fit_transform(feats)

    color_map = class_color_map(labels)
    point_colors = [color_map[l] for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, emb, name in [(axes[0], umap_emb, "UMAP"), (axes[1], tsne_emb, "t-SNE")]:
        ax.scatter(emb[:, 0], emb[:, 1], c=point_colors, s=14, alpha=0.8)
        ax.set_title(name, fontsize=13)
        ax.set_xlabel(f"{name} Dimension 1")
        ax.set_ylabel(f"{name} Dimension 2")

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[c],
                           markersize=7, label=c) for c in sorted(color_map)]
    fig.legend(handles=handles, title=legend_title, loc="center right",
               bbox_to_anchor=(1.04, 0.5), fontsize=7, ncol=1)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 0.92, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def mode_landmark(per_class: int, seed: int):
    rep = MediaPipeLandmarksRepresentation()
    feats, labels = [], []
    for path in sample_per_class(per_class, seed):
        out = rep.process(load_rgb(path))
        if out is None:
            continue
        feats.append(out.data)
        labels.append(path.parent.name)
    rep.close()
    print(f"Collected {len(feats)} landmark vectors "
          f"({len(sample_per_class(per_class, seed)) - len(feats)} hand-detect failures skipped)")
    plot_umap_tsne(np.asarray(feats, dtype=np.float32), labels,
                   "Landmark Feature Space (42-d) — colored by class",
                   OUT_DIR / "embedding_landmarks_by_class.png")


def _resnet_avgpool_feature(model, device, transform, img_rgb: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(img_rgb)
    tensor = transform(pil).unsqueeze(0).to(device)
    feat_holder = {}
    h = model.avgpool.register_forward_hook(
        lambda m, i, o: feat_holder.__setitem__("v", o.detach()[0].view(-1).cpu().numpy()))
    with torch.no_grad():
        model(tensor)
    h.remove()
    return feat_holder["v"]


def mode_cnn(per_class: int, seed: int):
    register_defaults()
    pipeline = build_pipeline("raw_resnet18")
    model = pipeline.recognizer.model
    device = pipeline.recognizer.device
    transform = pipeline.recognizer._transform

    feats, labels = [], []
    for path in sample_per_class(per_class, seed):
        raw_rgb = load_rgb(path)
        rep_out = pipeline.representation.process(raw_rgb)
        if rep_out is None:
            continue
        feats.append(_resnet_avgpool_feature(model, device, transform, rep_out.data))
        labels.append(path.parent.name)
    pipeline.close()
    print(f"Collected {len(feats)} CNN embeddings")
    plot_umap_tsne(np.asarray(feats, dtype=np.float32), labels,
                   "ResNet18 avgpool Feature Space (512-d) — colored by class",
                   OUT_DIR / "embedding_cnn_by_class.png")


def mode_repshift(per_class: int, seed: int):
    register_defaults()
    pipeline = build_pipeline("raw_resnet18")  # backbone weights, shared across representations
    model = pipeline.recognizer.model
    device = pipeline.recognizer.device
    transform = pipeline.recognizer._transform

    representations = [
        ("raw", RawImageRepresentation()),
        ("mediapipe_crop", MediaPipeCropRepresentation()),
        ("clahe", EnhancementRepresentation(method="clahe")),
        ("gamma", EnhancementRepresentation(method="gamma")),
    ]

    feats, labels = [], []
    paths = sample_per_class(per_class, seed)
    for rep_name, rep in representations:
        n_ok = 0
        for path in paths:
            raw_rgb = load_rgb(path)
            out = rep.process(raw_rgb)
            if out is None:
                continue
            feats.append(_resnet_avgpool_feature(model, device, transform, out.data))
            labels.append(rep_name)
            n_ok += 1
        print(f"  {rep_name}: {n_ok}/{len(paths)} images embedded")
        rep.close()
    pipeline.close()

    plot_umap_tsne(np.asarray(feats, dtype=np.float32), labels,
                   "Same images, different representations — colored by representation\n"
                   "(same ResNet18 backbone weights throughout)",
                   OUT_DIR / "embedding_representation_shift.png",
                   legend_title="Representation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["landmark", "cnn", "repshift"], required=True)
    parser.add_argument("--per-class", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.mode == "landmark":
        mode_landmark(args.per_class, args.seed)
    elif args.mode == "cnn":
        mode_cnn(args.per_class, args.seed)
    else:
        mode_repshift(args.per_class, args.seed)


if __name__ == "__main__":
    main()
