"""Visualize what happens *inside* a ResNet18 recognizer, layer by layer.

This goes one level deeper than visualize_steps.py: instead of just showing
representation-in / recognizer-out, it shows the activation maps produced by
each ResNet18 stage (conv1, layer1..layer4), a Grad-CAM heatmap of what the
network attended to for its predicted class, and how the final 512-d pooled
feature vector turns into class logits.

Only works for pipelines whose recognizer wraps a torchvision ResNet18
(resnet18_asl / torchvision_classifier) — ViT/SigLIP recognizers use a
different internal structure and are out of scope here.

Usage:
    python visualize_cnn_internals.py --pipeline raw_resnet18 --seed 0
    python visualize_cnn_internals.py --pipeline mediapipe_crop_resnet18 \
        --image data/asl_alphabet/test_split/A/A910.jpg
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
import torch
import torch.nn.functional as F
from PIL import Image

from asl.pipelines.registry import build_pipeline, register_defaults

PRED_DIR = Path("outputs/predictions")
OUT_DIR = Path("outputs/visualizations")

LAYER_NAMES = ["conv1", "layer1", "layer2", "layer3", "layer4"]


def load_rgb(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def pick_case(pipeline_name: str, seed: int = 0) -> Optional[dict]:
    csv_path = PRED_DIR / f"{pipeline_name}_predictions.csv"
    if not csv_path.exists():
        return None
    rng = random.Random(seed)
    correct = [r for r in csv.DictReader(open(csv_path))
               if r.get("pred_label") not in ("", "Unknown") and r["pred_label"] == r["true_label"]]
    return rng.choice(correct) if correct else None


def get_torch_model(recognizer):
    model = getattr(recognizer, "model", None)
    if model is None or not hasattr(model, "layer4"):
        raise ValueError(
            f"Recognizer '{type(recognizer).__name__}' is not a torchvision ResNet18 — "
            "this script only supports resnet18-based recognizers (raw_resnet18, "
            "mediapipe_crop_resnet18, enhancement_*_resnet18, no_enhance_resnet18)."
        )
    return model


def heatmap_overlay(activation: torch.Tensor, base_rgb: np.ndarray) -> np.ndarray:
    """activation: (C, H, W) -> mean over channels -> normalized heatmap over base image."""
    amap = activation.mean(dim=0)
    amap = amap.clamp(min=0)
    amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
    amap = amap.numpy()
    h, w = base_rgb.shape[:2]
    amap_resized = cv2.resize(amap, (w, h))
    heat = (plt.cm.jet(amap_resized)[:, :, :3] * 255).astype(np.uint8)
    overlay = (0.45 * heat + 0.55 * base_rgb).astype(np.uint8)
    return overlay


def grad_cam(model, tensor: torch.Tensor, class_idx: int) -> torch.Tensor:
    acts, grads = {}, {}

    def fwd(_m, _i, o):
        acts["v"] = o

    def bwd(_m, _gi, go):
        grads["v"] = go[0]

    h1 = model.layer4.register_forward_hook(fwd)
    h2 = model.layer4.register_full_backward_hook(bwd)
    try:
        model.zero_grad()
        out = model(tensor)
        score = out[0, class_idx]
        score.backward()
        act = acts["v"][0]          # (C, H, W)
        grad = grads["v"][0]        # (C, H, W)
        weights = grad.mean(dim=(1, 2))         # (C,)
        cam = F.relu((weights[:, None, None] * act).sum(0))  # (H, W)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu()
    finally:
        h1.remove()
        h2.remove()


def visualize(pipeline_name: str, image_path: Path, true_label: Optional[str], out_path: Path):
    register_defaults()
    pipeline = build_pipeline(pipeline_name)
    model = get_torch_model(pipeline.recognizer)
    device = pipeline.recognizer.device

    raw_rgb = load_rgb(image_path)
    rep_output = pipeline.representation.process(raw_rgb)
    if rep_output is None:
        raise RuntimeError(f"No hand detected for {image_path} with {pipeline_name}")

    input_img = rep_output.data  # what actually goes into the CNN
    pil = Image.fromarray(input_img)
    tensor = pipeline.recognizer._transform(pil).unsqueeze(0).to(device)
    tensor.requires_grad_(False)

    acts = {}
    hooks = []
    for name in LAYER_NAMES:
        layer = getattr(model, name)
        hooks.append(layer.register_forward_hook(
            lambda m, i, o, name=name: acts.__setitem__(name, o.detach()[0].cpu())))

    avgpool_feat = {}
    hooks.append(model.avgpool.register_forward_hook(
        lambda m, i, o: avgpool_feat.__setitem__("v", o.detach()[0].view(-1).cpu())))

    with torch.no_grad():
        logits = model(tensor).squeeze(0)
    for h in hooks:
        h.remove()

    probs = logits.softmax(-1).cpu().numpy()
    pred_idx = int(probs.argmax())
    class_names = pipeline.recognizer.class_names
    pred_label = class_names[pred_idx]
    top3_idx = probs.argsort()[::-1][:3]

    cam = grad_cam(model, tensor, pred_idx)
    cam_overlay = heatmap_overlay(cam.unsqueeze(0), input_img)

    n_cols = 2 + len(LAYER_NAMES) + 1 + 1  # raw + input + 5 layers + gradcam + final-vector/logits
    fig, axes = plt.subplots(1, n_cols, figsize=(3.0 * n_cols, 3.2))

    axes[0].imshow(raw_rgb)
    axes[0].set_title("ảnh gốc", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(input_img)
    axes[1].set_title("input CNN\n(sau representation)", fontsize=9)
    axes[1].axis("off")

    for i, name in enumerate(LAYER_NAMES):
        overlay = heatmap_overlay(acts[name], input_img)
        c, h, w = acts[name].shape
        axes[2 + i].imshow(overlay)
        axes[2 + i].set_title(f"{name}\n{c} kênh, {h}x{w}", fontsize=9)
        axes[2 + i].axis("off")

    ax_cam = axes[2 + len(LAYER_NAMES)]
    ax_cam.imshow(cam_overlay)
    ax_cam.set_title(f"Grad-CAM\n(vùng quyết định '{pred_label}')", fontsize=9)
    ax_cam.axis("off")

    ax_final = axes[-1]
    feat = avgpool_feat["v"].numpy()
    labels = [class_names[i] for i in top3_idx]
    scores = [probs[i] * 100 for i in top3_idx]
    is_correct = true_label is not None and pred_label == true_label
    colors = ["seagreen" if l == pred_label else "lightgray" for l in labels]
    if true_label is not None and not is_correct:
        colors = ["crimson" if l == pred_label else "lightgray" for l in labels]
    ax_final.barh(labels[::-1], scores[::-1], color=colors[::-1])
    ax_final.set_xlim(0, 100)
    title = f"512-d vector → FC → top-3\npred={pred_label} ({probs[pred_idx]*100:.1f}%)"
    if true_label:
        title += f"\ntrue={true_label} [{'ĐÚNG' if is_correct else 'SAI'}]"
    ax_final.set_title(title, fontsize=8,
                        color=("seagreen" if is_correct else "crimson") if true_label else "black")
    ax_final.tick_params(labelsize=8)

    fig.suptitle(
        f"{pipeline_name} — bên trong ResNet18 — {image_path.name}"
        f"  (avgpool feature mean={feat.mean():.3f}, std={feat.std():.3f})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved -> {out_path}")
    pipeline.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize ResNet18 internal layers")
    parser.add_argument("--pipeline", required=True,
                         help="A resnet18-based pipeline, e.g. raw_resnet18, "
                              "mediapipe_crop_resnet18, enhancement_clahe_resnet18")
    parser.add_argument("--image", default=None,
                         help="Specific image path. If omitted, sample a correct case "
                              "from outputs/predictions/<pipeline>_predictions.csv")
    parser.add_argument("--true-label", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.image:
        image_path = Path(args.image)
        true_label = args.true_label or image_path.parent.name
    else:
        row = pick_case(args.pipeline, seed=args.seed)
        if row is None:
            raise SystemExit(f"No correct-case example found for {args.pipeline}; pass --image explicitly.")
        image_path = Path(row["image_path"])
        true_label = row["true_label"]

    out_path = OUT_DIR / f"cnn_internals_{args.pipeline}_{image_path.stem}.png"
    visualize(args.pipeline, image_path, true_label, out_path)


if __name__ == "__main__":
    main()
