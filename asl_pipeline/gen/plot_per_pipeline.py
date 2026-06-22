"""
Output: ./plot/per_pipeline/
  - train_{name}.png  (25 files)
  - eval_{name}.png   (25 files)
  Total: 50 images

Usage:
    python gen/plot_per_pipeline.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = Path("results/25_full.json")
OUT_DIR = Path("plot/per_pipeline")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_PATH) as f:
    DATA = json.load(f)

PIPES = DATA["pipelines"]
CORRUPTIONS = DATA["_meta"]["corruptions"]

ENH_LABELS = {"raw":"Raw","clahe":"CLAHE","gamma":"Gamma","sharpening":"Sharpening","zero_dce":"Zero-DCE++"}
REP_LABELS = {"mediapipe":"MediaPipe Landmarks","mmpose":"MMPose RTMPose-Hand","yolo":"YOLOv11 Hand Crop"}
CLF_LABELS = {"mlp":"MLP (256→128→64)","xgboost":"XGBoost (200 trees)","resnet18":"ResNet18 (pretrained)"}
ENH_COLORS = {"raw":"#6b7280","clahe":"#3b82f6","gamma":"#22c55e","sharpening":"#f59e0b","zero_dce":"#ef4444"}

CORR_SHORT = ["clean","low\nlight","over\nexpose","motion\nblur","gauss\nnoise","rotation","scale\nshift","crop\nshift","partial\noccl."]


def plot_train(pipe):
    """Generate training plot for one pipeline."""
    name = pipe["name"]
    enh = pipe["enhancement"]
    rep = pipe["representation"]
    clf = pipe["classifier"]
    t = pipe.get("training", {})

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(f"Training — {name}\n"
                 f"Enhancement: {ENH_LABELS[enh]}  |  Representation: {REP_LABELS[rep]}  |  Classifier: {CLF_LABELS[clf]}\n"
                 ,fontsize=11, y=1.08)

    color = ENH_COLORS[enh]

    if t.get("pretrained"):
        # ResNet18 pretrained — no training curve
        for ax in axes:
            ax.text(0.5, 0.5, f"Pretrained model\n(no training performed)\n\n"
                    f"Checkpoint: HuggingFace\nhuzaifanasirrr/realtime-sign-language-translator\n\n"
                    f"Enhancement ({ENH_LABELS[enh]}) applied at inference time",
                    ha="center", va="center", fontsize=11, transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.8", facecolor="#f0f0f0"))
            ax.set_xticks([]); ax.set_yticks([])
        axes[0].set_title("Loss Curve")
        axes[1].set_title("Accuracy Curve")

    elif "epochs" in t:
        # MLP training
        epochs = t["epochs"]
        axes[0].plot(epochs, t["train_loss"], "-", color=color, linewidth=1.5, label="Train Loss")
        axes[0].plot(epochs, t["val_loss"], "--", color=color, linewidth=1.5, alpha=0.7, label="Val Loss")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].set_title(f"Loss — {CLF_LABELS[clf]}")
        axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

        axes[1].plot(epochs, t["train_acc"], "-", color=color, linewidth=1.5, label="Train Acc")
        axes[1].plot(epochs, t["val_acc"], "--", color=color, linewidth=1.5, alpha=0.7, label="Val Acc")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
        axes[1].set_title(f"Accuracy — {CLF_LABELS[clf]}")
        axes[1].set_ylim(0, 1.05)
        axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

        # Annotate final values
        final_tl = t["train_loss"][-1]
        final_vl = t["val_loss"][-1]
        final_va = t["val_acc"][-1]
        axes[0].annotate(f"train: {final_tl:.3f}\nval: {final_vl:.3f}",
                         xy=(epochs[-1], final_vl), fontsize=8, color=color,
                         xytext=(-60, 20), textcoords="offset points",
                         arrowprops=dict(arrowstyle="->", color=color, lw=0.8))
        axes[1].annotate(f"val: {final_va:.1%}",
                         xy=(epochs[-1], final_va), fontsize=9, fontweight="bold", color=color,
                         xytext=(-50, -25), textcoords="offset points",
                         arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

    elif "rounds" in t:
        # XGBoost boosting rounds
        rounds = t["rounds"]
        axes[0].plot(rounds, t["train_acc"], "-o", color=color, linewidth=1.5, markersize=4, label="Train Acc")
        axes[0].plot(rounds, t["val_acc"], "--s", color=color, linewidth=1.5, markersize=4, alpha=0.7, label="Val Acc")
        axes[0].set_xlabel("Boosting Rounds"); axes[0].set_ylabel("Accuracy")
        axes[0].set_title(f"XGBoost Accuracy vs Rounds")
        axes[0].set_ylim(0.5, 1.05)
        axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

        final_va = t["val_acc"][-1]
        axes[0].annotate(f"val: {final_va:.1%}",
                         xy=(rounds[-1], final_va), fontsize=9, fontweight="bold", color=color,
                         xytext=(-50, -20), textcoords="offset points",
                         arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

        # Right panel: XGBoost config
        config_text = (
            f"XGBoost Configuration\n\n"
            f"n_estimators: 200\n"
            f"max_depth: 6\n"
            f"learning_rate: 0.1\n"
            f"eval_metric: mlogloss\n\n"
            f"Input: 42-dim landmark vector\n"
            f"Classes: 26 (A-Z)\n\n"
            f"Final Val Acc: {final_va:.1%}"
        )
        axes[1].text(0.5, 0.5, config_text, ha="center", va="center", fontsize=10,
                     transform=axes[1].transAxes, family="monospace",
                     bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f8f0"))
        axes[1].set_xticks([]); axes[1].set_yticks([])
        axes[1].set_title("Model Configuration")

    fig.tight_layout()
    path = OUT_DIR / f"train_{name}.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_eval(pipe):
    """Generate evaluation plot for one pipeline."""
    name = pipe["name"]
    enh = pipe["enhancement"]
    rep = pipe["representation"]
    clf = pipe["classifier"]
    rob = pipe["robustness"]
    color = ENH_COLORS[enh]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Evaluation — {name}\n"
                 f"Enhancement: {ENH_LABELS[enh]}  |  Representation: {REP_LABELS[rep]}  |  Classifier: {CLF_LABELS[clf]}\n"
                 ,
                 fontsize=11, y=1.08)

    # Left: Key metrics bar chart
    metrics = {
        "Clean\nAcc": pipe["clean_acc"],
        "Real\nAcc": pipe["real_acc"],
        "100 −\nFail%": 100 - pipe["hand_fail"],
    }
    labels = list(metrics.keys())
    values = list(metrics.values())
    bar_colors = [color, color, "#22c55e" if pipe["hand_fail"] < 15 else "#f59e0b" if pipe["hand_fail"] < 25 else "#ef4444"]

    bars = axes[0].bar(labels, values, color=bar_colors, edgecolor="white", width=0.6)
    axes[0].set_ylim(0, 110)
    axes[0].set_ylabel("%")
    axes[0].set_title("Key Metrics")
    axes[0].grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                     f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")

    # Add fail rate text
    axes[0].text(0.98, 0.02, f"Hand Detection Failure: {pipe['hand_fail']:.2f}%",
                 ha="right", va="bottom", transform=axes[0].transAxes, fontsize=9,
                 color="#ef4444" if pipe["hand_fail"] > 20 else "#6b7280")

    # Right: Robustness bar chart (9 corruptions)
    x = np.arange(len(CORRUPTIONS))
    bar_colors_rob = []
    for v in rob:
        if v >= 80: bar_colors_rob.append("#22c55e")
        elif v >= 50: bar_colors_rob.append("#84cc16")
        elif v >= 20: bar_colors_rob.append("#f59e0b")
        else: bar_colors_rob.append("#ef4444")

    bars2 = axes[1].bar(x, rob, color=bar_colors_rob, edgecolor="white", width=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(CORR_SHORT, fontsize=7.5)
    axes[1].set_ylim(0, 110)
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Robustness Under Corruptions")
    axes[1].grid(axis="y", alpha=0.3)
    for bar, v in zip(bars2, rob):
        if v > 0:
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                         f"{v:.0f}", ha="center", fontsize=7, fontweight="bold")

    fig.tight_layout()
    path = OUT_DIR / f"eval_{name}.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(f"Output: {OUT_DIR}/\n")

    for pipe in PIPES:
        tp = plot_train(pipe)
        ep = plot_eval(pipe)
        print(f"  #{pipe['id']:2d} {pipe['name']:<30s}  train + eval")

    total = len(list(OUT_DIR.glob("*.png")))
    print(f"\nDone. {total} plots in {OUT_DIR}/")
