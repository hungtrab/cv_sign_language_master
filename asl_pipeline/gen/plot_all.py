"""
Outputs → ./plot/

Usage:
    python gen/plot_all.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RESULTS_PATH = Path("results/25_pipelines.json")
PLOT_DIR = Path("plot")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

with open(RESULTS_PATH) as f:
    DATA = json.load(f)

PIPELINES = DATA["pipelines"]

ENH_COLORS = {
    "raw": "#6b7280",
    "clahe": "#3b82f6",
    "gamma": "#22c55e",
    "sharpening": "#f59e0b",
    "zero_dce": "#ef4444",
}
ENH_LABELS = {
    "raw": "Raw",
    "clahe": "CLAHE",
    "gamma": "Gamma",
    "sharpening": "Sharpening",
    "zero_dce": "Zero-DCE++",
}
REPR_MARKERS = {"mediapipe": "o", "mmpose": "s", "yolo": "D"}


def fig_saver(name):
    path = PLOT_DIR / name
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────
# Plot 1: Real Accuracy grouped bar chart (enhancement × repr+clf)
# ──────────────────────────────────────────────────────────────
def plot_real_accuracy_grouped():
    fig, ax = plt.subplots(figsize=(16, 6))
    enhancements = ["raw", "clahe", "gamma", "sharpening", "zero_dce"]
    combos = ["mp_mlp", "mp_xgb", "mmpose_mlp", "mmpose_xgb", "yolo_resnet18"]
    combo_labels = ["MP→MLP", "MP→XGB", "MMPose→MLP", "MMPose→XGB", "YOLO→R18"]

    x = np.arange(len(enhancements))
    width = 0.15
    for i, (combo, label) in enumerate(zip(combos, combo_labels)):
        vals = []
        for enh in enhancements:
            name = f"{enh}_{combo}"
            p = next((p for p in PIPELINES if p["name"] == name), None)
            vals.append(p["real_acc"] if p else 0)
        offset = (i - 2) * width
        bars = ax.bar(x + offset, vals, width, label=label, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels([ENH_LABELS[e] for e in enhancements], fontsize=11)
    ax.set_ylabel("Real Accuracy (%)", fontsize=12)
    ax.set_title("Real Accuracy by Enhancement × Representation+Classifier",
                 fontsize=13)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    fig_saver("01_real_accuracy_grouped.png")


# ──────────────────────────────────────────────────────────────
# Plot 2: Detection failure rate heatmap
# ──────────────────────────────────────────────────────────────
def plot_detection_heatmap():
    det = DATA["detection_failure"]
    enhancements = ["raw", "clahe", "gamma", "sharpening", "zero_dce"]
    detectors = ["mediapipe", "mmpose", "yolo"]
    matrix = [[det[e][d] for d in detectors] for e in enhancements]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=40)
    ax.set_xticks(range(len(detectors)))
    ax.set_xticklabels(["MediaPipe", "MMPose", "YOLOv11"], fontsize=11)
    ax.set_yticks(range(len(enhancements)))
    ax.set_yticklabels([ENH_LABELS[e] for e in enhancements], fontsize=11)
    for i in range(len(enhancements)):
        for j in range(len(detectors)):
            v = matrix[i][j]
            color = "white" if v > 20 else "black"
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center", fontsize=12,
                    fontweight="bold", color=color)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Detection Failure Rate (%)", fontsize=10)
    ax.set_title("Hand Detection Failure Rate", fontsize=13)
    fig.tight_layout()
    fig_saver("02_detection_failure_heatmap.png")


# ──────────────────────────────────────────────────────────────
# Plot 3: Real Accuracy vs Low-Light scatter
# ──────────────────────────────────────────────────────────────
def plot_accuracy_vs_lowlight():
    fig, ax = plt.subplots(figsize=(10, 7))
    for p in PIPELINES:
        enh = p["enhancement"]
        rep = p["representation"]
        ax.scatter(p["real_acc"], p["low_light"],
                   c=ENH_COLORS[enh], marker=REPR_MARKERS[rep],
                   s=100, edgecolors="white", linewidths=0.5, zorder=3)
        ax.annotate(p["name"], (p["real_acc"], p["low_light"]),
                    fontsize=5.5, alpha=0.7, xytext=(4, 4),
                    textcoords="offset points")

    # Legend for enhancements (color)
    for enh, color in ENH_COLORS.items():
        ax.scatter([], [], c=color, s=80, label=ENH_LABELS[enh])
    # Legend for representations (marker)
    for rep, marker in REPR_MARKERS.items():
        ax.scatter([], [], c="gray", marker=marker, s=80, label=rep.capitalize())

    ax.set_xlabel("Real Accuracy (%)", fontsize=12)
    ax.set_ylabel("Low-Light Robustness (%)", fontsize=12)
    ax.set_title("Real Accuracy vs Low-Light Robustness", fontsize=13)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(alpha=0.3)
    ax.set_xlim(45, 100)
    ax.set_ylim(-5, 90)
    fig.tight_layout()
    fig_saver("03_accuracy_vs_lowlight_scatter.png")


# ──────────────────────────────────────────────────────────────
# Plot 4: Training loss curves (4 losses)
# ──────────────────────────────────────────────────────────────
def plot_training_curves():
    tc = DATA["training_curves"]
    epochs = tc["epochs"]
    losses = ["ce", "ce_label_smoothing", "focal", "weighted_ce"]
    loss_labels = ["Cross Entropy", "CE + Label Smoothing", "Focal Loss", "Weighted CE"]
    colors = ["#3b82f6", "#22c55e", "#ef4444", "#f59e0b"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Val loss comparison
    for loss, label, color in zip(losses, loss_labels, colors):
        axes[0].plot(epochs, tc[loss]["val_loss"], "-o", label=label, color=color,
                     markersize=3, linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation Loss")
    axes[0].set_title("Landmark MLP — Val Loss Comparison")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # Val accuracy comparison
    for loss, label, color in zip(losses, loss_labels, colors):
        axes[1].plot(epochs, tc[loss]["val_acc"], "-o", label=label, color=color,
                     markersize=3, linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Accuracy")
    axes[1].set_title("Landmark MLP — Val Accuracy Comparison")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig_saver("04_training_loss_comparison.png")

    # Individual train vs val curves
    for loss, label, color in zip(losses, loss_labels, colors):
        fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4.5))
        axes2[0].plot(epochs, tc[loss]["train_loss"], "-", label="Train", color=color, linewidth=1.5)
        axes2[0].plot(epochs, tc[loss]["val_loss"], "--", label="Val", color=color, linewidth=1.5, alpha=0.7)
        axes2[0].set_xlabel("Epoch"); axes2[0].set_ylabel("Loss")
        axes2[0].set_title(f"{label} — Loss Curve")
        axes2[0].legend(); axes2[0].grid(alpha=0.3)

        axes2[1].plot(epochs, tc[loss]["train_acc"], "-", label="Train", color=color, linewidth=1.5)
        axes2[1].plot(epochs, tc[loss]["val_acc"], "--", label="Val", color=color, linewidth=1.5, alpha=0.7)
        axes2[1].set_xlabel("Epoch"); axes2[1].set_ylabel("Accuracy")
        axes2[1].set_title(f"{label} — Accuracy Curve")
        axes2[1].set_ylim(0, 1.05); axes2[1].legend(); axes2[1].grid(alpha=0.3)
        fig2.tight_layout()
        fig_saver(f"04_training_{loss}_curves.png")


# ──────────────────────────────────────────────────────────────
# Plot 5: Per-class F1 bar chart
# ──────────────────────────────────────────────────────────────
def plot_per_class_f1():
    pcf = DATA["per_class_f1"]
    classes = pcf["classes"]

    pipelines_to_plot = ["gamma_yolo_resnet18", "raw_mp_mlp", "zero_dce_yolo_resnet18"]
    labels = ["Gamma→YOLO→R18 (top)", "Raw→MP→MLP (baseline)", "DCE++→YOLO→R18"]
    colors = ["#22c55e", "#6b7280", "#ef4444"]

    fig, ax = plt.subplots(figsize=(16, 5))
    x = np.arange(len(classes))
    width = 0.25
    for i, (pipe, label, color) in enumerate(zip(pipelines_to_plot, labels, colors)):
        vals = pcf[pipe]
        ax.bar(x + (i - 1) * width, vals, width, label=label, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Per-Class F1 Score — Top Pipelines", fontsize=13)
    ax.set_ylim(0.6, 1.05)
    ax.axhline(y=0.90, color="gray", linestyle="--", alpha=0.5, label="F1=0.90")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Highlight confused pairs
    for idx, cls in enumerate(classes):
        if cls in ("A", "E", "S", "M", "N", "T", "G", "H", "J", "Z", "U", "V", "R"):
            ax.get_xticklabels()[idx].set_color("red")
            ax.get_xticklabels()[idx].set_fontweight("bold")

    fig.tight_layout()
    fig_saver("05_per_class_f1.png")


# ──────────────────────────────────────────────────────────────
# Plot 6: Robustness radar chart
# ──────────────────────────────────────────────────────────────
def plot_robustness_radar():
    rob = DATA["robustness"]
    corruptions = rob["corruptions"]
    n = len(corruptions)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    pipelines_to_plot = [
        ("gamma_yolo_resnet18", "#22c55e", "Gamma→YOLO→R18"),
        ("zero_dce_yolo_resnet18", "#ef4444", "DCE++→YOLO→R18"),
        ("raw_mp_mlp", "#6b7280", "Raw→MP→MLP"),
        ("clahe_yolo_resnet18", "#3b82f6", "CLAHE→YOLO→R18"),
    ]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for pipe_name, color, label in pipelines_to_plot:
        if pipe_name not in rob:
            continue
        vals = rob[pipe_name]
        vals_plot = vals + vals[:1]
        ax.plot(angles, vals_plot, "-o", color=color, linewidth=1.5, markersize=4, label=label)
        ax.fill(angles, vals_plot, alpha=0.08, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]), corruptions, fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_title("Robustness Under Corruptions", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    fig.tight_layout()
    fig_saver("06_robustness_radar.png")


# ──────────────────────────────────────────────────────────────
# Plot 7: Confusion matrix (top pipeline)
# ──────────────────────────────────────────────────────────────
def plot_confusion_matrix():
    classes = DATA["per_class_f1"]["classes"]
    n = len(classes)
    # Build from confusions + assume 100 samples per class
    cm = np.zeros((n, n), dtype=int)
    samples_per_class = 100
    confusions = DATA["confusion_matrix_top_pipeline"]["confusions"]

    for c in confusions:
        i = classes.index(c["true"])
        j = classes.index(c["pred"])
        cm[i, j] = c["count"]

    # Fill diagonal (correct predictions = samples - sum of row errors)
    for i in range(n):
        off_diag = cm[i, :].sum()
        cm[i, i] = max(0, samples_per_class - off_diag)

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix — Gamma→YOLO→ResNet18", fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Annotate non-zero off-diagonal
    thresh = cm.max() / 2
    for i in range(n):
        for j in range(n):
            if cm[i, j] > 0:
                color = "white" if cm[i, j] > thresh else "black"
                fontsize = 7 if i == j else 6
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color=color, fontsize=fontsize)

    fig.tight_layout()
    fig_saver("07_confusion_matrix.png")


# ──────────────────────────────────────────────────────────────
# Plot 8: Enhancement effect summary (bar chart)
# ──────────────────────────────────────────────────────────────
def plot_enhancement_effect():
    det = DATA["detection_failure"]
    enhancements = ["raw", "clahe", "gamma", "sharpening", "zero_dce"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Detection failure (avg across detectors)
    avg_fails = []
    for e in enhancements:
        vals = list(det[e].values())
        avg_fails.append(np.mean(vals))
    colors = [ENH_COLORS[e] for e in enhancements]
    bars = axes[0].bar([ENH_LABELS[e] for e in enhancements], avg_fails, color=colors)
    axes[0].axhline(y=avg_fails[0], color="gray", linestyle="--", alpha=0.5, label="Raw baseline")
    axes[0].set_ylabel("Avg Detection Failure Rate (%)")
    axes[0].set_title("Enhancement Effect on Detection")
    for bar, v in zip(bars, avg_fails):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # Average Real Accuracy
    avg_real = []
    for e in enhancements:
        vals = [p["real_acc"] for p in PIPELINES if p["enhancement"] == e]
        avg_real.append(np.mean(vals))
    bars2 = axes[1].bar([ENH_LABELS[e] for e in enhancements], avg_real, color=colors)
    axes[1].set_ylabel("Avg Real Accuracy (%)")
    axes[1].set_title("Enhancement Effect on Real Accuracy")
    for bar, v in zip(bars2, avg_real):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
    axes[1].set_ylim(0, 105)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig_saver("08_enhancement_effect_summary.png")


# ──────────────────────────────────────────────────────────────
# Run all
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_real_accuracy_grouped()
    plot_detection_heatmap()
    plot_accuracy_vs_lowlight()
    plot_training_curves()
    plot_per_class_f1()
    plot_robustness_radar()
    plot_confusion_matrix()
    plot_enhancement_effect()

    total = len(list(PLOT_DIR.glob("*.png")))
    print(f"\nDone. {total} plots saved to {PLOT_DIR}/")
