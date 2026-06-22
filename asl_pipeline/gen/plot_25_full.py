"""
Generate plots for ALL 25 pipelines from PREDICTED full data.
All numbers are ESTIMATES, NOT measured results.

Outputs → ./plot/

Usage:
    python gen/plot_25_full.py
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = Path("results/predicted_25_full.json")
PLOT_DIR = Path("plot")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_PATH) as f:
    DATA = json.load(f)

PIPES = DATA["pipelines"]
CORRUPTIONS = DATA["_meta"]["corruptions"]

ENH_COLORS = {"raw":"#6b7280","clahe":"#3b82f6","gamma":"#22c55e","sharpening":"#f59e0b","zero_dce":"#ef4444"}
ENH_LABELS = {"raw":"Raw","clahe":"CLAHE","gamma":"Gamma","sharpening":"Sharpening","zero_dce":"Zero-DCE++"}

def save(name):
    p = PLOT_DIR / name
    plt.savefig(str(p), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {p}")


# ── Plot 9: Training curves for all 25 pipelines (MLP only, grouped by enhancement)
def plot_all_training_curves():
    mlp_pipes = [p for p in PIPES if p["classifier"] == "mlp" and "training" in p and "epochs" in p.get("training",{})]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # Group by enhancement
    for idx, enh in enumerate(["raw","clahe","gamma","sharpening","zero_dce"]):
        ax = axes[idx]
        for p in mlp_pipes:
            if p["enhancement"] != enh:
                continue
            t = p["training"]
            rep = p["representation"]
            color = {"mediapipe":"#3b82f6","mmpose":"#22c55e"}[rep]
            style = "-" if rep == "mediapipe" else "--"
            ax.plot(t["epochs"], t["val_loss"], style, color=color, linewidth=1.5,
                    label=f"{rep} val_loss", markersize=3, marker="o")
        ax.set_title(f"{ENH_LABELS[enh]} — MLP Val Loss (PREDICTED)", fontsize=10)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Val Loss")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_ylim(0, 3.2)

    # Last subplot: XGBoost comparison
    ax = axes[5]
    xgb_pipes = [p for p in PIPES if p["classifier"] == "xgboost" and "training" in p and "rounds" in p.get("training",{})]
    for p in xgb_pipes:
        t = p["training"]
        enh = p["enhancement"]
        rep = p["representation"]
        label = f"{ENH_LABELS[enh]}+{rep}"
        color = ENH_COLORS[enh]
        style = "-" if rep == "mediapipe" else "--"
        ax.plot(t["rounds"], t["val_acc"], style, color=color, linewidth=1.2,
                label=label, markersize=2, marker="s")
    ax.set_title("XGBoost Val Accuracy by Rounds (PREDICTED)", fontsize=10)
    ax.set_xlabel("Boosting Rounds"); ax.set_ylabel("Val Accuracy")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)
    ax.set_ylim(0.5, 1.02)

    fig.suptitle("Training Curves — All 25 Pipelines (PREDICTED)", fontsize=14, y=1.02)
    fig.tight_layout()
    save("09_all_training_curves.png")


# ── Plot 10: Val accuracy convergence comparison (MLP: MP vs MMPose × 5 enhancements)
def plot_val_acc_convergence():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # MediaPipe MLP across enhancements
    ax = axes[0]
    for p in PIPES:
        if p["representation"] != "mediapipe" or p["classifier"] != "mlp":
            continue
        t = p.get("training", {})
        if "epochs" not in t:
            continue
        enh = p["enhancement"]
        ax.plot(t["epochs"], t["val_acc"], "-o", color=ENH_COLORS[enh], label=ENH_LABELS[enh],
                linewidth=1.5, markersize=3)
    ax.set_title("MediaPipe → MLP: Val Acc by Enhancement (PREDICTED)", fontsize=11)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val Accuracy")
    ax.set_ylim(0.4, 1.02); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # MMPose MLP across enhancements
    ax = axes[1]
    for p in PIPES:
        if p["representation"] != "mmpose" or p["classifier"] != "mlp":
            continue
        t = p.get("training", {})
        if "epochs" not in t:
            continue
        enh = p["enhancement"]
        ax.plot(t["epochs"], t["val_acc"], "-o", color=ENH_COLORS[enh], label=ENH_LABELS[enh],
                linewidth=1.5, markersize=3)
    ax.set_title("MMPose → MLP: Val Acc by Enhancement (PREDICTED)", fontsize=11)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val Accuracy")
    ax.set_ylim(0.4, 1.02); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.tight_layout()
    save("10_val_acc_convergence.png")


# ── Plot 11: Full robustness heatmap (25 pipelines × 9 corruptions)
def plot_full_robustness_heatmap():
    names = [p["name"] for p in PIPES]
    matrix = [p["robustness"] for p in PIPES]
    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(CORRUPTIONS)))
    ax.set_xticklabels(CORRUPTIONS, fontsize=8, rotation=35, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    for i in range(len(names)):
        for j in range(len(CORRUPTIONS)):
            v = matrix[i][j]
            color = "white" if v < 40 else "black"
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6, color=color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Accuracy (%)", fontsize=10)
    ax.set_title("Robustness Heatmap — All 25 Pipelines × 9 Corruptions\n(PREDICTED)", fontsize=13)
    fig.tight_layout()
    save("11_full_robustness_heatmap.png")


# ── Plot 12: MLP vs XGBoost delta per condition
def plot_mlp_vs_xgboost():
    pairs = []
    for enh in ["raw","clahe","gamma","sharpening","zero_dce"]:
        for rep in ["mediapipe","mmpose"]:
            mlp = next((p for p in PIPES if p["enhancement"]==enh and p["representation"]==rep and p["classifier"]=="mlp"), None)
            xgb = next((p for p in PIPES if p["enhancement"]==enh and p["representation"]==rep and p["classifier"]=="xgboost"), None)
            if mlp and xgb:
                pairs.append({
                    "label": f"{ENH_LABELS[enh]}+{rep[:2].upper()}",
                    "mlp_clean": mlp["clean_acc"],
                    "xgb_clean": xgb["clean_acc"],
                    "delta": xgb["clean_acc"] - mlp["clean_acc"],
                    "enh": enh,
                })

    fig, ax = plt.subplots(figsize=(14, 5))
    x = range(len(pairs))
    labels = [p["label"] for p in pairs]
    deltas = [p["delta"] for p in pairs]
    colors = [ENH_COLORS[p["enh"]] for p in pairs]

    bars = ax.bar(x, deltas, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Clean Acc Delta (XGBoost − MLP) %")
    ax.set_title("XGBoost vs MLP: Clean Accuracy Delta per Condition\n(PREDICTED — variable, not fixed offset)", fontsize=12)
    for bar, d in zip(bars, deltas):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y + 0.05 if y >= 0 else y - 0.15,
                f"{d:+.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save("12_mlp_vs_xgboost_delta.png")


# ── Plot 13: Representation comparison (grouped by enhancement, Real Acc)
def plot_repr_comparison():
    fig, ax = plt.subplots(figsize=(14, 6))
    enhancements = ["raw","clahe","gamma","sharpening","zero_dce"]
    reprs = ["mediapipe","mmpose","yolo"]
    repr_labels = ["MediaPipe (MLP)","MMPose (MLP)","YOLO (ResNet18)"]
    repr_colors = ["#3b82f6","#22c55e","#f59e0b"]

    x = np.arange(len(enhancements))
    width = 0.25
    for i, (rep, label, color) in enumerate(zip(reprs, repr_labels, repr_colors)):
        vals = []
        for enh in enhancements:
            if rep == "yolo":
                p = next((p for p in PIPES if p["enhancement"]==enh and p["representation"]=="yolo"), None)
            else:
                p = next((p for p in PIPES if p["enhancement"]==enh and p["representation"]==rep and p["classifier"]=="mlp"), None)
            vals.append(p["real_acc"] if p else 0)
        ax.bar(x + (i-1)*width, vals, width, label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([ENH_LABELS[e] for e in enhancements], fontsize=11)
    ax.set_ylabel("Real Accuracy (%)")
    ax.set_title("Representation Comparison by Enhancement\n(PREDICTED)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    save("13_repr_comparison.png")


if __name__ == "__main__":
    print("Generating FULL 25-pipeline plots (PREDICTED data)...\n")
    plot_all_training_curves()
    plot_val_acc_convergence()
    plot_full_robustness_heatmap()
    plot_mlp_vs_xgboost()
    plot_repr_comparison()
    total = len(list(PLOT_DIR.glob("*.png")))
    print(f"\nTotal: {total} plots in {PLOT_DIR}/")
