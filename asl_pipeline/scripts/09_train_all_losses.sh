#!/usr/bin/env bash
# Step 9: Train landmark MLP with all 4 loss functions + train image classifier
# Generates loss curves for the report.
set -e

PYTHON="${PYTHON:-python}"
FEATURES="${FEATURES:-data/landmarks_fast.csv}"
DATASET="${DATASET:-data/asl_alphabet/train}"
OUTPUT="${OUTPUT:-outputs}"

echo "=== Step 9: Training with multiple losses ==="

# ── Part A: Landmark MLP with 4 losses ──
echo ""
echo "--- Landmark MLP Training ---"
for LOSS in ce ce_label_smoothing focal weighted_ce; do
    echo ""
    echo ">>> landmark MLP loss=$LOSS"
    $PYTHON train_landmark_pytorch.py \
        --features "$FEATURES" \
        --loss "$LOSS" \
        --epochs 100 \
        --output-dir "$OUTPUT" 2>&1 | tail -5
done

# ── Part B: Image classifier with 2 losses (ResNet18, small epochs) ──
echo ""
echo "--- Image Classifier Training ---"
for LOSS in ce focal; do
    echo ""
    echo ">>> ResNet18 loss=$LOSS"
    $PYTHON train_image_classifier.py \
        --dataset "$DATASET" \
        --model resnet18 \
        --loss "$LOSS" \
        --epochs 10 \
        --batch-size 64 \
        --output-dir "$OUTPUT" 2>&1 | tail -5
done

# ── Part C: Plot loss comparison ──
echo ""
echo "--- Generating loss comparison chart ---"
$PYTHON << 'PYEOF'
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

logs_dir = Path("outputs/logs")
figures_dir = Path("outputs/figures")
figures_dir.mkdir(parents=True, exist_ok=True)

# Landmark MLP loss comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for log_file in sorted(logs_dir.glob("landmark_mlp_*_train_log.csv")):
    loss_name = log_file.stem.replace("landmark_mlp_", "").replace("_train_log", "")
    epochs, train_loss, val_loss, val_acc = [], [], [], []
    with open(log_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            val_acc.append(float(row["val_accuracy"]))
    axes[0].plot(epochs, val_loss, label=loss_name)
    axes[1].plot(epochs, val_acc, label=loss_name)

axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Validation Loss")
axes[0].set_title("Landmark MLP — Loss Comparison")
axes[0].legend()

axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Validation Accuracy")
axes[1].set_title("Landmark MLP — Accuracy Comparison")
axes[1].set_ylim(0, 1.05)
axes[1].legend()

fig.tight_layout()
fig.savefig(str(figures_dir / "landmark_mlp_loss_comparison.png"), dpi=150)
plt.close(fig)
print("Saved: landmark_mlp_loss_comparison.png")

# Image classifier loss comparison (if logs exist)
img_logs = list(logs_dir.glob("resnet18_*_train_log.csv"))
if img_logs:
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    for log_file in sorted(img_logs):
        loss_name = log_file.stem.replace("resnet18_", "").replace("_train_log", "")
        epochs, train_loss, val_loss, val_acc = [], [], [], []
        with open(log_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row["epoch"]))
                train_loss.append(float(row["train_loss"]))
                val_loss.append(float(row["val_loss"]))
                val_acc.append(float(row["val_accuracy"]))
        axes2[0].plot(epochs, val_loss, label=loss_name)
        axes2[1].plot(epochs, val_acc, label=loss_name)

    axes2[0].set_xlabel("Epoch")
    axes2[0].set_ylabel("Validation Loss")
    axes2[0].set_title("ResNet18 — Loss Comparison")
    axes2[0].legend()
    axes2[1].set_xlabel("Epoch")
    axes2[1].set_ylabel("Validation Accuracy")
    axes2[1].set_title("ResNet18 — Accuracy Comparison")
    axes2[1].set_ylim(0, 1.05)
    axes2[1].legend()
    fig2.tight_layout()
    fig2.savefig(str(figures_dir / "resnet18_loss_comparison.png"), dpi=150)
    plt.close(fig2)
    print("Saved: resnet18_loss_comparison.png")
PYEOF

echo ""
echo "=== Training Done ==="
echo "Curves at: $OUTPUT/figures/"
echo "Logs at: $OUTPUT/logs/"
echo "Checkpoints at: $OUTPUT/checkpoints/"
