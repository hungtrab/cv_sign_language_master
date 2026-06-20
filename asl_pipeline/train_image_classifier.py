"""Train an image classifier for ASL A-Z.

Per eval doc §9.1:
  - Supports: ce, ce_label_smoothing, weighted_ce, focal
  - Outputs: train_log.csv, best checkpoint, loss/accuracy curves

Usage:
    python train_image_classifier.py --dataset data/asl_alphabet --model resnet18 --epochs 20
    python train_image_classifier.py --dataset data/asl --model mobilenet_v3_small --loss ce_label_smoothing
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from asl.data.asl_alphabet import ASLAlphabetDataset, AZ_CLASSES, build_dataloaders
from asl.data.transforms import build_train_transform, build_eval_transform
from asl.recognizers.torchvision_classifier import TorchvisionClassifier
from asl.utils.logging import get_logger

log = get_logger("asl.train")


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()


def build_criterion(loss_type: str, num_classes: int, class_weights=None):
    if loss_type == "ce":
        return torch.nn.CrossEntropyLoss()
    elif loss_type == "ce_label_smoothing":
        return torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    elif loss_type == "weighted_ce":
        if class_weights is not None:
            return torch.nn.CrossEntropyLoss(weight=class_weights)
        return torch.nn.CrossEntropyLoss()
    elif loss_type == "focal":
        return FocalLoss()
    else:
        raise ValueError(f"Unknown loss: {loss_type}. Options: ce, ce_label_smoothing, weighted_ce, focal")


def main():
    parser = argparse.ArgumentParser(description="Train image classifier for ASL A-Z")
    parser.add_argument("--dataset", required=True, help="Path to ASL alphabet train dir")
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "resnet50", "mobilenet_v3_small", "efficientnet_b0"])
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--loss", default="ce", choices=["ce", "ce_label_smoothing", "weighted_ce", "focal"])
    parser.add_argument("--output-dir", default="outputs/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}, Model: {args.model}, Loss: {args.loss}")

    run_name = f"{args.model}_{args.loss}"
    checkpoints_dir = Path(args.output_dir) / "checkpoints"
    logs_dir = Path(args.output_dir) / "logs"
    figures_dir = Path(args.output_dir) / "figures"
    for d in [checkpoints_dir, logs_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    train_tf = build_train_transform(args.input_size)
    eval_tf = build_eval_transform(args.input_size)
    train_loader, val_loader = build_dataloaders(
        args.dataset, class_names=AZ_CLASSES,
        train_tf=train_tf, eval_tf=eval_tf,
        batch_size=args.batch_size, seed=args.seed,
    )

    # Build model
    recognizer = TorchvisionClassifier(backbone=args.model, num_classes=len(AZ_CLASSES),
                                        image_size=args.input_size, device=str(device))
    model = recognizer.model
    model.train()

    criterion = build_criterion(args.loss, len(AZ_CLASSES))
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = -1.0
    best_path = checkpoints_dir / f"{run_name}_best.pth"
    history = []

    for epoch in range(args.epochs):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(-1) == y).sum().item()
            train_total += x.size(0)

        train_loss /= max(1, train_total)
        train_acc = train_correct / max(1, train_total)

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                val_correct += (logits.argmax(-1) == y).sum().item()
                val_total += x.size(0)
        val_loss /= max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        scheduler.step()
        epoch_time = time.time() - t0

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_accuracy": round(val_acc, 4),
            "learning_rate": round(optimizer.param_groups[0]["lr"], 6),
            "epoch_time_sec": round(epoch_time, 1),
        })

        log.info(f"Epoch {epoch + 1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                 f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            log.info(f"  -> new best: {best_path} acc={best_acc:.4f}")

    # Save training log CSV
    csv_path = logs_dir / f"{run_name}_train_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    # Loss and accuracy curves
    _plot_curves(history, figures_dir, run_name)

    log.info(f"Best validation accuracy: {best_acc:.4f}")
    log.info(f"Checkpoint: {best_path}")


def _plot_curves(history: list, figures_dir: Path, run_name: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_accuracy"] for h in history]
    val_acc = [h["val_accuracy"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_loss, label="Train Loss")
    ax1.plot(epochs, val_loss, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss Curve — {run_name}")
    ax1.legend()

    ax2.plot(epochs, train_acc, label="Train Accuracy")
    ax2.plot(epochs, val_acc, label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Accuracy Curve — {run_name}")
    ax2.legend()
    ax2.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(str(figures_dir / f"{run_name}_loss_curve.png"), dpi=150)
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, val_acc, "b-o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title(f"Accuracy Curve — {run_name}")
    ax.set_ylim(0, 1.05)
    fig2.tight_layout()
    fig2.savefig(str(figures_dir / f"{run_name}_accuracy_curve.png"), dpi=150)
    plt.close(fig2)


if __name__ == "__main__":
    main()
