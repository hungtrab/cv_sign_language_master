"""Train a PyTorch landmark MLP with per-epoch loss logging.

Outputs loss/accuracy curves for the report.

Usage:
    python train_landmark_pytorch.py --features data/landmarks_fast.csv --epochs 100
    python train_landmark_pytorch.py --features data/landmarks_fast.csv --loss focal --epochs 100
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from asl.utils.logging import get_logger

log = get_logger("asl.train_landmark")


class LandmarkMLP(nn.Module):
    def __init__(self, input_dim: int = 42, num_classes: int = 26,
                 hidden: tuple = (256, 128, 64), dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def load_dataset(path: str):
    labels, features = [], []
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header and "f0" in header:
            feat_start = header.index("f0")
            hand_col = header.index("hand_detected") if "hand_detected" in header else None
        else:
            feat_start = 1
            hand_col = None
            if header:
                try:
                    features.append([float(v) for v in header[feat_start:]])
                    labels.append(header[0].upper())
                except ValueError:
                    pass

        for row in reader:
            if not row:
                continue
            if hand_col is not None and row[hand_col].lower() != "true":
                continue
            lbl = row[0].upper()
            if len(lbl) == 1 and lbl.isalpha():
                labels.append(lbl)
                features.append([float(v) for v in row[feat_start:]])

    return np.array(features, dtype=np.float32), np.array(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/landmarks_fast.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", default="ce",
                        choices=["ce", "ce_label_smoothing", "focal", "weighted_ce"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}, Loss: {args.loss}")

    X, y = load_dataset(args.features)
    log.info(f"Loaded {len(X)} samples, {len(set(y))} classes")

    # Encode labels
    classes = sorted(set(y))
    cls2idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([cls2idx[c] for c in y])
    num_classes = len(classes)

    # Train/val split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_idx, test_size=0.2, random_state=args.seed, stratify=y_idx)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = LandmarkMLP(input_dim=X.shape[1], num_classes=num_classes).to(device)

    # Loss function
    if args.loss == "ce":
        criterion = nn.CrossEntropyLoss()
    elif args.loss == "ce_label_smoothing":
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    elif args.loss == "focal":
        criterion = FocalLoss(gamma=2.0)
    elif args.loss == "weighted_ce":
        counts = np.bincount(y_idx, minlength=num_classes).astype(np.float32)
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * num_classes
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device))

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_name = f"landmark_mlp_{args.loss}"
    history = []
    best_acc = 0.0
    ckpt_dir = Path(args.output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{run_name}_best.pt"

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            train_correct += (logits.argmax(-1) == yb).sum().item()
            train_total += xb.size(0)

        # Val
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                val_correct += (logits.argmax(-1) == yb).sum().item()
                val_total += xb.size(0)

        scheduler.step()
        tl = train_loss / max(1, train_total)
        ta = train_correct / max(1, train_total)
        vl = val_loss / max(1, val_total)
        va = val_correct / max(1, val_total)

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(tl, 5),
            "val_loss": round(vl, 5),
            "train_accuracy": round(ta, 4),
            "val_accuracy": round(va, 4),
            "learning_rate": round(optimizer.param_groups[0]["lr"], 6),
        })

        if va > best_acc:
            best_acc = va
            torch.save({
                "model": model.state_dict(),
                "classes": classes,
                "input_dim": X.shape[1],
                "num_classes": num_classes,
                "epoch": epoch + 1,
                "val_accuracy": va,
            }, best_path)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            log.info(f"Epoch {epoch+1:3d}: train_loss={tl:.4f} val_loss={vl:.4f} "
                     f"train_acc={ta:.4f} val_acc={va:.4f}")

    log.info(f"Best val accuracy: {best_acc:.4f} -> {best_path}")

    # Save history
    logs_dir = Path(args.output_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = logs_dir / f"{run_name}_train_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    # Plot curves
    _plot_curves(history, Path(args.output_dir) / "figures", run_name)
    log.info(f"Curves saved to {args.output_dir}/figures/")


def _plot_curves(history, figures_dir, run_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]
    tl = [h["train_loss"] for h in history]
    vl = [h["val_loss"] for h in history]
    ta = [h["train_accuracy"] for h in history]
    va = [h["val_accuracy"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, tl, label="Train Loss")
    ax1.plot(epochs, vl, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss — {run_name}")
    ax1.legend()

    ax2.plot(epochs, ta, label="Train Acc")
    ax2.plot(epochs, va, label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Accuracy — {run_name}")
    ax2.set_ylim(0, 1.05)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(str(figures_dir / f"{run_name}_curves.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
