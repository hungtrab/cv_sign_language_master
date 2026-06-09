"""Training loop for the second-stage letter classifier."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from ..data import build_dataloaders, build_eval_transform, build_train_transform
from ..utils import get_logger
from ..utils.config import load_class_names
from .models import build_classifier, count_parameters


@dataclass
class TrainHistory:
    epochs: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(-1) == labels).float().mean().item()


def train(cfg) -> Path:
    """Train the classifier defined by ``cfg`` and return the best-checkpoint path."""
    out_dir = Path(cfg.paths.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger("signlang.cls", log_file=out_dir / "train.log")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device={device}")

    class_names = load_class_names(cfg.paths.class_names)
    if len(class_names) != int(cfg.model.num_classes):
        raise ValueError(
            f"Number of class names ({len(class_names)}) does not match "
            f"cfg.model.num_classes ({cfg.model.num_classes})"
        )

    train_tf = build_train_transform(cfg)
    eval_tf = build_eval_transform(cfg)
    train_loader, val_loader = build_dataloaders(cfg, class_names,
                                                 train_tf=train_tf, eval_tf=eval_tf)

    model = build_classifier(cfg).to(device)
    trainable, total = count_parameters(model)
    log.info(f"model={cfg.model.arch}  params: trainable={trainable:,} / total={total:,}")

    optim = AdamW(model.parameters(), lr=float(cfg.training.learning_rate),
                  weight_decay=float(cfg.training.weight_decay))
    scheduler = CosineAnnealingLR(optim, T_max=int(cfg.training.epochs))
    scaler = torch.amp.GradScaler("cuda") if (cfg.training.amp and device.type == "cuda") else None

    history = TrainHistory()
    best_acc = -1.0
    best_path = out_dir / "best.pt"

    for epoch in range(int(cfg.training.epochs)):
        # ---- train -----------------------------------------------------
        model.train()
        running_loss = 0.0; running_n = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg.training.epochs}")
        for x, y in pbar:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optim); scaler.update()
            else:
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optim.step()
            running_loss += loss.item() * x.size(0); running_n += x.size(0)
            pbar.set_postfix(loss=running_loss / max(1, running_n))
        train_loss = running_loss / max(1, running_n)

        # ---- val -------------------------------------------------------
        model.eval()
        val_loss = 0.0; val_acc = 0.0; n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device); y = y.to(device)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                val_loss += loss.item() * x.size(0)
                val_acc += _accuracy(logits, y) * x.size(0)
                n += x.size(0)
        val_loss /= max(1, n); val_acc /= max(1, n)

        scheduler.step()
        log.info(f"epoch {epoch + 1}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        history.epochs.append(epoch + 1)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model": model.state_dict(),
                "arch": cfg.model.arch,
                "num_classes": int(cfg.model.num_classes),
                "class_names": class_names,
                "image_size": int(cfg.data.image_size),
            }, best_path)
            log.info(f"  -> new best: {best_path}  acc={best_acc:.4f}")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history.__dict__, f, indent=2)
    return best_path
