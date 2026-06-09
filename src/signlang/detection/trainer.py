"""Wrapper around the Ultralytics YOLO training API.

Why a wrapper? So we keep the same YAML-driven workflow as the rest of
the project, and so the training code can be unit-tested without
actually loading 2 GB of weights.
"""

from __future__ import annotations

from pathlib import Path


def train(cfg) -> Path:
    """Train YOLOv8 from ``cfg`` and return the best-weights path."""
    from ultralytics import YOLO

    model = YOLO(str(cfg.model.weights))
    out_dir = Path(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = model.train(
        data=str(cfg.paths.data_yaml),
        epochs=int(cfg.training.epochs),
        imgsz=int(cfg.model.image_size),
        batch=int(cfg.training.batch_size),
        device=cfg.training.device,
        patience=int(cfg.training.patience),
        optimizer=cfg.training.optimizer,
        lr0=float(cfg.training.lr0),
        momentum=float(cfg.training.momentum),
        weight_decay=float(cfg.training.weight_decay),
        workers=int(cfg.training.workers),
        amp=bool(cfg.training.amp),
        project=str(out_dir.parent),
        name=out_dir.name,
        exist_ok=True,
    )

    # Ultralytics writes ``best.pt`` and ``last.pt`` under the run dir.
    return Path(results.save_dir) / "weights" / "best.pt"
