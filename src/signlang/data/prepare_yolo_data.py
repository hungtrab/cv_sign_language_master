"""Prepare a HaGRID-derived hand dataset for YOLOv8.

Output layout (compatible with Ultralytics)::

    data/yolo/
        images/train/*.jpg
        images/val/*.jpg
        labels/train/*.txt          # YOLO format (one row per box)
        labels/val/*.txt
        dataset.yaml                # passed to ``yolo train data=...``

A row in a ``.txt`` label file is::

    class_id  cx_norm  cy_norm  w_norm  h_norm

We have a single class — ``hand``.

This module is intentionally minimal — a 3-person team can extend it to
support other annotation formats (COCO, VOC, etc.).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable

import yaml


def write_dataset_yaml(out_dir: Path, *, class_names: Iterable[str] = ("hand",)) -> Path:
    out = out_dir / "dataset.yaml"
    payload = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": dict(enumerate(class_names)),
    }
    with open(out, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return out


def convert_hagrid_subset(
    images_dir: Path,
    json_annotations: Path,
    *,
    out_root: Path,
    val_fraction: float = 0.15,
) -> None:
    """Convert a HaGRID-style ``annotations.json`` into YOLO labels.

    ``annotations.json`` is expected to map *image stem* -> dict containing
    a ``bboxes`` list of ``[x_min, y_min, w, h]`` in normalised coords.
    """
    with open(json_annotations) as f:
        ann = json.load(f)

    train_img = out_root / "images" / "train"
    val_img = out_root / "images" / "val"
    train_lbl = out_root / "labels" / "train"
    val_lbl = out_root / "labels" / "val"
    for d in (train_img, val_img, train_lbl, val_lbl):
        d.mkdir(parents=True, exist_ok=True)

    items = sorted(ann.items())
    n_val = int(len(items) * val_fraction)

    for i, (stem, meta) in enumerate(items):
        target_img = val_img if i < n_val else train_img
        target_lbl = val_lbl if i < n_val else train_lbl
        src = next(images_dir.glob(f"{stem}.*"), None)
        if src is None:
            continue
        shutil.copy(src, target_img / src.name)
        lines = []
        for box in meta.get("bboxes", []):
            x_min, y_min, w, h = box
            cx = x_min + w / 2.0
            cy = y_min + h / 2.0
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (target_lbl / f"{stem}.txt").write_text("\n".join(lines))

    write_dataset_yaml(out_root, class_names=("hand",))


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="Convert HaGRID JSON annotations into YOLOv8 layout.")
    p.add_argument("--images", required=True, type=Path)
    p.add_argument("--annotations", required=True, type=Path)
    p.add_argument("--output", default=Path("data/yolo"), type=Path)
    p.add_argument("--val-fraction", default=0.15, type=float)
    args = p.parse_args()
    convert_hagrid_subset(args.images, args.annotations, out_root=args.output,
                          val_fraction=args.val_fraction)
    print(f"Done. dataset.yaml -> {args.output / 'dataset.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
