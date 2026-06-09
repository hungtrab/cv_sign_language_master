"""Dataset wrapper around the Kaggle ASL Alphabet folder layout.

Expected directory tree (passed as ``cfg.paths.data_dir``)::

    asl_alphabet_train/
        A/    *.jpg
        B/    *.jpg
        ...
        Z/    *.jpg
        space/ *.jpg
        del/   *.jpg
        nothing/ *.jpg

Internally we use :class:`torchvision.datasets.ImageFolder`, which
labels classes alphabetically and can therefore disagree with the
order in ``configs/asl_classes.txt``. We re-map labels to match the
configured order so the classifier output is consistent with the
class-names file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import ImageFolder


class ASLAlphabetDataset(Dataset):
    """ImageFolder wrapped with a label remap to match a canonical order."""

    def __init__(self, root: str | Path, *, class_names: Sequence[str], transform=None):
        super().__init__()
        self.inner = ImageFolder(root=str(root), transform=transform)
        self.class_names = list(class_names)

        # Build a mapping from ImageFolder's class index to our canonical index.
        canonical = {name: i for i, name in enumerate(self.class_names)}
        try:
            self.remap = [canonical[name] for name in self.inner.classes]
        except KeyError as exc:
            missing = exc.args[0]
            raise ValueError(
                f"Folder name '{missing}' not in configs/asl_classes.txt — "
                f"folders found: {self.inner.classes}"
            ) from exc

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int):
        x, y = self.inner[idx]
        return x, self.remap[y]


def build_dataloaders(
    cfg, class_names: Sequence[str], *, train_tf, eval_tf,
) -> Tuple[DataLoader, DataLoader]:
    """Train/val DataLoaders with a deterministic stratified-by-shuffle split."""
    full_train = ASLAlphabetDataset(cfg.paths.data_dir, class_names=class_names, transform=train_tf)
    val_frac = float(cfg.data.val_split)
    val_size = int(round(val_frac * len(full_train)))
    train_size = len(full_train) - val_size

    g = torch.Generator().manual_seed(int(cfg.seed))
    train_subset, val_subset = random_split(full_train, [train_size, val_size], generator=g)

    # Replace transform on val subset so it uses the eval pipeline.
    val_subset.dataset = ASLAlphabetDataset(  # type: ignore[attr-defined]
        cfg.paths.data_dir, class_names=class_names, transform=eval_tf,
    )

    return (
        DataLoader(train_subset, batch_size=int(cfg.training.batch_size), shuffle=True,
                   num_workers=int(cfg.training.num_workers), pin_memory=True),
        DataLoader(val_subset, batch_size=int(cfg.training.batch_size), shuffle=False,
                   num_workers=int(cfg.training.num_workers), pin_memory=True),
    )
