from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder

AZ_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class ASLAlphabetDataset(Dataset):
    """ImageFolder wrapped with label remap to match a canonical A-Z order."""

    def __init__(self, root: str | Path, *, class_names: Sequence[str] = AZ_CLASSES,
                 transform=None, filter_to_az: bool = True):
        super().__init__()
        self.class_names = list(class_names)
        self.inner = ImageFolder(root=str(root), transform=transform)

        canonical = {name: i for i, name in enumerate(self.class_names)}

        if filter_to_az:
            self.remap = {}
            self.valid_indices = []
            for folder_idx, folder_name in enumerate(self.inner.classes):
                if folder_name in canonical:
                    self.remap[folder_idx] = canonical[folder_name]
            for idx in range(len(self.inner)):
                _, y = self.inner.samples[idx]
                if y in self.remap:
                    self.valid_indices.append(idx)
        else:
            self.valid_indices = list(range(len(self.inner)))
            self.remap = {}
            for folder_idx, folder_name in enumerate(self.inner.classes):
                if folder_name in canonical:
                    self.remap[folder_idx] = canonical[folder_name]

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        real_idx = self.valid_indices[idx]
        x, y = self.inner[real_idx]
        return x, self.remap[y]


def build_dataloaders(
    data_dir: str | Path,
    *,
    class_names: Sequence[str] = AZ_CLASSES,
    train_tf=None,
    eval_tf=None,
    val_split: float = 0.15,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    full = ASLAlphabetDataset(data_dir, class_names=class_names, transform=train_tf)
    val_size = int(round(val_split * len(full)))
    train_size = len(full) - val_size

    g = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full, [train_size, val_size], generator=g)

    val_dataset = ASLAlphabetDataset(data_dir, class_names=class_names, transform=eval_tf)
    val_subset.dataset = val_dataset

    return (
        DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=True),
        DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=True),
    )
