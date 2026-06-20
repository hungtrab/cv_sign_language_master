from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    output_path: str | Path,
    *,
    title: str = "Confusion Matrix",
    figsize: tuple[int, int] = (12, 10),
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    n = len(class_names)
    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=6)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


def save_predictions_csv(
    predictions: Sequence[str],
    ground_truths: Sequence[str],
    output_path: str | Path,
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ground_truth", "prediction", "correct"])
        for gt, pred in zip(ground_truths, predictions):
            writer.writerow([gt, pred, int(gt == pred)])
