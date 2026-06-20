from __future__ import annotations

import numpy as np


def expected_calibration_error(confidences: np.ndarray, accuracies: np.ndarray,
                                n_bins: int = 15) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(confidences)
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc = accuracies[mask].mean()
        ece += (mask.sum() / total) * abs(avg_acc - avg_conf)
    return float(ece)


def reliability_diagram_data(confidences: np.ndarray, accuracies: np.ndarray,
                              n_bins: int = 15) -> dict:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accs = []
    bin_confs = []
    bin_counts = []
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        count = int(mask.sum())
        bin_counts.append(count)
        bin_centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
        if count > 0:
            bin_accs.append(float(accuracies[mask].mean()))
            bin_confs.append(float(confidences[mask].mean()))
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
    return {
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_counts": bin_counts,
        "ece": expected_calibration_error(confidences, accuracies, n_bins),
    }
