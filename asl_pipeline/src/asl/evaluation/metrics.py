from __future__ import annotations

import numpy as np


def accuracy(preds, labels) -> float:
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return float((preds == labels).mean())


def top_k_accuracy(logits: np.ndarray, labels: np.ndarray, *, k: int = 5) -> float:
    topk = np.argsort(-logits, axis=-1)[:, :k]
    correct = (topk == labels[:, None]).any(-1)
    return float(correct.mean())


def confusion_matrix(preds, labels, *, num_classes: int) -> np.ndarray:
    preds = np.asarray(preds, dtype=int)
    labels = np.asarray(labels, dtype=int)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for p, t in zip(preds, labels):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def macro_f1(preds, labels, *, num_classes: int) -> float:
    cm = confusion_matrix(preds, labels, num_classes=num_classes)
    f1s = []
    for c in range(num_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def per_class_f1(preds, labels, *, num_classes: int) -> list[float]:
    cm = confusion_matrix(preds, labels, num_classes=num_classes)
    f1s = []
    for c in range(num_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return f1s


def per_class_precision(preds, labels, *, num_classes: int) -> list[float]:
    cm = confusion_matrix(preds, labels, num_classes=num_classes)
    precs = []
    for c in range(num_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        precs.append(tp / (tp + fp) if (tp + fp) else 0.0)
    return precs


def per_class_recall(preds, labels, *, num_classes: int) -> list[float]:
    cm = confusion_matrix(preds, labels, num_classes=num_classes)
    recs = []
    for c in range(num_classes):
        tp = int(cm[c, c])
        fn = int(cm[c, :].sum() - tp)
        recs.append(tp / (tp + fn) if (tp + fn) else 0.0)
    return recs


def weighted_f1(preds, labels, *, num_classes: int) -> float:
    cm = confusion_matrix(preds, labels, num_classes=num_classes)
    f1s = []
    supports = []
    for c in range(num_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        support = int(cm[c, :].sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
        supports.append(support)
    total = sum(supports)
    if total == 0:
        return 0.0
    return float(sum(f * s for f, s in zip(f1s, supports)) / total)
