"""Train a landmark-based MLP classifier for ASL A-Z.

Usage:
    python train_landmark_mlp.py --features data/landmarks.csv --output weights/landmark_mlp.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle

import numpy as np


def load_dataset(path: str):
    labels, features = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            labels.append(parts[0].upper())
            features.append([float(v) for v in parts[1:]])
    if not features:
        raise SystemExit(f"No samples found in {path}")
    return np.array(features, dtype=np.float32), np.array(labels)


def main():
    parser = argparse.ArgumentParser(description="Train landmark MLP classifier")
    parser.add_argument("--features", default="data/landmarks.csv",
                        help="CSV file: label,f0,f1,...,f41")
    parser.add_argument("--output", default="weights/landmark_mlp.pkl")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.features):
        raise SystemExit(f"Features file not found: {args.features}")

    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    X, y = load_dataset(args.features)
    az_mask = np.array([len(l) == 1 and l.isalpha() for l in y])
    X, y = X[az_mask], y[az_mask]
    print(f"Loaded {len(X)} A-Z samples across {len(set(y))} classes: {sorted(set(y))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=15,
        random_state=args.seed,
    )
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="weighted")
    print(f"\nTest accuracy: {acc:.4f}")
    print(f"Weighted F1:   {f1:.4f}\n")
    print(classification_report(y_test, pred, zero_division=0))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(clf, f)
    print(f"Saved model -> {args.output}")

    labels_path = os.path.join(os.path.dirname(args.output), "labels.pkl")
    with open(labels_path, "wb") as f:
        pickle.dump(sorted(set(y.tolist())), f)
    print(f"Saved labels -> {labels_path}")


if __name__ == "__main__":
    main()
