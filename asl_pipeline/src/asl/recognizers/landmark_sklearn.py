from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from .base import BaseRecognizer, Prediction
from ..representations.base import RepresentationOutput

AZ_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class LandmarkSVMRecognizer(BaseRecognizer):
    name = "landmark_svm"
    input_type = "features"

    def __init__(self, *, model_path: Optional[str] = None, train_csv: Optional[str] = None):
        self.class_names = AZ_CLASSES
        if model_path and Path(model_path).exists():
            with open(model_path, "rb") as f:
                self._model = pickle.load(f)
            if hasattr(self._model, "classes_"):
                self.class_names = [str(c).upper() for c in self._model.classes_]
        elif train_csv and Path(train_csv).exists():
            self._model = self._train(train_csv)
        else:
            self._model = self._train("data/landmarks_fast.csv")

    def _train(self, csv_path: str):
        import csv
        from sklearn.svm import SVC
        labels, features = [], []
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            feat_start = header.index("f0") if header and "f0" in header else 1
            hand_col = header.index("hand_detected") if header and "hand_detected" in header else None
            for row in reader:
                if hand_col is not None and row[hand_col].lower() != "true":
                    continue
                lbl = row[0].upper()
                if len(lbl) == 1 and lbl.isalpha():
                    labels.append(lbl)
                    features.append([float(v) for v in row[feat_start:]])
        X = np.array(features, dtype=np.float32)
        y = np.array(labels)
        clf = SVC(probability=True, kernel="rbf", random_state=42)
        clf.fit(X, y)
        self.class_names = [str(c).upper() for c in clf.classes_]
        return clf

    def predict(self, rep_output: RepresentationOutput) -> Prediction:
        features = np.array(rep_output.data, dtype=np.float32).reshape(1, -1)
        probs = self._model.predict_proba(features)[0]
        idx = int(probs.argmax())
        top_idx = probs.argsort()[::-1][:5]
        top_k = [(self.class_names[i], float(probs[i])) for i in top_idx if i < len(self.class_names)]
        return Prediction(
            label=self.class_names[idx],
            confidence=float(probs[idx]),
            class_id=idx,
            top_k=top_k,
        )


class LandmarkRFRecognizer(BaseRecognizer):
    name = "landmark_rf"
    input_type = "features"

    def __init__(self, *, model_path: Optional[str] = None, train_csv: Optional[str] = None):
        self.class_names = AZ_CLASSES
        if model_path and Path(model_path).exists():
            with open(model_path, "rb") as f:
                self._model = pickle.load(f)
            if hasattr(self._model, "classes_"):
                self.class_names = [str(c).upper() for c in self._model.classes_]
        elif train_csv and Path(train_csv).exists():
            self._model = self._train(train_csv)
        else:
            self._model = self._train("data/landmarks_fast.csv")

    def _train(self, csv_path: str):
        import csv
        from sklearn.ensemble import RandomForestClassifier
        labels, features = [], []
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            feat_start = header.index("f0") if header and "f0" in header else 1
            hand_col = header.index("hand_detected") if header and "hand_detected" in header else None
            for row in reader:
                if hand_col is not None and row[hand_col].lower() != "true":
                    continue
                lbl = row[0].upper()
                if len(lbl) == 1 and lbl.isalpha():
                    labels.append(lbl)
                    features.append([float(v) for v in row[feat_start:]])
        X = np.array(features, dtype=np.float32)
        y = np.array(labels)
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X, y)
        self.class_names = [str(c).upper() for c in clf.classes_]
        return clf

    def predict(self, rep_output: RepresentationOutput) -> Prediction:
        features = np.array(rep_output.data, dtype=np.float32).reshape(1, -1)
        probs = self._model.predict_proba(features)[0]
        idx = int(probs.argmax())
        top_idx = probs.argsort()[::-1][:5]
        top_k = [(self.class_names[i], float(probs[i])) for i in top_idx if i < len(self.class_names)]
        return Prediction(
            label=self.class_names[idx],
            confidence=float(probs[idx]),
            class_id=idx,
            top_k=top_k,
        )
