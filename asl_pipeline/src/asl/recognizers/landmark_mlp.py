from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

from .base import BaseRecognizer, Prediction
from ..representations.base import RepresentationOutput
from ..utils.logging import get_logger

log = get_logger("asl.landmark_mlp")

AZ_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class LandmarkMLPRecognizer(BaseRecognizer):
    name = "landmark_mlp"
    input_type = "features"

    def __init__(self, *, model_path: Optional[str] = None,
                 model_json_path: Optional[str] = None):
        self._sklearn_model = None
        self._json_model = None
        self.class_names = AZ_CLASSES

        if model_path and Path(model_path).exists():
            self._load_sklearn(model_path)
        elif model_json_path and Path(model_json_path).exists():
            self._load_json(model_json_path)
        else:
            for candidate in [
                Path("model/classifier.pkl"),
                Path("weights/landmark_mlp.pkl"),
            ]:
                if candidate.exists():
                    self._load_sklearn(str(candidate))
                    break
            else:
                cv2_model = Path(__file__).resolve().parents[4] / "cv2" / "web" / "src" / "model.json"
                if cv2_model.exists():
                    self._load_json(str(cv2_model))
                else:
                    log.warning("No landmark MLP model found. Predictions will fail.")

    def _load_sklearn(self, path: str) -> None:
        with open(path, "rb") as f:
            self._sklearn_model = pickle.load(f)
        log.info(f"Loaded sklearn model from {path}")
        if hasattr(self._sklearn_model, "classes_"):
            self.class_names = [str(c).upper() for c in self._sklearn_model.classes_]

    def _load_json(self, path: str) -> None:
        with open(path) as f:
            self._json_model = json.load(f)
        self.class_names = [str(l).upper() for l in self._json_model["labels"]]
        log.info(f"Loaded JSON MLP from {path} ({len(self.class_names)} classes)")

    def predict(self, rep_output: RepresentationOutput) -> Prediction:
        features = rep_output.data
        if not isinstance(features, (list, np.ndarray)):
            raise ValueError(f"Expected feature vector, got {type(features)}")

        if self._sklearn_model is not None:
            return self._predict_sklearn(features)
        elif self._json_model is not None:
            return self._predict_json(features)
        else:
            raise RuntimeError("No model loaded")

    def _predict_sklearn(self, features) -> Prediction:
        features = np.array(features, dtype=np.float32).reshape(1, -1)
        if hasattr(self._sklearn_model, "predict_proba"):
            probs = self._sklearn_model.predict_proba(features)[0]
            idx = int(probs.argmax())
            conf = float(probs[idx])
            label = str(self._sklearn_model.classes_[idx]).upper()
        else:
            pred = self._sklearn_model.predict(features)[0]
            label = str(pred).upper()
            idx = self.class_names.index(label) if label in self.class_names else 0
            conf = 1.0
            probs = np.zeros(len(self.class_names))
            probs[idx] = 1.0

        top_idx = probs.argsort()[::-1][:5]
        top_k = [(self.class_names[i] if i < len(self.class_names) else str(i),
                   float(probs[i])) for i in top_idx]

        return Prediction(label=label, confidence=conf, class_id=idx, top_k=top_k)

    def _predict_json(self, features) -> Prediction:
        model = self._json_model
        x = list(features)

        # StandardScaler
        mean = model.get("scaler_mean")
        scale = model.get("scaler_scale")
        if mean and scale:
            x = [(x[i] - mean[i]) / (scale[i] or 1) for i in range(len(x))]

        # Forward pass: dense -> relu -> ... -> dense -> softmax
        layers = model["layers"]
        h = x
        for i, layer in enumerate(layers):
            w = layer["w"]
            b = layer["b"]
            n_out = len(b)
            out = list(b)
            for j in range(len(h)):
                if h[j] == 0:
                    continue
                row = w[j]
                for k in range(n_out):
                    out[k] += h[j] * row[k]
            if i < len(layers) - 1:
                out = [max(0, v) for v in out]
            h = out

        # Softmax
        max_val = max(h)
        exp_h = [np.exp(v - max_val) for v in h]
        total = sum(exp_h)
        probs = [v / total for v in exp_h]

        labels = self.class_names
        best = int(np.argmax(probs))
        top_idx = np.argsort(probs)[::-1][:5]
        top_k = [(labels[i] if i < len(labels) else str(i), probs[i]) for i in top_idx]

        return Prediction(
            label=labels[best] if best < len(labels) else str(best),
            confidence=probs[best],
            class_id=best,
            top_k=top_k,
        )
