"""Tiny classifiers: sklearn MLP / logistic on geometric features."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ClassifierName = Literal["mlp", "logistic"]


def build_classifier(
    name: ClassifierName = "mlp",
    *,
    hidden: list[int] | tuple[int, ...] = (32, 16),
    max_iter: int = 400,
    seed: int = 7,
) -> Pipeline:
    """
    StandardScaler + tiny classifier.

    - mlp: PointNet-adjacent idea — small MLP on pooled geometric features
      (here the pooling already happened in ``features.cloud_feature_vector``)
    - logistic: linear baseline on the same features
    """
    if name == "logistic":
        clf: Any = LogisticRegression(
            max_iter=max_iter,
            random_state=seed,
        )
    elif name == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=tuple(hidden),
            activation="relu",
            solver="lbfgs",
            max_iter=max_iter,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown classifier {name!r}; use 'mlp' or 'logistic'")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def predict_proba_safe(model: Pipeline, x: np.ndarray) -> np.ndarray:
    """Predict class probabilities; fall back to one-hot of predict if needed."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)
    pred = model.predict(x)
    n_classes = int(np.max(pred)) + 1
    out = np.zeros((len(pred), n_classes), dtype=np.float64)
    out[np.arange(len(pred)), pred] = 1.0
    return out
