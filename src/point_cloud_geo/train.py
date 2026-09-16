"""Train / eval tiny geometric-feature classifier."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from point_cloud_geo.data import CLASS_NAMES, make_dataset
from point_cloud_geo.features import featurize_dataset
from point_cloud_geo.model import build_classifier


def train_eval(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    End-to-end: synthetic clouds → FPS + PCA normals → features → classifier.

    Returns a result dict with train/test accuracy and a fitted model.
    """
    from point_cloud_geo.config import load_config

    cfg = load_config() if cfg is None else dict(cfg)
    seed = int(cfg.get("seed", 7))

    clouds, labels = make_dataset(
        n_per_class=int(cfg.get("n_per_class", 40)),
        n_points=int(cfg.get("n_points", 128)),
        noise=float(cfg.get("noise", 0.02)),
        seed=seed,
    )
    x = featurize_dataset(
        clouds,
        fps_points=int(cfg.get("fps_points", 64)),
        knn_k=int(cfg.get("knn_k", 8)),
        voxel_size=float(cfg.get("voxel_size", 0.25)),
        seed=seed,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        labels,
        test_size=float(cfg.get("test_size", 0.25)),
        random_state=seed,
        stratify=labels,
    )

    model = build_classifier(
        cfg.get("classifier", "mlp"),  # type: ignore[arg-type]
        hidden=list(cfg.get("mlp_hidden", [32, 16])),
        max_iter=int(cfg.get("max_iter", 400)),
        seed=seed,
    )
    model.fit(x_train, y_train)

    y_tr = model.predict(x_train)
    y_te = model.predict(x_test)
    train_acc = float(accuracy_score(y_train, y_tr))
    test_acc = float(accuracy_score(y_test, y_te))
    report = classification_report(
        y_test,
        y_te,
        target_names=list(CLASS_NAMES),
        zero_division=0,
    )

    return {
        "model": model,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features": int(x.shape[1]),
        "classes": list(CLASS_NAMES),
        "report": report,
        "y_test": y_test,
        "y_pred": y_te,
    }
