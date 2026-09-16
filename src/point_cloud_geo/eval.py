"""Eval harness: run pipeline and emit a small JSON-friendly report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from point_cloud_geo.config import load_config
from point_cloud_geo.train import train_eval


def run_pipeline(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Train/eval and return a serializable summary."""
    cfg = load_config() if cfg is None else dict(cfg)
    result = train_eval(cfg)
    summary = {
        "train_acc": result["train_acc"],
        "test_acc": result["test_acc"],
        "n_train": result["n_train"],
        "n_test": result["n_test"],
        "n_features": result["n_features"],
        "classes": result["classes"],
        "classifier": cfg.get("classifier", "mlp"),
        "fps_points": cfg.get("fps_points", 64),
        "knn_k": cfg.get("knn_k", 8),
        "voxel_size": cfg.get("voxel_size", 0.25),
        "disclaimer": (
            "OSS/learning demo only — not employer production software."
        ),
    }
    return summary


def write_report(summary: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
