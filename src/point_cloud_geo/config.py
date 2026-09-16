"""Load YAML config with sensible defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_DEFAULTS: dict[str, Any] = {
    "seed": 7,
    "n_per_class": 40,
    "n_points": 128,
    "noise": 0.02,
    "fps_points": 64,
    "knn_k": 8,
    "voxel_size": 0.25,
    "test_size": 0.25,
    "classifier": "mlp",  # "mlp" | "logistic"
    "mlp_hidden": [32, 16],
    "max_iter": 400,
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg = dict(_DEFAULTS)
    if path is None:
        candidate = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
        path = candidate if candidate.is_file() else None
    if path is not None:
        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config at {path} must be a mapping")
        cfg.update(loaded)
    return cfg
