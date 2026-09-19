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
    "voxel": {"reduction": "occupancy"},  # occupancy | count | max_count_bin
    "test_size": 0.25,
    "classifier": "mlp",  # "mlp" | "logistic"
    "mlp_hidden": [32, 16],
    "max_iter": 400,
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg = dict(_DEFAULTS)
    if isinstance(cfg.get("voxel"), dict):
        cfg["voxel"] = dict(cfg["voxel"])
    if path is None:
        candidate = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
        path = candidate if candidate.is_file() else None
    if path is not None:
        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config at {path} must be a mapping")
        # Shallow update, then deep-merge known nested maps (e.g. voxel.reduction).
        nested_keys = ("voxel",)
        nested_loaded = {k: loaded.pop(k) for k in list(loaded) if k in nested_keys}
        cfg.update(loaded)
        for key, val in nested_loaded.items():
            if isinstance(val, dict) and isinstance(cfg.get(key), dict):
                merged = dict(cfg[key])
                merged.update(val)
                cfg[key] = merged
            else:
                cfg[key] = val
    return cfg
