"""Config loading."""

from __future__ import annotations

from point_cloud_geo.config import load_config


def test_default_config_keys():
    cfg = load_config()
    for key in ("seed", "n_per_class", "fps_points", "knn_k", "classifier"):
        assert key in cfg
