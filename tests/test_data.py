"""Tests for synthetic point-cloud generation."""

from __future__ import annotations

import numpy as np

from point_cloud_geo.data import CLASS_NAMES, make_dataset, sample_cloud


def test_sample_shapes_and_finite():
    rng = np.random.default_rng(1)
    for name in CLASS_NAMES:
        pts = sample_cloud(name, n_points=64, rng=rng)
        assert pts.shape == (64, 3)
        assert np.isfinite(pts).all()


def test_make_dataset_labels():
    clouds, labels = make_dataset(n_per_class=5, n_points=32, seed=2)
    assert len(clouds) == 15
    assert labels.shape == (15,)
    assert set(labels.tolist()) == {0, 1, 2}
    assert all(c.shape == (32, 3) for c in clouds)


def test_int_class_index():
    pts = sample_cloud(1, n_points=16, rng=np.random.default_rng(0))
    assert pts.shape == (16, 3)
