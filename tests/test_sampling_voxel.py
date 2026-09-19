"""Tests for FPS and voxelization."""

from __future__ import annotations

import numpy as np

from point_cloud_geo.data import sample_cloud
from point_cloud_geo.sampling import farthest_point_sampling, fps_downsample
from point_cloud_geo.voxelize import voxel_centroids, voxel_occupancy_features


def test_fps_count_and_subset():
    rng = np.random.default_rng(3)
    pts = sample_cloud("sphere", n_points=100, rng=rng)
    idx = farthest_point_sampling(pts, 20, seed=0)
    assert idx.shape == (20,)
    assert len(set(idx.tolist())) == 20
    sub = fps_downsample(pts, 20, seed=0)
    assert sub.shape == (20, 3)
    # Selected points must come from the original cloud
    for p in sub:
        assert np.any(np.all(np.isclose(pts, p), axis=1))


def test_fps_clamps_to_n():
    pts = np.random.default_rng(0).normal(size=(5, 3))
    idx = farthest_point_sampling(pts, 50, seed=1)
    assert idx.shape == (5,)


def test_voxel_occupancy_fixed_dim():
    pts = sample_cloud("cube", n_points=80, rng=np.random.default_rng(4))
    feat = voxel_occupancy_features(pts, voxel_size=0.25)
    assert feat.ndim == 1
    assert feat.size >= 8
    assert np.isclose(feat.sum(), 1.0) or feat.sum() == 0.0


def test_voxel_centroids_nonempty():
    pts = sample_cloud("plane", n_points=50, rng=np.random.default_rng(5))
    cents = voxel_centroids(pts, voxel_size=0.4)
    assert cents.ndim == 2 and cents.shape[1] == 3
    assert cents.shape[0] >= 1


def test_voxel_reduction_modes_differ_on_dense_vs_sparse():
    """occupancy | count | max_count_bin must disagree on a clumpy synthetic cloud."""
    from point_cloud_geo.voxelize import VALID_REDUCTIONS, voxel_occupancy_features

    rng = np.random.default_rng(21)
    # Dense cluster in one corner + sparse outliers → count/max peak diverge from binary occupancy
    dense = rng.normal(loc=(-0.8, -0.8, -0.8), scale=0.05, size=(80, 3))
    sparse = rng.normal(loc=(0.9, 0.9, 0.9), scale=0.15, size=(12, 3))
    pts = np.vstack([dense, sparse])

    feats = {
        mode: voxel_occupancy_features(pts, voxel_size=0.3, reduction=mode)
        for mode in VALID_REDUCTIONS
    }
    assert feats["occupancy"].shape == feats["count"].shape == feats["max_count_bin"].shape
    # Binary occupancy spreads mass across occupied bins; max_count_bin concentrates on one bin.
    assert not np.allclose(feats["occupancy"], feats["max_count_bin"])
    assert not np.allclose(feats["count"], feats["occupancy"])
    # max_count_bin should be (near) one-hot after normalization
    assert int(np.count_nonzero(feats["max_count_bin"])) == 1
    assert np.isclose(feats["max_count_bin"].sum(), 1.0)


def test_voxel_reduction_invalid_raises():
    from point_cloud_geo.voxelize import voxel_occupancy_features

    pts = np.zeros((4, 3))
    try:
        voxel_occupancy_features(pts, reduction="mean")
        raised = False
    except ValueError:
        raised = True
    assert raised
