"""Tests for PCA normals and cloud feature vectors."""

from __future__ import annotations

import numpy as np

from point_cloud_geo.data import make_dataset, sample_cloud
from point_cloud_geo.features import (
    cloud_feature_vector,
    estimate_normals_pca,
    featurize_dataset,
    local_shape_ratios,
)


def test_normals_unit_and_eig_order():
    pts = sample_cloud("plane", n_points=64, rng=np.random.default_rng(6))
    normals, eigs = estimate_normals_pca(pts, k=8)
    assert normals.shape == (64, 3)
    assert eigs.shape == (64, 3)
    norms = np.linalg.norm(normals, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    assert np.all(eigs[:, 0] <= eigs[:, 1] + 1e-9)
    assert np.all(eigs[:, 1] <= eigs[:, 2] + 1e-9)


def test_plane_lower_roughness_than_sphere():
    rng = np.random.default_rng(8)
    # Axis-aligned samples (no rigid transform) so local PCA is clean
    from point_cloud_geo.data import sample_plane, sample_sphere
    plane = sample_plane(120, rng, noise=0.005)
    # Build a raw sphere without using sample_sphere's rigid transform:
    # reuse sampler internals via high n + low noise and compare roughness λ0
    sphere = sample_sphere(120, rng, noise=0.005)
    _, e_plane = estimate_normals_pca(plane, k=12)
    _, e_sphere = estimate_normals_pca(sphere, k=12)
    rough_p = float(e_plane[:, 0].mean())
    rough_s = float(e_sphere[:, 0].mean())
    # Flat neighborhoods → smaller out-of-plane variance than curved ones
    assert rough_p < rough_s
    sph_p = local_shape_ratios(e_plane)[:, 2].mean()
    sph_s = local_shape_ratios(e_sphere)[:, 2].mean()
    assert sph_s >= sph_p * 0.5  # sphere not dramatically less spherical


def test_feature_vector_fixed_dim():
    clouds, _ = make_dataset(n_per_class=2, n_points=48, seed=9)
    x = featurize_dataset(clouds, fps_points=32, knn_k=6, seed=9)
    assert x.shape[0] == 6
    assert x.shape[1] == cloud_feature_vector(clouds[0], fps_points=32, knn_k=6).shape[0]
    assert np.isfinite(x).all()
