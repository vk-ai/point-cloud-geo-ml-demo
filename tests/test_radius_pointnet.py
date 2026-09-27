"""Radius PCA normals + PointNet-lite tests."""

from __future__ import annotations

import numpy as np

from point_cloud_geo.data import make_dataset, sample_plane, sample_sphere
from point_cloud_geo.normals_radius import estimate_normals_radius
from point_cloud_geo.pointnet_lite import (
    PointNetLite,
    cloud_to_point_features,
    compare_normals_ablation,
    run_pointnet_lite_demo,
)
from point_cloud_geo.config import load_config
from point_cloud_geo.eval import run_pipeline


def test_radius_normals_unit_on_plane():
    rng = np.random.default_rng(0)
    # Axis-aligned plane (no rigid) for clean normals
    pts = sample_plane(80, rng, noise=0.002)
    # sample_plane applies rigid — use raw grid instead
    xy = rng.uniform(-1, 1, size=(100, 2))
    z = rng.normal(0, 0.002, size=(100, 1))
    plane = np.concatenate([xy, z], axis=1)
    normals, eigs, counts = estimate_normals_radius(plane, radius=0.4, min_nn=5)
    assert normals.shape == (100, 3)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)
    assert counts.min() >= 5
    # Plane normals should mostly align with ±z
    abs_z = np.abs(normals[:, 2]).mean()
    assert abs_z > 0.7


def test_radius_normals_on_sphere_vary():
    rng = np.random.default_rng(1)
    # Unit sphere without rigid
    xyz = rng.normal(size=(120, 3))
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    # Orient toward a far viewpoint so normals are consistently radial
    normals, _, _ = estimate_normals_radius(
        xyz, radius=0.5, min_nn=6, orient_toward=np.array([0.0, 0.0, 10.0])
    )
    # |n · x| should be high (radial); sign depends on viewpoint hemisphere
    abs_dots = np.abs(np.sum(normals * xyz, axis=1))
    assert float(abs_dots.mean()) > 0.5


def test_pointnet_lite_trains():
    clouds, labels = make_dataset(n_per_class=12, n_points=64, seed=3)
    out = run_pointnet_lite_demo(
        clouds,
        labels,
        mode="xyz",
        n_points=32,
        hidden=16,
        epochs=25,
        lr=0.1,
        seed=3,
    )
    assert out["n_params"] > 0
    assert 0.0 <= out["test_acc"] <= 1.0
    assert out["train_acc"] >= 0.3  # soft — toy may vary


def test_normals_ablation_runs():
    clouds, labels = make_dataset(n_per_class=10, n_points=48, seed=4)
    table = compare_normals_ablation(
        clouds,
        labels,
        n_points=32,
        radius=0.4,
        hidden=16,
        epochs=20,
        lr=0.1,
        seed=4,
    )
    assert "xyz" in table and "xyz_normals" in table
    assert "delta_test_acc" in table


def test_pipeline_includes_pointnet_when_enabled():
    cfg = load_config()
    cfg["n_per_class"] = 12
    cfg["n_points"] = 48
    cfg["fps_points"] = 24
    cfg["pointnet_lite"] = {
        "enabled": True,
        "n_points": 24,
        "hidden": 16,
        "epochs": 15,
        "lr": 0.1,
        "compare_normals": True,
    }
    cfg["normals"] = {"radius": 0.4, "min_nn": 4}
    summary = run_pipeline(cfg)
    assert "pointnet_lite" in summary
    assert "xyz" in summary["pointnet_lite"]
