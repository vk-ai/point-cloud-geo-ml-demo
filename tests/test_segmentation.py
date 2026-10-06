"""Round 5: seeded (deterministic) RANSAC plane segmentation."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from point_cloud_geo.config import load_config
from point_cloud_geo.segmentation import (
    canonical_plane,
    fit_plane_lstsq,
    inlier_precision_recall,
    make_plane_scene,
    plane_angle_deg,
    plane_segmentation_eval,
    point_plane_distance,
    segment_plane,
    segment_planes,
)


@pytest.fixture(scope="module")
def ground_scene():
    # 400 points on z = 0.1 (σ=0.005) + 200 uniform outliers (33% of the cloud).
    return make_plane_scene([(0.0, 0.0, 1.0, -0.1)], n_per_plane=400, n_outliers=200,
                            noise=0.005, seed=3)


def test_recovers_plane_with_noise_and_outliers(ground_scene):
    pts, labels, (truth,) = ground_scene
    res = segment_plane(pts, distance_threshold=0.02, num_iterations=1000, seed=0)
    assert plane_angle_deg(res.plane_model, truth) < 1.0
    assert abs(res.plane_model[3] - truth[3]) < 0.01
    pr = inlier_precision_recall(res.inliers, labels, 0)
    assert pr["recall"] > 0.98 and pr["precision"] > 0.9
    assert res.inlier_rmse < 0.02
    assert np.isclose(res.fitness, len(res.inliers) / len(pts))


def test_same_seed_is_bit_identical(ground_scene):
    pts, _, _ = ground_scene
    a = segment_plane(pts, 0.02, num_iterations=500, seed=123)
    b = segment_plane(pts, 0.02, num_iterations=500, seed=123)
    assert a.plane_model.tobytes() == b.plane_model.tobytes()
    assert np.array_equal(a.inliers, b.inliers)
    assert a.n_iter == b.n_iter


def test_same_seed_identical_across_processes(ground_scene):
    # Guards against hidden global-RNG / hash-order dependence (Open3D#5647 class of bug).
    code = textwrap.dedent(
        """
        import hashlib
        from point_cloud_geo.segmentation import make_plane_scene, segment_plane
        pts, _, _ = make_plane_scene([(0, 0, 1, -0.1)], n_per_plane=400, n_outliers=200,
                                     noise=0.005, seed=3)
        r = segment_plane(pts, 0.02, num_iterations=500, seed=123)
        print(hashlib.sha256(r.plane_model.tobytes() + r.inliers.tobytes()).hexdigest())
        """
    )
    outs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       check=True, env={"PYTHONHASHSEED": h, "PATH": ""} | _pp()).stdout
        for h in ("0", "1")
    }
    assert len(outs) == 1


def _pp() -> dict:
    import os
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    return {"PYTHONPATH": os.pathsep.join([str(root / "src"), os.environ.get("PYTHONPATH", "")])}


def test_global_rng_is_not_used(ground_scene):
    pts, _, _ = ground_scene
    np.random.seed(0)
    a = segment_plane(pts, 0.02, num_iterations=200, seed=5)
    np.random.seed(999)
    b = segment_plane(pts, 0.02, num_iterations=200, seed=5)
    assert np.array_equal(a.inliers, b.inliers)
    state = np.random.get_state()[1].copy()
    segment_plane(pts, 0.02, num_iterations=50, seed=5)
    assert np.array_equal(np.random.get_state()[1], state)


def test_unpacks_like_open3d(ground_scene):
    pts, _, _ = ground_scene
    plane, inliers = segment_plane(pts, 0.02, seed=0)
    assert plane.shape == (4,) and np.isclose(np.linalg.norm(plane[:3]), 1.0)
    assert inliers.dtype == np.int64 and np.all(np.diff(inliers) > 0)


def test_canonical_sign():
    p = canonical_plane(np.array([0.0, 0.0, -2.0]), 0.4)
    assert np.allclose(p, [0.0, 0.0, 1.0, -0.2])
    q = canonical_plane(np.array([0.0, 0.0, 2.0]), -0.4)
    assert np.allclose(p, q)
    with pytest.raises(ValueError):
        canonical_plane(np.zeros(3), 1.0)


def test_lstsq_fit_and_distance():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-1, 1, size=(50, 2))
    pts = np.c_[xy, 0.5 * xy[:, 0] - 0.2]  # z = 0.5x − 0.2
    plane = fit_plane_lstsq(pts)
    assert np.max(point_plane_distance(pts, plane)) < 1e-12
    assert plane_angle_deg(plane, (0.5, 0.0, -1.0, -0.2)) < 1e-6


def test_early_stop_vs_exhaustive(ground_scene):
    pts, _, (truth,) = ground_scene
    fast = segment_plane(pts, 0.02, num_iterations=5000, seed=0)
    full = segment_plane(pts, 0.02, num_iterations=300, seed=0, probability=1.0)
    assert fast.n_iter < 5000  # adaptive stop fired (inlier ratio ≈ 0.67)
    assert full.n_iter == 300
    for r in (fast, full):
        assert plane_angle_deg(r.plane_model, truth) < 1.0


def test_refine_never_loses_support(ground_scene):
    pts, _, _ = ground_scene
    raw = segment_plane(pts, 0.02, num_iterations=200, seed=1, refine=False)
    ref = segment_plane(pts, 0.02, num_iterations=200, seed=1, refine=True)
    assert len(ref.inliers) >= len(raw.inliers)
    assert raw.refined is False


def test_ransac_n_larger_than_three(ground_scene):
    pts, _, (truth,) = ground_scene
    res = segment_plane(pts, 0.02, ransac_n=5, num_iterations=500, seed=0)
    assert plane_angle_deg(res.plane_model, truth) < 1.0


def test_degenerate_inputs():
    line = np.c_[np.linspace(0, 1, 20), np.zeros(20), np.zeros(20)]  # collinear
    res = segment_plane(line, 0.01, num_iterations=20, seed=0)
    assert len(res.inliers) == 0 and np.all(np.isnan(res.plane_model))
    with pytest.raises(ValueError):
        segment_plane(np.zeros((2, 3)), 0.01)
    with pytest.raises(ValueError):
        segment_plane(np.zeros((10, 2)), 0.01)
    with pytest.raises(ValueError):
        segment_plane(np.array([[0, 0, np.nan]] * 5), 0.01)
    for kw in ({"distance_threshold": 0}, {"ransac_n": 2}, {"num_iterations": 0},
               {"probability": 0.0}):
        with pytest.raises(ValueError):
            segment_plane(line, **{"distance_threshold": 0.01, **kw})


def test_segment_planes_two_planes_plus_outliers():
    planes = [(0.0, 0.0, 1.0, 0.0), (1.0, 0.0, 0.0, -1.0)]  # floor z=0, wall x=1
    pts, labels, truth = make_plane_scene(planes, n_per_plane=[500, 300], n_outliers=120,
                                          noise=0.004, centers=[(0, 0, 0), (1, 0, 0.8)],
                                          seed=11)
    found = segment_planes(pts, max_planes=3, min_inliers=80, distance_threshold=0.015,
                           seed=0)
    assert len(found) == 2  # the third attempt only finds < 80 outlier points
    for res, pid in zip(found, (0, 1)):  # bigger plane first
        assert plane_angle_deg(res.plane_model, truth[pid]) < 1.0
        assert inlier_precision_recall(res.inliers, labels, pid)["recall"] > 0.9
    assert not set(found[0].inliers) & set(found[1].inliers)


def test_scene_generator_shapes():
    pts, labels, truth = make_plane_scene([(0, 0, 1, 0)], n_per_plane=10, n_outliers=5, seed=0)
    assert pts.shape == (15, 3) and labels.tolist().count(-1) == 5
    assert np.allclose(truth[0], [0, 0, 1, 0])
    with pytest.raises(ValueError):
        make_plane_scene([(0, 0, 1, 0)], n_per_plane=[1, 2])


def test_eval_row():
    out = plane_segmentation_eval(load_config())["plane_segmentation"]
    assert out["deterministic_same_seed"] is True
    assert sorted(r["plane_id"] for r in out["planes"]) == [0, 1]
    for r in out["planes"]:
        assert r["normal_angle_err_deg"] < 2.0
        assert r["recall"] > 0.9 and r["precision"] > 0.9
    assert len(out["inliers_by_seed"]) == 5
