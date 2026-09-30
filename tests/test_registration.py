"""Round 4: numpy ICP (point-to-point + point-to-plane) with explicit output definitions."""

from __future__ import annotations

import numpy as np
import pytest

from point_cloud_geo.config import load_config
from point_cloud_geo.normals_radius import estimate_normals_radius
from point_cloud_geo.registration import (
    VALID_METHODS,
    apply_transform,
    evaluate_registration,
    icp,
    icp_recovery,
    kabsch,
    make_registration_pair,
    make_transform,
    point_to_plane_step,
    random_rigid_transform,
    rotation_about_axis,
    rotation_error_deg,
    transform_errors,
)

NOISE = 0.01


def test_se3_helpers_and_kabsch_exact():
    rng = np.random.default_rng(0)
    R = rotation_about_axis(np.array([1.0, 2.0, 3.0]), np.deg2rad(37.0))
    assert np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
    assert rotation_error_deg(R, np.eye(3)) == pytest.approx(37.0)
    T = random_rigid_transform(rng, 25.0, 0.3)
    assert rotation_error_deg(T[:3, :3], np.eye(3)) == pytest.approx(25.0)
    src = rng.normal(size=(40, 3))
    est = kabsch(src, apply_transform(src, T))
    assert transform_errors(est, T)["rotation_error_deg"] < 1e-6
    assert transform_errors(est, T)["translation_error"] < 1e-9


def test_point_to_plane_step_small_motion():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(200, 3))
    normals = rng.normal(size=(200, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    T = make_transform(rotation_about_axis(np.array([0, 0, 1.0]), np.deg2rad(0.5)), [0.01, 0, 0])
    step = point_to_plane_step(src, apply_transform(src, T), normals)
    assert transform_errors(step, T)["rotation_error_deg"] < 1e-2


@pytest.mark.parametrize("method", VALID_METHODS)
@pytest.mark.parametrize("angle", [10.0, 20.0, 30.0])
def test_icp_recovers_known_transform_on_cube(method, angle):
    src, tgt, T_true = make_registration_pair("cube", angle_deg=angle, noise=NOISE, seed=7)
    res = icp(src, tgt, method=method, max_corr_dist=0.25, max_iter=50)
    err = transform_errors(res.T, T_true)
    assert err["rotation_error_deg"] < 1.0
    assert err["translation_error"] < 0.01
    assert res.converged and res.stop_reason.startswith("converged")
    assert res.fitness > 0.99
    # inlier_rmse ≈ target noise (3-D Gaussian, σ=0.01 → ~0.017), far below corr dist
    assert res.inlier_rmse < 3 * NOISE


def test_point_to_plane_converges_in_fewer_iterations_on_cube():
    src, tgt, _ = make_registration_pair("cube", angle_deg=20.0, noise=NOISE, seed=7)
    p2p = icp(src, tgt, method="point_to_point")
    p2l = icp(src, tgt, method="point_to_plane")
    assert p2p.converged and p2l.converged
    assert p2l.n_iter < p2p.n_iter


def test_point_to_plane_accepts_precomputed_radius_normals():
    src, tgt, T_true = make_registration_pair("cube", angle_deg=20.0, noise=NOISE, seed=7)
    normals, _, _ = estimate_normals_radius(tgt, 0.35, min_nn=4)
    a = icp(src, tgt, method="point_to_plane", target_normals=normals)
    b = icp(src, tgt, method="point_to_plane")  # computes the same normals internally
    np.testing.assert_allclose(a.T, b.T)
    with pytest.raises(ValueError, match="target_normals"):
        icp(src, tgt, method="point_to_plane", target_normals=normals[:10])


def test_large_angle_local_minimum_is_reported_honestly():
    """Cube 90°: ICP 'converges' with fitness ≈ 1 to a symmetric but WRONG pose."""
    src, tgt, T_true = make_registration_pair("cube", angle_deg=90.0, noise=NOISE, seed=7)
    good_src, good_tgt, _ = make_registration_pair("cube", angle_deg=20.0, noise=NOISE, seed=7)
    good = icp(good_src, good_tgt)
    res = icp(src, tgt, method="point_to_point")
    assert transform_errors(res.T, T_true)["rotation_error_deg"] > 30.0
    assert res.fitness > 0.95  # fitness alone does not certify correctness...
    assert res.inlier_rmse > 3 * good.inlier_rmse  # ...but rmse is visibly worse


def test_sphere_rotation_is_unobservable_despite_perfect_fitness():
    src, tgt, T_true = make_registration_pair("sphere", angle_deg=30.0, noise=NOISE, seed=7)
    res = icp(src, tgt)
    assert res.fitness == pytest.approx(1.0)
    assert transform_errors(res.T, T_true)["rotation_error_deg"] > 10.0


def test_fitness_and_rmse_definitions_with_outliers():
    src, tgt, T_true = make_registration_pair("cube", angle_deg=15.0, noise=NOISE, seed=3)
    far = np.random.default_rng(3).uniform(20.0, 30.0, size=(50, 3))  # 50 junk points
    res_clean = icp(src, tgt)
    res = icp(np.vstack([src, far]), tgt)
    # fitness = inliers / len(SOURCE) → 200 / 250
    assert res.n_inliers == 200
    assert res.fitness == pytest.approx(200 / 250)
    # inlier_rmse only over inliers: junk points do not inflate it
    assert res.inlier_rmse == pytest.approx(res_clean.inlier_rmse, rel=1e-3)
    # metrics are recomputed at the returned T with the documented formula
    ev = evaluate_registration(np.vstack([src, far]), tgt, res.T, 0.25)
    assert ev["fitness"] == pytest.approx(res.fitness)
    assert ev["inlier_rmse"] == pytest.approx(res.inlier_rmse)


def test_converged_flag_and_stop_reasons():
    src, tgt, _ = make_registration_pair("cube", angle_deg=20.0, noise=NOISE, seed=7)
    capped = icp(src, tgt, max_iter=2)
    assert capped.converged is False and capped.stop_reason == "max_iter"
    assert capped.n_iter == 2 and len(capped.history) == 2
    far = icp(src, tgt + 100.0)
    assert far.converged is False and far.stop_reason == "too_few_inliers"
    assert far.fitness == 0.0 and np.isnan(far.inlier_rmse)
    same = icp(tgt[:200], tgt)  # already aligned
    assert same.converged and same.n_iter <= 2
    assert same.inlier_rmse < 1e-9 and same.fitness == 1.0


def test_icp_is_deterministic():
    src, tgt, _ = make_registration_pair("cube", angle_deg=20.0, seed=11)
    src2, tgt2, _ = make_registration_pair("cube", angle_deg=20.0, seed=11)
    np.testing.assert_array_equal(src, src2)
    for method in VALID_METHODS:
        a, b = icp(src, tgt, method=method), icp(src2, tgt2, method=method)
        np.testing.assert_array_equal(a.T, b.T)
        assert a.history == b.history


def test_icp_input_validation():
    pts = np.zeros((10, 3))
    with pytest.raises(ValueError, match="method"):
        icp(pts, pts, method="generalized")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        icp(pts[:, :2], pts)
    with pytest.raises(ValueError):
        icp(pts, pts, max_corr_dist=0.0)


def test_icp_recovery_eval_rows():
    rep = icp_recovery(load_config())
    rows = {(r["case"], r["method"]): r for r in rep["icp_recovery"]}
    assert set(rows) == {(c, m) for c in ("small", "large") for m in VALID_METHODS}
    for m in VALID_METHODS:
        assert rows[("small", m)]["rotation_error_deg"] < 1.0
        assert rows[("small", m)]["converged"] is True
    assert rows[("large", "point_to_point")]["rotation_error_deg"] > 30.0
