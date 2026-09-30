"""Teaching ICP (iterative closest point) registration in numpy.

``icp(source, target, ...)`` estimates a rigid 4×4 transform ``T`` that maps
``source`` onto ``target`` (``target ≈ R @ source + t``). Two update rules:

- ``point_to_point``: closed-form Kabsch / SVD on the inlier pairs, minimizing
  Σ‖R pᵢ + t − qᵢ‖².
- ``point_to_plane``: linearized least squares on Σ((R pᵢ + t − qᵢ)·nᵢ)², using
  target normals from the shipped radius-PCA estimator
  (:func:`point_cloud_geo.normals_radius.estimate_normals_radius`). A 6×6
  solve gives (ω, t), and ω is mapped back to an exact rotation (Rodrigues).

Output definitions. These are spelled out because Open3D users found the
upstream docs ambiguous or wrong (isl-org/Open3D#7503, #7296, #7367):

- ``fitness = n_inliers / len(source)``. The fraction of **source** points
  whose nearest target point (after applying the returned ``T``) lies within
  ``max_corr_dist``. The range is 0…1 and higher is better. It is *not* the
  fraction of target points, and not an overlap IoU.
- ``inlier_rmse = sqrt(mean(dᵢ²))`` over **inlier** correspondences only,
  where dᵢ is the Euclidean point-to-point distance (for both methods). It is
  ``nan`` when there are no inliers.
- ``converged`` is True only if a stopping test fired **before**
  ``max_iter``, either because the incremental update was tiny
  (rotation < ``tol_rotation_deg`` and ‖Δt‖ < ``tol_translation``) or
  because both |Δfitness| < ``tol_fitness`` and |Δrmse| < ``tol_rmse``
  between consecutive iterations. Hitting ``max_iter`` means
  ``converged=False`` even if the alignment looks fine.

Both metrics are always recomputed at the *returned* ``T``. Neither one
certifies the transform is **correct**. A symmetric shape or a bad start can
converge to a wrong local minimum with high fitness (see README and tests).

Teaching ICP only. There is no global registration (FPFH/RANSAC), no robust
kernels, and it is not Open3D or PCL. Refs:
https://learnopencv.com/iterative-closest-point-icp-explained/ ·
https://stackoverflow.com/questions/69490955/how-to-implement-icp-with-point-to-plane-distance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from sklearn.neighbors import NearestNeighbors

from point_cloud_geo.normals_radius import estimate_normals_radius

Method = Literal["point_to_point", "point_to_plane"]
VALID_METHODS: tuple[str, ...] = ("point_to_point", "point_to_plane")


# --------------------------------------------------------------------- SE(3)
def _check_points(name: str, pts: np.ndarray) -> np.ndarray:
    arr = np.asarray(pts, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 3:
        raise ValueError(f"{name} must be (N>=3, 3); got {arr.shape}")
    return arr


def apply_transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4×4 rigid transform to (N, 3) points."""
    T = np.asarray(T, dtype=np.float64)
    return np.asarray(points, dtype=np.float64) @ T[:3, :3].T + T[:3, 3]


def rotation_about_axis(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues: 3×3 rotation by ``angle_rad`` about ``axis``."""
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-15 or abs(angle_rad) < 1e-15:
        return np.eye(3)
    k = axis / norm
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle_rad) * K + (1.0 - np.cos(angle_rad)) * (K @ K)


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def random_rigid_transform(
    rng: np.random.Generator,
    angle_deg: float,
    max_translation: float = 0.2,
) -> np.ndarray:
    """Rotation of exactly ``angle_deg`` about a random axis + uniform translation."""
    axis = rng.normal(size=3)
    t = rng.uniform(-max_translation, max_translation, size=3)
    return make_transform(rotation_about_axis(axis, np.deg2rad(angle_deg)), t)


def rotation_error_deg(R_est: np.ndarray, R_true: np.ndarray) -> float:
    """Geodesic angle (degrees) between two rotations (atan2 form: precise near 0)."""
    D = np.asarray(R_est).T @ np.asarray(R_true)
    c = (np.trace(D) - 1.0) / 2.0
    s = 0.5 * np.linalg.norm([D[2, 1] - D[1, 2], D[0, 2] - D[2, 0], D[1, 0] - D[0, 1]])
    return float(np.rad2deg(np.arctan2(s, c)))


def transform_errors(T_est: np.ndarray, T_true: np.ndarray) -> dict[str, float]:
    return {
        "rotation_error_deg": rotation_error_deg(T_est[:3, :3], T_true[:3, :3]),
        "translation_error": float(np.linalg.norm(T_est[:3, 3] - T_true[:3, 3])),
    }


# ------------------------------------------------------------------ solvers
def kabsch(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares rigid transform (4×4) mapping ``src`` → ``dst`` (paired rows)."""
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    H = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0  # guard against reflections
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return make_transform(R, mu_d - R @ mu_s)


def point_to_plane_step(
    src: np.ndarray,
    dst: np.ndarray,
    normals: np.ndarray,
    *,
    rcond: float = 1e-2,
) -> np.ndarray:
    """One linearized point-to-plane update (small-angle), returned as an exact rigid 4×4.

    ``rcond`` truncates near-degenerate directions of the 6×6 system. On a flat
    plane every normal is the same, so in-plane sliding and spin are
    *unobservable* (on a sphere, every rotation about the centre is). Without
    truncation, noise turns those null directions into huge, diverging steps.
    With truncation they are simply left alone, which is why point-to-plane
    "slides" on a plane (see README).
    """
    A = np.hstack([np.cross(src, normals), normals])  # (M, 6): [p×n, n]
    b = np.sum((dst - src) * normals, axis=1)  # (M,)
    x, *_ = np.linalg.lstsq(A, b, rcond=rcond)
    omega, t = x[:3], x[3:]
    angle = float(np.linalg.norm(omega))
    return make_transform(rotation_about_axis(omega, angle), t)


# --------------------------------------------------------------------- ICP
@dataclass
class ICPResult:
    """ICP output. See module docstring for exact metric definitions."""

    T: np.ndarray
    fitness: float
    inlier_rmse: float
    converged: bool
    n_iter: int
    method: str
    stop_reason: str  # converged_small_update | converged_metrics | max_iter | too_few_inliers
    n_inliers: int
    history: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "T": self.T.tolist(),
            "fitness": self.fitness,
            "inlier_rmse": self.inlier_rmse,
            "converged": self.converged,
            "n_iter": self.n_iter,
            "method": self.method,
            "stop_reason": self.stop_reason,
            "n_inliers": self.n_inliers,
            "history": list(self.history),
        }


def evaluate_registration(
    source: np.ndarray,
    target: np.ndarray,
    T: np.ndarray,
    max_corr_dist: float,
    *,
    _nn: NearestNeighbors | None = None,
) -> dict[str, Any]:
    """Score ``T`` with the same definitions ICP uses (fitness / inlier_rmse)."""
    src = _check_points("source", source)
    tgt = _check_points("target", target)
    nn = _nn if _nn is not None else NearestNeighbors(n_neighbors=1).fit(tgt)
    moved = apply_transform(src, T)
    dist, idx = nn.kneighbors(moved)
    dist, idx = dist[:, 0], idx[:, 0]
    inl = dist <= float(max_corr_dist)
    n_in = int(inl.sum())
    return {
        "fitness": n_in / float(len(src)),
        "inlier_rmse": float(np.sqrt(np.mean(dist[inl] ** 2))) if n_in else float("nan"),
        "n_inliers": n_in,
        "moved": moved,
        "idx": idx,
        "inliers": inl,
    }


def icp(
    source: np.ndarray,
    target: np.ndarray,
    *,
    max_corr_dist: float = 0.25,
    max_iter: int = 50,
    method: Method = "point_to_point",
    init: np.ndarray | None = None,
    target_normals: np.ndarray | None = None,
    normals_radius: float = 0.35,
    normals_min_nn: int = 4,
    plane_rcond: float = 1e-2,
    tol_fitness: float = 1e-6,
    tol_rmse: float = 1e-6,
    tol_rotation_deg: float = 1e-4,
    tol_translation: float = 1e-6,
) -> ICPResult:
    """Register ``source`` onto ``target``; returns :class:`ICPResult`.

    Deterministic: there is no randomness inside ICP (one NN per point, ties
    broken by sklearn's order), so the same inputs give the same result.
    """
    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}; got {method!r}")
    if max_corr_dist <= 0 or max_iter < 1:
        raise ValueError("max_corr_dist must be > 0 and max_iter >= 1")
    src = _check_points("source", source)
    tgt = _check_points("target", target)
    T = np.eye(4) if init is None else np.asarray(init, dtype=np.float64).copy()
    normals = None
    if method == "point_to_plane":
        if target_normals is None:
            normals, _, _ = estimate_normals_radius(tgt, normals_radius, min_nn=normals_min_nn)
        else:
            normals = np.asarray(target_normals, dtype=np.float64)
            if normals.shape != tgt.shape:
                raise ValueError("target_normals must match target shape")
    min_pairs = 3 if method == "point_to_point" else 6
    nn = NearestNeighbors(n_neighbors=1).fit(tgt)

    history: list[dict[str, float]] = []
    converged = False
    stop_reason = "max_iter"
    n_iter = 0
    prev: dict[str, Any] | None = None
    for k in range(int(max_iter)):
        cur = evaluate_registration(src, tgt, T, max_corr_dist, _nn=nn)
        history.append(
            {"iter": k, "fitness": cur["fitness"], "inlier_rmse": cur["inlier_rmse"]}
        )
        if cur["n_inliers"] < min_pairs:
            stop_reason = "too_few_inliers"
            break
        if prev is not None and (
            abs(cur["fitness"] - prev["fitness"]) < tol_fitness
            and abs(cur["inlier_rmse"] - prev["inlier_rmse"]) < tol_rmse
        ):
            converged, stop_reason = True, "converged_metrics"
            break
        inl = cur["inliers"]
        p = cur["moved"][inl]
        q = tgt[cur["idx"][inl]]
        if method == "point_to_point":
            dT = kabsch(p, q)
        else:
            dT = point_to_plane_step(
                p, q, normals[cur["idx"][inl]], rcond=plane_rcond  # type: ignore[index]
            )
        T = dT @ T
        n_iter = k + 1
        d_rot = rotation_error_deg(dT[:3, :3], np.eye(3))
        d_trans = float(np.linalg.norm(dT[:3, 3]))
        if d_rot < tol_rotation_deg and d_trans < tol_translation:
            converged, stop_reason = True, "converged_small_update"
            break
        prev = cur

    final = evaluate_registration(src, tgt, T, max_corr_dist, _nn=nn)
    return ICPResult(
        T=T,
        fitness=final["fitness"],
        inlier_rmse=final["inlier_rmse"],
        converged=converged,
        n_iter=n_iter,
        method=method,
        stop_reason=stop_reason,
        n_inliers=final["n_inliers"],
        history=history,
    )


# ------------------------------------------------------------- demo / eval
def make_registration_pair(
    shape: str = "cube",
    *,
    n_target: int = 800,
    n_source: int = 200,
    angle_deg: float = 20.0,
    max_translation: float = 0.2,
    noise: float = 0.01,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic pair with a **known** transform: returns ``(source, target, T_true)``.

    Sample ``n_target`` points of one synthetic surface (``plane`` / ``sphere`` /
    ``cube``) and centre them. The target is those points moved by ``T_true``
    plus Gaussian ``noise``. The source is a random ``n_source`` subset of the
    *un-moved* points, so true correspondences exist up to noise, and
    ``icp(source, target)`` should recover ``T_true``.
    """
    from point_cloud_geo.data import sample_cloud

    if n_source > n_target:
        raise ValueError("n_source must be <= n_target")
    rng = np.random.default_rng(seed)
    dense = sample_cloud(shape, n_points=n_target, noise=0.0, rng=rng)  # type: ignore[arg-type]
    dense = dense - dense.mean(axis=0)
    src = dense[rng.choice(n_target, size=n_source, replace=False)]
    T_true = random_rigid_transform(rng, angle_deg, max_translation)
    tgt = apply_transform(dense, T_true) + rng.normal(0.0, noise, size=dense.shape)
    return src, tgt, T_true


def icp_recovery(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Eval row: recover a known transform with both methods (+ a large-angle case)."""
    rc = dict((cfg or {}).get("registration") or {})
    seed = int((cfg or {}).get("seed", 7))
    shape = str(rc.get("shape", "cube"))
    kwargs = dict(
        max_corr_dist=float(rc.get("max_corr_dist", 0.25)),
        max_iter=int(rc.get("max_iter", 50)),
        normals_radius=float(rc.get("normals_radius", 0.35)),
    )
    rows = []
    cases = [("small", float(rc.get("angle_deg", 20.0))),
             ("large", float(rc.get("large_angle_deg", 90.0)))]
    for case, angle in cases:
        src, tgt, T_true = make_registration_pair(
            shape,
            angle_deg=angle,
            max_translation=float(rc.get("max_translation", 0.2)),
            noise=float(rc.get("noise", 0.01)),
            seed=seed,
        )
        for method in VALID_METHODS:
            res = icp(src, tgt, method=method, **kwargs)  # type: ignore[arg-type]
            rows.append(
                {
                    "case": case,
                    "angle_deg": angle,
                    "method": method,
                    "fitness": res.fitness,
                    "inlier_rmse": res.inlier_rmse,
                    "converged": res.converged,
                    "n_iter": res.n_iter,
                    "stop_reason": res.stop_reason,
                    **transform_errors(res.T, T_true),
                }
            )
    return {"shape": shape, "icp_recovery": rows}


__all__ = [
    "ICPResult",
    "VALID_METHODS",
    "apply_transform",
    "evaluate_registration",
    "icp",
    "icp_recovery",
    "kabsch",
    "make_registration_pair",
    "make_transform",
    "point_to_plane_step",
    "random_rigid_transform",
    "rotation_about_axis",
    "rotation_error_deg",
    "transform_errors",
]
