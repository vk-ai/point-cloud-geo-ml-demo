"""Seeded (deterministic) RANSAC plane segmentation in numpy.

``segment_plane(points, distance_threshold, ransac_n=3, num_iterations, seed)``
mirrors the shape of Open3D's ``PointCloud.segment_plane``: it returns the plane
model ``(a, b, c, d)`` with ``a·x + b·y + c·z + d = 0`` and the inlier indices.
It is **bit-identical for a given seed**. Upstream, determinism was the
recurring pain point: seeds stopped working in 0.16 (isl-org/Open3D#5647), and
two PRs were needed to make it deterministic again (#6308, #6580). Results still
looked random in 2025 (#7270).

How determinism is kept here:

- A private ``np.random.default_rng(seed)`` (never the global RNG) draws the
  minimal samples **sequentially** in a single thread, so there are no races
  between parallel workers.
- The best hypothesis has the most inliers. Ties go to the lower inlier RMSE,
  then to the earliest iteration, which gives a total order with no
  "whichever thread finished first".
- Adaptive early stopping (``probability``) depends only on that sequence.
- The plane sign is canonical: the normal is unit length and its largest-|·|
  component is positive, so the same plane always prints the same ``abcd``.

After RANSAC, the model is optionally **refined** by a least-squares (SVD) fit on
the inliers, and the inliers are recomputed once, as Open3D and PCL do.

Teaching code: numpy only, O(num_iterations · N), and not Open3D or PCL.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np


@dataclass
class PlaneResult:
    """Plane model + inliers. It unpacks like Open3D: ``plane, inliers = segment_plane(...)``."""

    plane_model: np.ndarray  # (4,) a, b, c, d with unit normal (a, b, c)
    inliers: np.ndarray  # (K,) sorted int64 indices into the input points
    fitness: float  # len(inliers) / len(points)
    inlier_rmse: float  # RMS point-to-plane distance over inliers (nan if none)
    n_iter: int  # RANSAC iterations actually run (≤ num_iterations)
    seed: int | None
    refined: bool = True

    def __iter__(self) -> Iterator[np.ndarray]:
        yield self.plane_model
        yield self.inliers

    def to_dict(self) -> dict[str, Any]:
        return {
            "plane_model": [float(v) for v in self.plane_model],
            "n_inliers": int(len(self.inliers)),
            "fitness": float(self.fitness),
            "inlier_rmse": float(self.inlier_rmse),
            "n_iter": int(self.n_iter),
            "seed": self.seed,
        }


def _check_points(points: np.ndarray, min_n: int) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"points must be (N, 3); got {arr.shape}")
    if arr.shape[0] < min_n:
        raise ValueError(f"need at least {min_n} points; got {arr.shape[0]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("points contain NaN/inf")
    return arr


def canonical_plane(normal: np.ndarray, d: float) -> np.ndarray:
    """Unit normal with its largest-|·| component positive → ``(a, b, c, d)``."""
    n = np.asarray(normal, dtype=np.float64)
    norm = float(np.linalg.norm(n))
    if norm < 1e-15:
        raise ValueError("degenerate plane normal")
    n = n / norm
    d = float(d) / norm
    if n[int(np.argmax(np.abs(n)))] < 0:
        n, d = -n, -d
    return np.array([n[0], n[1], n[2], d], dtype=np.float64)


def fit_plane_lstsq(points: np.ndarray) -> np.ndarray:
    """Total-least-squares plane through ``points`` (SVD of centred coords)."""
    pts = _check_points(points, 3)
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    normal = vt[-1]
    return canonical_plane(normal, -float(normal @ c))


def point_plane_distance(points: np.ndarray, plane: np.ndarray) -> np.ndarray:
    """Absolute distance of each point to plane ``(a, b, c, d)`` (normal need not be unit)."""
    plane = np.asarray(plane, dtype=np.float64)
    n = plane[:3]
    return np.abs(np.asarray(points, dtype=np.float64) @ n + plane[3]) / np.linalg.norm(n)


def _required_iterations(inlier_ratio: float, ransac_n: int, probability: float) -> float:
    """Iterations needed to draw one all-inlier sample with ``probability``."""
    if probability >= 1.0:
        return math.inf  # early stopping disabled
    w_n = inlier_ratio**ransac_n
    if w_n <= 0.0:
        return math.inf
    if w_n >= 1.0:
        return 0.0
    return math.log(1.0 - probability) / math.log(1.0 - w_n)


def segment_plane(
    points: np.ndarray,
    distance_threshold: float = 0.01,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    seed: int | None = 0,
    *,
    probability: float = 0.99999999,
    refine: bool = True,
) -> PlaneResult:
    """Fit the dominant plane with RANSAC. It is deterministic for a given ``seed``.

    Parameters follow Open3D's names: ``distance_threshold`` (max point-to-plane
    distance for an inlier), ``ransac_n`` (points per hypothesis, ≥ 3; >3 uses a
    least-squares fit of the sample), ``num_iterations`` (upper bound) and
    ``probability`` (adaptive early stop once an all-inlier sample has been drawn
    with this confidence; 1.0 disables it). ``seed=None`` uses fresh OS
    entropy (non-deterministic, opt-in).
    """
    if distance_threshold <= 0:
        raise ValueError("distance_threshold must be > 0")
    if ransac_n < 3:
        raise ValueError("ransac_n must be >= 3")
    if num_iterations < 1:
        raise ValueError("num_iterations must be >= 1")
    if not 0.0 < probability <= 1.0:
        raise ValueError("probability must be in (0, 1]")
    pts = _check_points(points, ransac_n)
    n = pts.shape[0]
    rng = np.random.default_rng(seed)

    best_count = -1
    best_rmse = math.inf
    best_plane: np.ndarray | None = None
    required = math.inf
    it = 0
    for it in range(1, num_iterations + 1):
        sample = pts[rng.choice(n, size=ransac_n, replace=False)]
        if ransac_n == 3:
            normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            norm = float(np.linalg.norm(normal))
            if norm < 1e-12:
                continue  # collinear sample: no plane
            normal = normal / norm
            plane = np.array([*normal, -float(normal @ sample[0])])
        else:
            c = sample.mean(axis=0)
            _, s, vt = np.linalg.svd(sample - c, full_matrices=False)
            if s[1] < 1e-12:
                continue
            plane = np.array([*vt[-1], -float(vt[-1] @ c)])
        dist = np.abs(pts @ plane[:3] + plane[3])
        mask = dist <= distance_threshold
        count = int(mask.sum())
        if count == 0:
            continue
        rmse = float(np.sqrt(np.mean(dist[mask] ** 2)))
        if count > best_count or (count == best_count and rmse < best_rmse):
            best_count, best_rmse, best_plane = count, rmse, plane
            required = _required_iterations(count / n, ransac_n, probability)
        if probability < 1.0 and it >= required:
            break

    if best_plane is None:
        return PlaneResult(
            plane_model=np.full(4, np.nan),
            inliers=np.zeros(0, dtype=np.int64),
            fitness=0.0,
            inlier_rmse=float("nan"),
            n_iter=it,
            seed=seed,
            refined=False,
        )
    plane = canonical_plane(best_plane[:3], best_plane[3])
    inliers = np.flatnonzero(point_plane_distance(pts, plane) <= distance_threshold)
    if refine and len(inliers) >= 3:
        refit = fit_plane_lstsq(pts[inliers])
        new_inliers = np.flatnonzero(point_plane_distance(pts, refit) <= distance_threshold)
        if len(new_inliers) >= len(inliers):  # never let refinement lose support
            plane, inliers = refit, new_inliers
    dist = point_plane_distance(pts[inliers], plane)
    return PlaneResult(
        plane_model=plane,
        inliers=inliers.astype(np.int64),
        fitness=len(inliers) / n,
        inlier_rmse=float(np.sqrt(np.mean(dist**2))) if len(inliers) else float("nan"),
        n_iter=it,
        seed=seed,
        refined=refine,
    )


def segment_planes(
    points: np.ndarray,
    max_planes: int = 3,
    min_inliers: int = 50,
    *,
    distance_threshold: float = 0.01,
    num_iterations: int = 1000,
    seed: int = 0,
    **kwargs: Any,
) -> list[PlaneResult]:
    """Sequential multi-plane extraction: fit, remove inliers, repeat (seed + i).

    Returned ``inliers`` index the **original** ``points``. Stops when a plane has
    fewer than ``min_inliers`` support or fewer than 3 points remain.
    """
    pts = _check_points(points, 3)
    remaining = np.arange(pts.shape[0])
    out: list[PlaneResult] = []
    for i in range(max_planes):
        if len(remaining) < 3:
            break
        res = segment_plane(
            pts[remaining], distance_threshold, num_iterations=num_iterations,
            seed=seed + i, **kwargs,
        )
        if len(res.inliers) < min_inliers:
            break
        res.inliers = remaining[res.inliers]
        res.fitness = len(res.inliers) / pts.shape[0]
        out.append(res)
        remaining = np.setdiff1d(remaining, res.inliers, assume_unique=True)
    return out


def plane_angle_deg(p: np.ndarray, q: np.ndarray) -> float:
    """Angle between two plane normals in degrees, ignoring the sign (0…90)."""
    a = np.asarray(p, dtype=np.float64)[:3]
    b = np.asarray(q, dtype=np.float64)[:3]
    cos = abs(float(a @ b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(min(1.0, cos))))


# ------------------------------------------------------------- synthetic scenes
def _plane_points(
    plane: np.ndarray, n: int, extent: float, noise: float, center: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    normal = np.asarray(plane[:3], dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    # Project the requested centre onto the plane, then scatter in-plane.
    c = np.asarray(center, dtype=np.float64)
    c = c - (c @ normal + plane[3] / np.linalg.norm(plane[:3])) * normal
    uv = rng.uniform(-extent, extent, size=(n, 2))
    return c + uv[:, :1] * u + uv[:, 1:] * v + rng.normal(0.0, noise, size=(n, 1)) * normal


def make_plane_scene(
    planes: Sequence[Sequence[float]] = ((0.0, 0.0, 1.0, 0.0),),
    n_per_plane: int | Sequence[int] = 400,
    n_outliers: int = 100,
    noise: float = 0.005,
    extent: float = 1.0,
    centers: Sequence[Sequence[float]] | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Points on given planes (+ Gaussian noise along the normal) + uniform outliers.

    Returns ``(points, labels, true_planes)``: labels are the plane index or ``-1``
    for outliers, and ``true_planes`` are canonical ``(a, b, c, d)``. Points are
    shuffled with the same seed.
    """
    rng = np.random.default_rng(seed)
    counts = [int(n_per_plane)] * len(planes) if np.isscalar(n_per_plane) else list(n_per_plane)
    if len(counts) != len(planes):
        raise ValueError("n_per_plane must be an int or one count per plane")
    centers = centers or [(0.0, 0.0, 0.0)] * len(planes)
    chunks, labels, truth = [], [], []
    for i, (pl, cnt, ctr) in enumerate(zip(planes, counts, centers)):
        pl = np.asarray(pl, dtype=np.float64)
        truth.append(canonical_plane(pl[:3], pl[3]))
        chunks.append(_plane_points(pl, cnt, extent, noise, np.asarray(ctr), rng))
        labels.append(np.full(cnt, i, dtype=np.int64))
    if n_outliers:
        lo = np.min(np.vstack(chunks), axis=0) - 0.2
        hi = np.max(np.vstack(chunks), axis=0) + 0.2
        chunks.append(rng.uniform(lo, hi, size=(n_outliers, 3)))
        labels.append(np.full(n_outliers, -1, dtype=np.int64))
    pts = np.vstack(chunks)
    lab = np.concatenate(labels)
    order = rng.permutation(len(pts))
    return pts[order], lab[order], truth


def inlier_precision_recall(inliers: np.ndarray, labels: np.ndarray, plane_id: int) -> dict:
    """Precision / recall of predicted inliers vs ground-truth plane membership."""
    pred = np.zeros(len(labels), dtype=bool)
    pred[np.asarray(inliers, dtype=np.int64)] = True
    true = labels == plane_id
    tp = int(np.sum(pred & true))
    return {
        "precision": tp / max(int(pred.sum()), 1),
        "recall": tp / max(int(true.sum()), 1),
    }


def plane_segmentation_eval(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Eval row: ground plane + tilted wall + outliers. Recover both planes; check determinism."""
    pc = dict((cfg or {}).get("plane_segmentation") or {})
    seed = int((cfg or {}).get("seed", 7))
    thresh = float(pc.get("distance_threshold", 0.02))
    n_iter = int(pc.get("num_iterations", 1000))
    planes = [(0.0, 0.0, 1.0, 0.0), (1.0, 0.0, 0.3, -0.8)]  # ground z=0, tilted wall
    pts, labels, truth = make_plane_scene(
        planes,
        n_per_plane=[int(pc.get("n_ground", 500)), int(pc.get("n_wall", 250))],
        n_outliers=int(pc.get("n_outliers", 150)),
        noise=float(pc.get("noise", 0.005)),
        centers=[(0.0, 0.0, 0.0), (0.8, 0.0, 0.5)],
        seed=seed,
    )
    found = segment_planes(
        pts, max_planes=2, min_inliers=50, distance_threshold=thresh,
        num_iterations=n_iter, seed=seed,
    )
    rows = []
    for res in found:
        # Match each found plane to the closest true plane by normal angle.
        angles = [plane_angle_deg(res.plane_model, t) for t in truth]
        pid = int(np.argmin(angles))
        t = truth[pid]
        d_err = abs(float(res.plane_model[3]) - float(t[3]))  # both canonical
        rows.append(
            {
                "plane_id": pid,
                **res.to_dict(),
                "normal_angle_err_deg": angles[pid],
                "offset_err": d_err,
                **inlier_precision_recall(res.inliers, labels, pid),
            }
        )
    first = segment_plane(pts, thresh, num_iterations=n_iter, seed=seed)
    again = segment_plane(pts, thresh, num_iterations=n_iter, seed=seed)
    by_seed = [
        segment_plane(pts, thresh, num_iterations=n_iter, seed=seed + i) for i in range(5)
    ]
    return {
        "plane_segmentation": {
            "n_points": int(len(pts)),
            "n_outliers": int(np.sum(labels == -1)),
            "distance_threshold": thresh,
            "num_iterations": n_iter,
            "seed": seed,
            "planes": rows,
            "deterministic_same_seed": bool(
                np.array_equal(again.plane_model, first.plane_model)
                and np.array_equal(again.inliers, first.inliers)
            ),
            # Different seeds may pick slightly different inlier sets; same seed never does.
            "inliers_by_seed": [int(len(r.inliers)) for r in by_seed],
            "angle_spread_by_seed_deg": float(
                max(plane_angle_deg(r.plane_model, first.plane_model) for r in by_seed)
            ),
        }
    }


__all__ = [
    "PlaneResult",
    "segment_plane",
    "segment_planes",
    "fit_plane_lstsq",
    "canonical_plane",
    "point_plane_distance",
    "plane_angle_deg",
    "make_plane_scene",
    "inlier_precision_recall",
    "plane_segmentation_eval",
]
