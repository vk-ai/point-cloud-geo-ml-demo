"""Radius-neighborhood PCA normals (Open3D-free; sklearn BallTree / numpy fallback).

For each point, gather neighbors within radius ``r`` (at least ``min_nn``);
PCA; smallest eigenvector = normal. Optional orient toward viewpoint/centroid.

PCL / Open3D analogue: radius vs k search + PCA.
https://pcl.readthedocs.io/projects/tutorials/en/master/normal_estimation.html
"""

from __future__ import annotations

import numpy as np

try:
    from sklearn.neighbors import BallTree

    _HAS_BALLTREE = True
except ImportError:  # pragma: no cover
    BallTree = None  # type: ignore
    _HAS_BALLTREE = False


def _radius_indices_balltree(points: np.ndarray, radius: float) -> list[np.ndarray]:
    tree = BallTree(points, leaf_size=16)
    return tree.query_radius(points, r=float(radius))


def _radius_indices_bruteforce(points: np.ndarray, radius: float) -> list[np.ndarray]:
    d2 = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2)
    r2 = float(radius) ** 2
    return [np.where(d2[i] <= r2)[0] for i in range(points.shape[0])]


def radius_neighbor_indices(
    points: np.ndarray,
    radius: float,
) -> list[np.ndarray]:
    """Return per-point index arrays of neighbors within ``radius`` (includes self)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N, 3); got {pts.shape}")
    if radius <= 0:
        raise ValueError("radius must be > 0")
    if _HAS_BALLTREE:
        return _radius_indices_balltree(pts, radius)
    return _radius_indices_bruteforce(pts, radius)  # pragma: no cover


def estimate_normals_radius(
    points: np.ndarray,
    radius: float,
    *,
    min_nn: int = 3,
    orient_toward: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate per-point normals via PCA over a radius neighborhood.

    Parameters
    ----------
    points : (N, 3)
    radius : neighborhood radius (must match cloud metric spacing)
    min_nn : if fewer than this many neighbors (including self), fall back to
             the ``min_nn`` nearest points (kNN fallback)
    orient_toward : optional (3,) viewpoint; default = cloud centroid

    Returns
    -------
    normals : (N, 3) unit normals
    eigenvalues : (N, 3) ascending
    nn_counts : (N,) neighbor counts used
    """
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    min_nn = int(max(min_nn, 3))
    idx_lists = radius_neighbor_indices(pts, radius)

    # Precompute full distance matrix only if we need kNN fallback often — use
    # argpartition per point when needed.
    normals = np.zeros((n, 3), dtype=np.float64)
    eigs = np.zeros((n, 3), dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)

    for i in range(n):
        nb_idx = np.asarray(idx_lists[i], dtype=np.int64)
        if nb_idx.size < min_nn:
            # kNN fallback
            d2 = np.sum((pts - pts[i]) ** 2, axis=1)
            nb_idx = np.argpartition(d2, min_nn - 1)[:min_nn]
        counts[i] = int(nb_idx.size)
        nb = pts[nb_idx]
        centered = nb - nb.mean(axis=0, keepdims=True)
        cov = (centered.T @ centered) / max(nb.shape[0] - 1, 1)
        w, v = np.linalg.eigh(cov)
        eigs[i] = w
        normals[i] = v[:, 0]

    viewpoint = (
        np.asarray(orient_toward, dtype=np.float64)
        if orient_toward is not None
        else pts.mean(axis=0)
    )
    toward = viewpoint - pts
    flip = np.sum(normals * toward, axis=1) < 0
    normals[flip] *= -1.0
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-12)
    return normals, eigs, counts
