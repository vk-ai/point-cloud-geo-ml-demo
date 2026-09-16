"""Farthest Point Sampling (FPS) — classic geometric ML downsampling."""

from __future__ import annotations

import numpy as np


def farthest_point_sampling(points: np.ndarray, n_samples: int, *, seed: int = 0) -> np.ndarray:
    """
    Greedy FPS: iteratively pick the point farthest from the current set.

    Parameters
    ----------
    points : (N, 3)
    n_samples : number of indices to return (clamped to N)

    Returns
    -------
    indices : (M,) int64 into ``points``
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N, 3); got {pts.shape}")
    n = pts.shape[0]
    m = int(min(max(n_samples, 1), n))
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, n))
    selected = np.empty(m, dtype=np.int64)
    selected[0] = start
    # min squared distance to the selected set
    dist2 = np.full(n, np.inf, dtype=np.float64)
    for i in range(1, m):
        last = pts[selected[i - 1]]
        d2 = np.sum((pts - last) ** 2, axis=1)
        dist2 = np.minimum(dist2, d2)
        selected[i] = int(np.argmax(dist2))
    return selected


def fps_downsample(points: np.ndarray, n_samples: int, *, seed: int = 0) -> np.ndarray:
    """Return the FPS-selected subset of points as (M, 3)."""
    idx = farthest_point_sampling(points, n_samples, seed=seed)
    return np.asarray(points, dtype=np.float64)[idx]
