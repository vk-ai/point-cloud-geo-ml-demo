"""Farthest Point Sampling (FPS) + random baseline — geometric ML downsampling."""

from __future__ import annotations

import numpy as np

VALID_SAMPLING = ("fps", "random")


def farthest_point_sampling(
    points: np.ndarray,
    n_samples: int,
    *,
    seed: int = 0,
    start_index: int | None = 0,
) -> np.ndarray:
    """
    Greedy FPS: iteratively pick the point farthest from the current set.

    Parameters
    ----------
    points : (N, 3)
    n_samples : number of indices to return (clamped to N)
    start_index : first point index (default 0). ``None`` → seeded random start
      (Open3D-inspired teaching knob — numpy only, no Open3D).

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
    if start_index is None:
        start = int(rng.integers(0, n))
    else:
        start = int(start_index) % n
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


def random_point_sampling(
    points: np.ndarray,
    n_samples: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Uniform random subset of indices (seeded). Teaching baseline vs FPS."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N, 3); got {pts.shape}")
    n = pts.shape[0]
    m = int(min(max(n_samples, 1), n))
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=m, replace=False).astype(np.int64)


def fps_downsample(
    points: np.ndarray,
    n_samples: int,
    *,
    seed: int = 0,
    start_index: int | None = 0,
) -> np.ndarray:
    """Return the FPS-selected subset of points as (M, 3)."""
    idx = farthest_point_sampling(points, n_samples, seed=seed, start_index=start_index)
    return np.asarray(points, dtype=np.float64)[idx]


def random_downsample(points: np.ndarray, n_samples: int, *, seed: int = 0) -> np.ndarray:
    """Return a seeded random subset of points as (M, 3)."""
    idx = random_point_sampling(points, n_samples, seed=seed)
    return np.asarray(points, dtype=np.float64)[idx]


def downsample(
    points: np.ndarray,
    n_samples: int,
    *,
    method: str = "fps",
    seed: int = 0,
    start_index: int | None = 0,
) -> np.ndarray:
    """Dispatch FPS vs random downsampling (same N for fair compare)."""
    method = str(method).lower()
    if method not in VALID_SAMPLING:
        raise ValueError(f"sampling must be one of {VALID_SAMPLING}; got {method!r}")
    if method == "fps":
        return fps_downsample(points, n_samples, seed=seed, start_index=start_index)
    return random_downsample(points, n_samples, seed=seed)


def coverage_mean_nn_spacing(points: np.ndarray) -> float:
    """
    Cheap coverage proxy: mean nearest-neighbor Euclidean distance among points.

    For a fixed sample count, higher spacing ≈ better spatial spread (FPS tends to win
    on spread-out surfaces; random can clump). Not a ScanNet/KITTI metric.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N, 3); got {pts.shape}")
    n = pts.shape[0]
    if n < 2:
        return 0.0
    d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    nn = np.sqrt(np.min(d2, axis=1))
    return float(nn.mean())


def coverage_bbox_fill_ratio(points: np.ndarray, grid: int = 4) -> float:
    """Fraction of coarse bbox cells that contain ≥1 sample (0..1)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] == 0:
        return 0.0
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-12)
    norm = (pts - lo) / span
    bins = np.clip((norm * grid).astype(np.int64), 0, grid - 1)
    keys = bins[:, 0] * grid * grid + bins[:, 1] * grid + bins[:, 2]
    return float(len(np.unique(keys)) / float(grid**3))
