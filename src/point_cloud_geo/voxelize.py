"""Simple voxelization — occupancy / count / max-bin histograms for cloud features."""

from __future__ import annotations

from typing import Literal

import numpy as np

ReductionMode = Literal["occupancy", "count", "max_count_bin"]
VALID_REDUCTIONS: tuple[str, ...] = ("occupancy", "count", "max_count_bin")


def voxel_grid_indices(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Map (N, 3) points to integer voxel coordinates."""
    pts = np.asarray(points, dtype=np.float64)
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")
    # Shift so min corner is near origin for stable bins
    mins = pts.min(axis=0)
    return np.floor((pts - mins) / voxel_size).astype(np.int64)


def _binned_volume(
    points: np.ndarray,
    voxel_size: float,
    *,
    grid_extent: float = 2.5,
) -> tuple[np.ndarray, int]:
    """Center cloud and accumulate raw per-bin counts in a G×G×G volume."""
    pts = np.asarray(points, dtype=np.float64)
    center = pts.mean(axis=0, keepdims=True)
    centered = pts - center
    g = int(max(2, min(8, round(grid_extent / voxel_size))))
    half = g / 2.0
    coords = np.floor((centered / voxel_size) + half).astype(np.int64)
    coords = np.clip(coords, 0, g - 1)
    vol = np.zeros((g, g, g), dtype=np.float64)
    for i, j, k in coords:
        vol[i, j, k] += 1.0
    return vol, g


def voxel_occupancy_features(
    points: np.ndarray,
    voxel_size: float = 0.25,
    *,
    grid_extent: float = 2.5,
    reduction: ReductionMode | str = "occupancy",
) -> np.ndarray:
    """
    Fixed-size voxel histogram over a coarse 3D grid.

    reduction
    ---------
    occupancy :
        Binary occupied bins, density-normalized (legacy default).
    count :
        Raw point counts per bin, L1-normalized.
    max_count_bin :
        One-hot (soft) peak: only the densest bin keeps mass (then L1-normalized).
        Useful teaching contrast vs uniform occupancy on dense vs sparse clouds.
    """
    mode = str(reduction).lower().strip()
    if mode not in VALID_REDUCTIONS:
        raise ValueError(
            f"unknown voxel reduction {reduction!r}; expected one of {VALID_REDUCTIONS}"
        )

    vol, _g = _binned_volume(points, voxel_size, grid_extent=grid_extent)

    if mode == "occupancy":
        out = (vol > 0).astype(np.float64)
        dens = out.sum()
        if dens > 0:
            out = out / dens
        return out.ravel()

    if mode == "count":
        out = vol.copy()
        dens = out.sum()
        if dens > 0:
            out = out / dens
        return out.ravel()

    # max_count_bin: keep only the argmax bin's count
    out = np.zeros_like(vol)
    if vol.max() > 0:
        flat_idx = int(np.argmax(vol))
        out.ravel()[flat_idx] = vol.ravel()[flat_idx]
        dens = out.sum()
        if dens > 0:
            out = out / dens
    return out.ravel()


def voxel_centroids(points: np.ndarray, voxel_size: float = 0.25) -> np.ndarray:
    """Return one centroid per occupied voxel (variable length)."""
    pts = np.asarray(points, dtype=np.float64)
    idx = voxel_grid_indices(pts, voxel_size)
    # Hash voxel keys
    keys = idx[:, 0] * 1_000_003 + idx[:, 1] * 1_009 + idx[:, 2]
    uniq, inv = np.unique(keys, return_inverse=True)
    cents = np.zeros((len(uniq), 3), dtype=np.float64)
    counts = np.zeros(len(uniq), dtype=np.float64)
    for i in range(pts.shape[0]):
        cents[inv[i]] += pts[i]
        counts[inv[i]] += 1.0
    cents /= counts[:, None]
    return cents
