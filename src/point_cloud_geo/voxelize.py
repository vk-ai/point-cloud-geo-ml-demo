"""Simple voxelization — occupancy / centroid histograms for cloud features."""

from __future__ import annotations

import numpy as np


def voxel_grid_indices(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Map (N, 3) points to integer voxel coordinates."""
    pts = np.asarray(points, dtype=np.float64)
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")
    # Shift so min corner is near origin for stable bins
    mins = pts.min(axis=0)
    return np.floor((pts - mins) / voxel_size).astype(np.int64)


def voxel_occupancy_features(
    points: np.ndarray,
    voxel_size: float = 0.25,
    *,
    grid_extent: float = 2.5,
) -> np.ndarray:
    """
    Fixed-size occupancy histogram over a coarse 3D grid.

    Centers the cloud, bins into a G×G×G occupancy volume flattened to a vector.
    G is chosen from grid_extent / voxel_size (clamped).
    """
    pts = np.asarray(points, dtype=np.float64)
    center = pts.mean(axis=0, keepdims=True)
    centered = pts - center
    g = int(max(2, min(8, round(grid_extent / voxel_size))))
    half = g / 2.0
    # Map [-half*vs, half*vs] roughly into [0, g)
    coords = np.floor((centered / voxel_size) + half).astype(np.int64)
    coords = np.clip(coords, 0, g - 1)
    vol = np.zeros((g, g, g), dtype=np.float64)
    for i, j, k in coords:
        vol[i, j, k] = 1.0
    # Density-normalized occupancy
    dens = vol.sum()
    if dens > 0:
        vol = vol / dens
    return vol.ravel()


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
