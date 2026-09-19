"""Geometric features: PCA normals on neighborhoods + shape descriptors."""

from __future__ import annotations

import numpy as np

from point_cloud_geo.sampling import fps_downsample
from point_cloud_geo.voxelize import voxel_occupancy_features


def knn_indices(points: np.ndarray, k: int) -> np.ndarray:
    """Brute-force kNN indices including self; returns (N, k)."""
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    kk = int(min(max(k, 2), n))
    # (N, N) squared distances
    d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2)
    return np.argpartition(d2, kk - 1, axis=1)[:, :kk]


def estimate_normals_pca(points: np.ndarray, k: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate per-point normals via PCA on k-neighborhoods.

    Returns
    -------
    normals : (N, 3) unit normals (smallest eigenvector)
    eigenvalues : (N, 3) sorted ascending λ0 ≤ λ1 ≤ λ2
    """
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    idx = knn_indices(pts, k)
    normals = np.zeros((n, 3), dtype=np.float64)
    eigs = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        nb = pts[idx[i]]
        centered = nb - nb.mean(axis=0, keepdims=True)
        cov = (centered.T @ centered) / max(nb.shape[0] - 1, 1)
        w, v = np.linalg.eigh(cov)
        # eigh returns ascending eigenvalues
        eigs[i] = w
        normals[i] = v[:, 0]
    # Flip normals toward cloud centroid for sign consistency
    toward = pts.mean(axis=0) - pts
    flip = np.sum(normals * toward, axis=1) < 0
    normals[flip] *= -1.0
    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-12)
    return normals, eigs


def local_shape_ratios(eigenvalues: np.ndarray) -> np.ndarray:
    """
    Per-point linearity / planarity / sphericity from eigenvalues.

    Returns (N, 3) = [linearity, planarity, sphericity].
    """
    lam = np.maximum(np.asarray(eigenvalues, dtype=np.float64), 1e-12)
    l0, l1, l2 = lam[:, 0], lam[:, 1], lam[:, 2]
    linearity = (l2 - l1) / l2
    planarity = (l1 - l0) / l2
    sphericity = l0 / l2
    return np.stack([linearity, planarity, sphericity], axis=1)


def cloud_feature_vector(
    points: np.ndarray,
    *,
    fps_points: int = 64,
    knn_k: int = 8,
    voxel_size: float = 0.25,
    voxel_reduction: str = "occupancy",
    seed: int = 0,
) -> np.ndarray:
    """
    Aggregate a single cloud into a fixed feature vector for classification.

    Pipeline (geometric ML fundamentals):
      1. FPS downsample to fixed size
      2. PCA normals + eigenvalue shape ratios on neighborhoods
      3. Global stats of coords / normals / ratios
      4. Coarse voxel occupancy histogram
    """
    pts = fps_downsample(points, fps_points, seed=seed)
    normals, eigs = estimate_normals_pca(pts, k=knn_k)
    ratios = local_shape_ratios(eigs)

    # Center coords for translation robustness of stats
    centered = pts - pts.mean(axis=0, keepdims=True)
    coord_mean = centered.mean(axis=0)
    coord_std = centered.std(axis=0)
    coord_abs_mean = np.abs(centered).mean(axis=0)

    n_mean = normals.mean(axis=0)
    n_std = normals.std(axis=0)
    # Mean absolute normal components (orientation-ish)
    n_abs = np.abs(normals).mean(axis=0)

    ratio_mean = ratios.mean(axis=0)
    ratio_std = ratios.std(axis=0)

    # Roughness proxy: mean smallest eigenvalue
    roughness = float(eigs[:, 0].mean())

    voxel = voxel_occupancy_features(
        pts, voxel_size=voxel_size, reduction=voxel_reduction
    )

    parts = [
        coord_mean,
        coord_std,
        coord_abs_mean,
        n_mean,
        n_std,
        n_abs,
        ratio_mean,
        ratio_std,
        np.array([roughness, float(pts.shape[0])], dtype=np.float64),
        voxel,
    ]
    return np.concatenate(parts).astype(np.float64)


def featurize_dataset(
    clouds: list[np.ndarray],
    *,
    fps_points: int = 64,
    knn_k: int = 8,
    voxel_size: float = 0.25,
    voxel_reduction: str = "occupancy",
    seed: int = 0,
) -> np.ndarray:
    """Stack cloud_feature_vector for each cloud → (B, D)."""
    feats = [
        cloud_feature_vector(
            c,
            fps_points=fps_points,
            knn_k=knn_k,
            voxel_size=voxel_size,
            voxel_reduction=voxel_reduction,
            seed=seed + i,
        )
        for i, c in enumerate(clouds)
    ]
    return np.stack(feats, axis=0)
