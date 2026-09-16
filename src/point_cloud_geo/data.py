"""Synthetic labeled point clouds: plane, sphere, cube (no large datasets)."""

from __future__ import annotations

from typing import Literal

import numpy as np

CLASS_NAMES: tuple[str, ...] = ("plane", "sphere", "cube")
ClassName = Literal["plane", "sphere", "cube"]


def _unit_sphere_points(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform-ish samples on the unit sphere via Gaussian normalization."""
    xyz = rng.normal(size=(n, 3))
    norms = np.linalg.norm(xyz, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return xyz / norms


def sample_plane(n: int, rng: np.random.Generator, noise: float = 0.02) -> np.ndarray:
    """Points on z≈0 plane in [-1, 1]^2 with small Gaussian noise."""
    xy = rng.uniform(-1.0, 1.0, size=(n, 2))
    z = rng.normal(0.0, noise, size=(n, 1))
    pts = np.concatenate([xy, z], axis=1)
    # Random rigid transform so orientation varies
    return _random_rigid(pts, rng)


def sample_sphere(n: int, rng: np.random.Generator, noise: float = 0.02) -> np.ndarray:
    """Points near the unit sphere surface."""
    pts = _unit_sphere_points(n, rng)
    pts = pts + rng.normal(0.0, noise, size=pts.shape)
    return _random_rigid(pts, rng, scale_jitter=True)


def sample_cube(n: int, rng: np.random.Generator, noise: float = 0.02) -> np.ndarray:
    """Points on faces of a cube in [-1, 1]^3 with noise."""
    # Pick faces uniformly, then UV on that face
    face = rng.integers(0, 6, size=n)
    u = rng.uniform(-1.0, 1.0, size=n)
    v = rng.uniform(-1.0, 1.0, size=n)
    pts = np.zeros((n, 3), dtype=np.float64)
    # ±x, ±y, ±z faces
    for i in range(n):
        f = int(face[i])
        if f == 0:
            pts[i] = [1.0, u[i], v[i]]
        elif f == 1:
            pts[i] = [-1.0, u[i], v[i]]
        elif f == 2:
            pts[i] = [u[i], 1.0, v[i]]
        elif f == 3:
            pts[i] = [u[i], -1.0, v[i]]
        elif f == 4:
            pts[i] = [u[i], v[i], 1.0]
        else:
            pts[i] = [u[i], v[i], -1.0]
    pts = pts + rng.normal(0.0, noise, size=pts.shape)
    return _random_rigid(pts, rng, scale_jitter=True)


def _random_rigid(
    pts: np.ndarray,
    rng: np.random.Generator,
    *,
    scale_jitter: bool = False,
) -> np.ndarray:
    """Apply random rotation + small translation (+ optional scale)."""
    # Random rotation via QR of Gaussian matrix
    a = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(a)
    # Fix reflections so det(Q)=+1
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    scale = float(rng.uniform(0.8, 1.2)) if scale_jitter else 1.0
    t = rng.uniform(-0.3, 0.3, size=(1, 3))
    return (pts * scale) @ q.T + t


_SAMPLERS = {
    "plane": sample_plane,
    "sphere": sample_sphere,
    "cube": sample_cube,
}


def sample_cloud(
    class_name: ClassName | int,
    n_points: int = 128,
    *,
    noise: float = 0.02,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample one labeled synthetic point cloud as (N, 3) float64."""
    if rng is None:
        rng = np.random.default_rng()
    if isinstance(class_name, int):
        class_name = CLASS_NAMES[class_name]  # type: ignore[assignment]
    if class_name not in _SAMPLERS:
        raise ValueError(f"Unknown class {class_name!r}; expected one of {CLASS_NAMES}")
    return _SAMPLERS[class_name](n_points, rng, noise=noise)


def make_dataset(
    n_per_class: int = 40,
    n_points: int = 128,
    *,
    noise: float = 0.02,
    seed: int = 7,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Build a tiny multi-class dataset.

    Returns
    -------
    clouds : list of (N, 3) arrays
    labels : (n_per_class * 3,) int64 in {0,1,2}
    """
    rng = np.random.default_rng(seed)
    clouds: list[np.ndarray] = []
    labels: list[int] = []
    for label, name in enumerate(CLASS_NAMES):
        for _ in range(n_per_class):
            clouds.append(sample_cloud(name, n_points, noise=noise, rng=rng))
            labels.append(label)
    return clouds, np.asarray(labels, dtype=np.int64)
