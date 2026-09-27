"""PointNet-lite teaching stub: shared MLP + symmetric max-pool (+ optional normals).

Classic PointNet idea (Qi et al.): per-point MLP → max-pool → class MLP.
Here: tiny numpy MLP on XYZ or XYZ+normals — **not** PointNet++/LitePT/Open3D.

Motivation cite (efficiency claims, not a dependency): PointNetLite SPIE 2025
https://doi.org/10.1117/12.3063179
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from point_cloud_geo.normals_radius import estimate_normals_radius
from point_cloud_geo.sampling import downsample

FeatMode = Literal["xyz", "xyz_normals"]


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-12)


@dataclass
class PointNetLite:
    """Shared per-point linear → ReLU → max-pool → linear classifier (numpy)."""

    in_dim: int
    hidden: int
    n_classes: int
    W1: np.ndarray  # (hidden, in_dim)
    b1: np.ndarray
    W2: np.ndarray  # (n_classes, hidden)
    b2: np.ndarray

    @classmethod
    def create(
        cls,
        in_dim: int,
        n_classes: int,
        rng: np.random.Generator,
        *,
        hidden: int = 32,
    ) -> "PointNetLite":
        W1 = rng.normal(0.0, 0.5 / np.sqrt(in_dim), size=(hidden, in_dim))
        b1 = np.zeros(hidden, dtype=np.float64)
        W2 = rng.normal(0.0, 0.5 / np.sqrt(hidden), size=(n_classes, hidden))
        b2 = np.zeros(n_classes, dtype=np.float64)
        return cls(
            in_dim=in_dim,
            hidden=hidden,
            n_classes=n_classes,
            W1=W1.astype(np.float64),
            b1=b1,
            W2=W2.astype(np.float64),
            b2=b2,
        )

    def encode(self, pts_feat: np.ndarray) -> np.ndarray:
        """pts_feat: (N, C) → global (hidden,) via shared MLP + max-pool."""
        h = _relu(pts_feat @ self.W1.T + self.b1)  # (N, H)
        return h.max(axis=0)

    def logits_cloud(self, pts_feat: np.ndarray) -> np.ndarray:
        g = self.encode(pts_feat)
        return g @ self.W2.T + self.b2

    def n_params(self) -> int:
        return int(self.W1.size + self.b1.size + self.W2.size + self.b2.size)


def cloud_to_point_features(
    points: np.ndarray,
    *,
    mode: FeatMode = "xyz",
    n_points: int = 64,
    radius: float = 0.35,
    min_nn: int = 4,
    seed: int = 0,
) -> np.ndarray:
    """Downsample + optional radius normals → (M, 3) or (M, 6)."""
    pts = downsample(points, n_points, method="fps", seed=seed, start_index=0)
    # Center for translation robustness
    pts = pts - pts.mean(axis=0, keepdims=True)
    if mode == "xyz":
        return pts.astype(np.float64)
    normals, _, _ = estimate_normals_radius(pts, radius, min_nn=min_nn)
    return np.concatenate([pts, normals], axis=1).astype(np.float64)


def _cross_entropy(logits: np.ndarray, y: int) -> float:
    p = _softmax(logits[None, :])[0]
    return float(-np.log(p[y] + 1e-12))


def train_pointnet_lite(
    model: PointNetLite,
    clouds_feat: list[np.ndarray],
    labels: np.ndarray,
    *,
    epochs: int = 40,
    lr: float = 0.05,
    seed: int = 0,
) -> list[float]:
    """SGD over clouds (batch size 1) — keep tiny/fast."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=np.int64)
    losses: list[float] = []
    n = len(clouds_feat)
    for _ in range(epochs):
        order = rng.permutation(n)
        epoch_loss = 0.0
        for i in order:
            x = clouds_feat[i]  # (N, C)
            y = int(labels[i])
            # Forward with cache
            pre = x @ model.W1.T + model.b1  # (N, H)
            h = _relu(pre)
            # max-pool + argmax indices for backprop
            g = h.max(axis=0)
            idx = h.argmax(axis=0)
            logits = g @ model.W2.T + model.b2
            loss = _cross_entropy(logits, y)
            epoch_loss += loss

            probs = _softmax(logits[None, :])[0]
            dlogits = probs.copy()
            dlogits[y] -= 1.0

            dW2 = np.outer(dlogits, g)
            db2 = dlogits
            dg = model.W2.T @ dlogits  # (H,)

            # Route dg only to argmax rows (max-pool STE)
            dh = np.zeros_like(h)
            dh[idx, np.arange(model.hidden)] = dg
            dpre = dh * (pre > 0)

            dW1 = dpre.T @ x
            db1 = dpre.sum(axis=0)

            model.W2 -= lr * dW2
            model.b2 -= lr * db2
            model.W1 -= lr * dW1
            model.b1 -= lr * db1
        losses.append(epoch_loss / max(n, 1))
    return losses


def predict_clouds(model: PointNetLite, clouds_feat: list[np.ndarray]) -> np.ndarray:
    logits = np.stack([model.logits_cloud(c) for c in clouds_feat], axis=0)
    return logits.argmax(axis=1)


def run_pointnet_lite_demo(
    clouds: list[np.ndarray],
    labels: np.ndarray,
    *,
    mode: FeatMode = "xyz",
    n_points: int = 48,
    radius: float = 0.35,
    hidden: int = 24,
    epochs: int = 35,
    lr: float = 0.08,
    test_size: float = 0.3,
    seed: int = 7,
    n_classes: int = 3,
) -> dict[str, Any]:
    """Train/eval PointNet-lite; optional normals-on vs will be compared by caller."""
    from sklearn.model_selection import train_test_split

    labels = np.asarray(labels, dtype=np.int64)
    feats = [
        cloud_to_point_features(
            c, mode=mode, n_points=n_points, radius=radius, seed=seed + i
        )
        for i, c in enumerate(clouds)
    ]
    idx = np.arange(len(feats))
    tr, te = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=labels
    )
    in_dim = 3 if mode == "xyz" else 6
    rng = np.random.default_rng(seed)
    model = PointNetLite.create(in_dim, n_classes, rng, hidden=hidden)
    train_feats = [feats[i] for i in tr]
    train_y = labels[tr]
    losses = train_pointnet_lite(
        model, train_feats, train_y, epochs=epochs, lr=lr, seed=seed
    )
    pred_tr = predict_clouds(model, train_feats)
    pred_te = predict_clouds(model, [feats[i] for i in te])
    train_acc = float((pred_tr == train_y).mean())
    test_acc = float((pred_te == labels[te]).mean())
    return {
        "mode": mode,
        "n_points": n_points,
        "radius": radius if mode == "xyz_normals" else None,
        "hidden": hidden,
        "n_params": model.n_params(),
        "train_acc": train_acc,
        "test_acc": test_acc,
        "final_loss": losses[-1] if losses else None,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "model": model,
    }


def compare_normals_ablation(
    clouds: list[np.ndarray],
    labels: np.ndarray,
    **kwargs: Any,
) -> dict[str, Any]:
    """XYZ-only vs XYZ+radius-normals PointNet-lite table."""
    xyz = run_pointnet_lite_demo(clouds, labels, mode="xyz", **kwargs)
    xyz_n = run_pointnet_lite_demo(clouds, labels, mode="xyz_normals", **kwargs)
    return {
        "xyz": {k: v for k, v in xyz.items() if k != "model"},
        "xyz_normals": {k: v for k, v in xyz_n.items() if k != "model"},
        "delta_test_acc": xyz_n["test_acc"] - xyz["test_acc"],
        "notes": (
            "OSS learning stub — radius must match cloud spacing; "
            "too-small r → noisy normals; too-large → oversmoothed."
        ),
    }
