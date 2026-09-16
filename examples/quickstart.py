#!/usr/bin/env python3
"""Quickstart: sample clouds, featurize one, train a tiny classifier."""

from __future__ import annotations

import numpy as np

from point_cloud_geo.config import load_config
from point_cloud_geo.data import CLASS_NAMES, sample_cloud
from point_cloud_geo.features import cloud_feature_vector, estimate_normals_pca
from point_cloud_geo.sampling import fps_downsample
from point_cloud_geo.train import train_eval


def main() -> None:
    cfg = load_config()
    rng = np.random.default_rng(0)

    print("Synthetic clouds (one per class):")
    for name in CLASS_NAMES:
        cloud = sample_cloud(name, n_points=128, rng=rng)
        fps = fps_downsample(cloud, 64, seed=0)
        normals, eigs = estimate_normals_pca(fps, k=8)
        feat = cloud_feature_vector(cloud, fps_points=64, knn_k=8)
        print(
            f"  {name:7s}  shape={cloud.shape}  fps={fps.shape}  "
            f"normal_norm≈{np.linalg.norm(normals, axis=1).mean():.3f}  "
            f"feat_dim={feat.shape[0]}  λ0_mean={eigs[:, 0].mean():.4f}"
        )

    print("\nTrain / eval tiny classifier on geometric features…")
    result = train_eval(cfg)
    print(f"  train_acc={result['train_acc']:.3f}  test_acc={result['test_acc']:.3f}")
    print(result["report"])


if __name__ == "__main__":
    main()
