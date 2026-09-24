"""Eval harness: run pipeline and emit a small JSON-friendly report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from point_cloud_geo.config import load_config
from point_cloud_geo.data import sample_cloud
from point_cloud_geo.sampling import (
    VALID_SAMPLING,
    coverage_bbox_fill_ratio,
    coverage_mean_nn_spacing,
    downsample,
)
from point_cloud_geo.train import train_eval
from point_cloud_geo.data import make_dataset
from point_cloud_geo.pointnet_lite import compare_normals_ablation


def _fps_start(cfg: dict[str, Any]) -> int | None:
    fps_cfg = cfg.get("fps") or {}
    start = fps_cfg.get("start_index", 0) if isinstance(fps_cfg, dict) else 0
    return None if start is None else int(start)


def _coverage_on_fixed_cloud(cfg: dict[str, Any], method: str) -> dict[str, float]:
    """Coverage proxies on one seeded synthetic cloud (same N for fps vs random)."""
    seed = int(cfg.get("seed", 7))
    n_points = int(cfg.get("n_points", 128))
    n_keep = int(cfg.get("fps_points", 64))
    rng = np.random.default_rng(seed)
    cloud = sample_cloud("sphere", n_points=n_points, rng=rng)
    # Stretch so coverage differences are visible
    cloud = cloud * np.array([1.5, 1.0, 0.7], dtype=np.float64)
    pts = downsample(
        cloud,
        n_keep,
        method=method,
        seed=seed,
        start_index=_fps_start(cfg),
    )
    return {
        "mean_nn_spacing": coverage_mean_nn_spacing(pts),
        "bbox_fill_ratio": coverage_bbox_fill_ratio(pts, grid=4),
        "n_samples": float(pts.shape[0]),
    }


def run_pipeline(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Train/eval and return a serializable summary."""
    cfg = load_config() if cfg is None else dict(cfg)
    result = train_eval(cfg)
    voxel_cfg = cfg.get("voxel") or {}
    voxel_reduction = (
        str(voxel_cfg.get("reduction", "occupancy"))
        if isinstance(voxel_cfg, dict)
        else "occupancy"
    )
    sampling = str(cfg.get("sampling", "fps"))
    cov = _coverage_on_fixed_cloud(cfg, sampling)
    summary = {
        "train_acc": result["train_acc"],
        "test_acc": result["test_acc"],
        "n_train": result["n_train"],
        "n_test": result["n_test"],
        "n_features": result["n_features"],
        "classes": result["classes"],
        "classifier": cfg.get("classifier", "mlp"),
        "sampling": sampling,
        "fps_points": cfg.get("fps_points", 64),
        "fps_start_index": _fps_start(cfg),
        "knn_k": cfg.get("knn_k", 8),
        "voxel_size": cfg.get("voxel_size", 0.25),
        "voxel_reduction": voxel_reduction,
        "coverage_mean_nn_spacing": cov["mean_nn_spacing"],
        "coverage_bbox_fill_ratio": cov["bbox_fill_ratio"],
        "disclaimer": (
            "OSS/learning demo only — not employer production software."
        ),
    }
    pn_cfg = cfg.get("pointnet_lite") or {}
    if pn_cfg.get("enabled", False):
        clouds, labels = make_dataset(
            n_per_class=int(cfg.get("n_per_class", 40)),
            n_points=int(cfg.get("n_points", 128)),
            noise=float(cfg.get("noise", 0.02)),
            seed=int(cfg.get("seed", 7)),
        )
        normals_cfg = cfg.get("normals") or {}
        radius = float(normals_cfg.get("radius", 0.35))
        kwargs = dict(
            n_points=int(pn_cfg.get("n_points", 48)),
            radius=radius,
            hidden=int(pn_cfg.get("hidden", 24)),
            epochs=int(pn_cfg.get("epochs", 30)),
            lr=float(pn_cfg.get("lr", 0.08)),
            seed=int(cfg.get("seed", 7)),
            n_classes=len(summary["classes"]),
        )
        if pn_cfg.get("compare_normals", True):
            summary["pointnet_lite"] = compare_normals_ablation(clouds, labels, **kwargs)
        else:
            from point_cloud_geo.pointnet_lite import run_pointnet_lite_demo
            out = run_pointnet_lite_demo(clouds, labels, mode="xyz", **kwargs)
            summary["pointnet_lite"] = {k: v for k, v in out.items() if k != "model"}
    return summary


def compare_sampling(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    FPS vs random coverage + accuracy comparison table (same n_points).

    Teaching ablation — numpy + sklearn only; not Open3D/PointNet benchmarks.
    """
    base = load_config() if cfg is None else dict(cfg)
    rows: list[dict[str, Any]] = []
    for method in VALID_SAMPLING:
        c = dict(base)
        c["sampling"] = method
        if isinstance(c.get("voxel"), dict):
            c["voxel"] = dict(c["voxel"])
        if isinstance(c.get("fps"), dict):
            c["fps"] = dict(c["fps"])
        summary = run_pipeline(c)
        rows.append(
            {
                "sampling": method,
                "test_acc": summary["test_acc"],
                "train_acc": summary["train_acc"],
                "coverage_mean_nn_spacing": summary["coverage_mean_nn_spacing"],
                "coverage_bbox_fill_ratio": summary["coverage_bbox_fill_ratio"],
                "fps_points": summary["fps_points"],
                "fps_start_index": summary["fps_start_index"],
            }
        )
    return {
        "sampling_compare": rows,
        "disclaimer": (
            "OSS/learning demo only — FPS vs random teaching ablation; "
            "not Open3D/PointNet/ScanNet numbers."
        ),
    }


def write_report(summary: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
