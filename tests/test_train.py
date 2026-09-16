"""Train/eval smoke tests — must finish in seconds on CPU."""

from __future__ import annotations

from point_cloud_geo.config import load_config
from point_cloud_geo.eval import run_pipeline
from point_cloud_geo.train import train_eval


def test_train_eval_accuracy_floor():
    cfg = load_config()
    cfg.update(
        {
            "n_per_class": 24,
            "n_points": 64,
            "fps_points": 32,
            "knn_k": 6,
            "max_iter": 300,
            "seed": 11,
        }
    )
    result = train_eval(cfg)
    assert result["train_acc"] >= 0.7
    assert result["test_acc"] >= 0.55
    assert result["n_features"] > 10


def test_logistic_also_runs():
    cfg = load_config()
    cfg.update(
        {
            "classifier": "logistic",
            "n_per_class": 18,
            "n_points": 48,
            "fps_points": 24,
            "knn_k": 5,
            "seed": 12,
        }
    )
    result = train_eval(cfg)
    assert 0.0 <= result["test_acc"] <= 1.0


def test_run_pipeline_serializable():
    cfg = load_config()
    cfg.update({"n_per_class": 15, "n_points": 40, "fps_points": 20, "seed": 13})
    summary = run_pipeline(cfg)
    assert "test_acc" in summary
    assert "disclaimer" in summary
    assert summary["classes"] == ["plane", "sphere", "cube"]
