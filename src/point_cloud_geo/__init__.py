"""Tiny point-cloud / geometric ML fundamentals demo (CPU, OSS learning)."""

__version__ = "0.1.0"

from point_cloud_geo.config import load_config
from point_cloud_geo.data import CLASS_NAMES, make_dataset
from point_cloud_geo.eval import run_pipeline
from point_cloud_geo.normals_radius import estimate_normals_radius
from point_cloud_geo.pointnet_lite import PointNetLite, compare_normals_ablation
from point_cloud_geo.train import train_eval

__all__ = [
    "__version__",
    "CLASS_NAMES",
    "load_config",
    "make_dataset",
    "run_pipeline",
    "train_eval",
    "estimate_normals_radius",
    "PointNetLite",
    "compare_normals_ablation",
]
