"""CLI: python -m point_cloud_geo"""

from __future__ import annotations

from point_cloud_geo.config import load_config
from point_cloud_geo.eval import run_pipeline


def main() -> None:
    cfg = load_config()
    summary = run_pipeline(cfg)
    print("Point-cloud geometric ML demo (OSS / learning only)")
    print(f"  classes     {summary['classes']}")
    print(f"  features    {summary['n_features']}  classifier={summary['classifier']}")
    print(f"  FPS / knn   {summary['fps_points']} pts, k={summary['knn_k']}")
    print(f"  voxel_size  {summary['voxel_size']}")
    print(f"  train_acc   {summary['train_acc']:.3f}  (n={summary['n_train']})")
    print(f"  test_acc    {summary['test_acc']:.3f}  (n={summary['n_test']})")
    print(f"  note        {summary['disclaimer']}")


if __name__ == "__main__":
    main()
