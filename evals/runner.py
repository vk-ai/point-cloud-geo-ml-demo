#!/usr/bin/env python3
"""CI-friendly eval CLI + JSON report (includes FPS vs random compare table)."""

from __future__ import annotations

from pathlib import Path

from point_cloud_geo.config import load_config
from point_cloud_geo.eval import compare_sampling, run_pipeline, write_report
from point_cloud_geo.registration import icp_recovery
from point_cloud_geo.segmentation import plane_segmentation_eval


def main() -> None:
    summary = run_pipeline()
    compare = compare_sampling()
    registration = icp_recovery(load_config())
    planes = plane_segmentation_eval(load_config())
    out_payload = {**summary, **compare, **registration, **planes}
    out = Path(__file__).resolve().parent / "last_report.json"
    write_report(out_payload, out)
    print("Geometric ML eval report")
    print(f"  classifier  {summary['classifier']}")
    print(f"  sampling    {summary['sampling']}")
    print(f"  n_features  {summary['n_features']}")
    print(f"  train_acc   {summary['train_acc']:.3f}")
    print(f"  test_acc    {summary['test_acc']:.3f}")
    print(
        f"  coverage    mean_nn={summary['coverage_mean_nn_spacing']:.4f}  "
        f"bbox_fill={summary['coverage_bbox_fill_ratio']:.3f}"
    )
    print("  FPS vs random compare:")
    for row in compare["sampling_compare"]:
        print(
            f"    {row['sampling']:6s}  test_acc={row['test_acc']:.3f}  "
            f"mean_nn={row['coverage_mean_nn_spacing']:.4f}  "
            f"bbox_fill={row['coverage_bbox_fill_ratio']:.3f}"
        )
    print(f"  ICP recovery ({registration['shape']}, known rigid transform):")
    for r in registration["icp_recovery"]:
        print(
            f"    {r['case']:5s} {r['angle_deg']:5.1f}°  {r['method']:14s}  "
            f"rot_err={r['rotation_error_deg']:7.3f}°  t_err={r['translation_error']:.4f}  "
            f"fitness={r['fitness']:.3f}  inlier_rmse={r['inlier_rmse']:.4f}  "
            f"iters={r['n_iter']:2d}  converged={r['converged']}"
        )
    ps = planes["plane_segmentation"]
    print(
        f"  RANSAC planes ({ps['n_points']} pts incl. {ps['n_outliers']} outliers, "
        f"thresh={ps['distance_threshold']}, seed={ps['seed']}):"
    )
    for r in ps["planes"]:
        a, b, c, d = r["plane_model"]
        print(
            f"    plane {r['plane_id']}  abcd=({a:+.3f},{b:+.3f},{c:+.3f},{d:+.3f})  "
            f"angle_err={r['normal_angle_err_deg']:.3f}°  d_err={r['offset_err']:.4f}  "
            f"inliers={r['n_inliers']}  precision={r['precision']:.3f}  "
            f"recall={r['recall']:.3f}  iters={r['n_iter']}"
        )
    print(
        f"    same seed → identical: {ps['deterministic_same_seed']}  "
        f"inliers by seed {ps['inliers_by_seed']}"
    )
    print(f"  wrote       {out}")
    small = [r for r in registration["icp_recovery"] if r["case"] == "small"]
    if any(r["rotation_error_deg"] > 1.0 or not r["converged"] for r in small):
        raise SystemExit("ICP failed to recover the small known transform — check registration")
    if not ps["deterministic_same_seed"] or len(ps["planes"]) != 2 or any(
        r["normal_angle_err_deg"] > 2.0 or r["recall"] < 0.9 for r in ps["planes"]
    ):
        raise SystemExit("RANSAC plane segmentation failed on the synthetic scene")
    if summary["test_acc"] < 0.55:
        raise SystemExit(
            f"test_acc {summary['test_acc']:.3f} below soft floor 0.55 — check pipeline"
        )


if __name__ == "__main__":
    main()
