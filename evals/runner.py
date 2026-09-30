#!/usr/bin/env python3
"""CI-friendly eval CLI + JSON report (includes FPS vs random compare table)."""

from __future__ import annotations

from pathlib import Path

from point_cloud_geo.config import load_config
from point_cloud_geo.eval import compare_sampling, run_pipeline, write_report
from point_cloud_geo.registration import icp_recovery


def main() -> None:
    summary = run_pipeline()
    compare = compare_sampling()
    registration = icp_recovery(load_config())
    out_payload = {**summary, **compare, **registration}
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
    print(f"  wrote       {out}")
    small = [r for r in registration["icp_recovery"] if r["case"] == "small"]
    if any(r["rotation_error_deg"] > 1.0 or not r["converged"] for r in small):
        raise SystemExit("ICP failed to recover the small known transform — check registration")
    if summary["test_acc"] < 0.55:
        raise SystemExit(
            f"test_acc {summary['test_acc']:.3f} below soft floor 0.55 — check pipeline"
        )


if __name__ == "__main__":
    main()
