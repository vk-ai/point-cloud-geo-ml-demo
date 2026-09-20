#!/usr/bin/env python3
"""CI-friendly eval CLI + JSON report (includes FPS vs random compare table)."""

from __future__ import annotations

from pathlib import Path

from point_cloud_geo.eval import compare_sampling, run_pipeline, write_report


def main() -> None:
    summary = run_pipeline()
    compare = compare_sampling()
    out_payload = {**summary, **compare}
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
    print(f"  wrote       {out}")
    if summary["test_acc"] < 0.55:
        raise SystemExit(
            f"test_acc {summary['test_acc']:.3f} below soft floor 0.55 — check pipeline"
        )


if __name__ == "__main__":
    main()
