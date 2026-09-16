#!/usr/bin/env python3
"""CI-friendly eval CLI + JSON report."""

from __future__ import annotations

from pathlib import Path

from point_cloud_geo.eval import run_pipeline, write_report


def main() -> None:
    summary = run_pipeline()
    out = Path(__file__).resolve().parent / "last_report.json"
    write_report(summary, out)
    print("Geometric ML eval report")
    print(f"  classifier  {summary['classifier']}")
    print(f"  n_features  {summary['n_features']}")
    print(f"  train_acc   {summary['train_acc']:.3f}")
    print(f"  test_acc    {summary['test_acc']:.3f}")
    print(f"  wrote       {out}")
    # Soft gate: synthetic classes should be largely separable
    if summary["test_acc"] < 0.55:
        raise SystemExit(
            f"test_acc {summary['test_acc']:.3f} below soft floor 0.55 — check pipeline"
        )


if __name__ == "__main__":
    main()
