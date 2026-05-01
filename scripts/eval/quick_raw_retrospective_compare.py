#!/usr/bin/env python3
"""
Quick leak-free retrospective comparison: naive vs Ridge vs ensemble on raw DA only.

Uses a single engine pass (model_type='ensemble'): each fold already trains
XGB, RF, Ridge, and naive; we report metrics from naive_prediction,
predicted_da_linear, and ensemble_prediction.

Does not use panel imputation / dense parquet ``da`` fill quality as a metric.

Usage (repo root):
    python3 scripts/eval/quick_raw_retrospective_compare.py
    python3 scripts/eval/quick_raw_retrospective_compare.py --fraction 0.05 --seed 42
    python3 scripts/eval/quick_raw_retrospective_compare.py --output-json eval_results/quick_retro_compare.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# Repo root on sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return {"r2": float("nan"), "mae": float("nan"), "n": int(mask.sum())}
    yt = y_true[mask]
    yp = y_pred[mask]
    return {
        "r2": float(r2_score(yt, yp)),
        "mae": float(mean_absolute_error(yt, yp)),
        "n": int(len(yt)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.20,
        help="TEST_SAMPLE_FRACTION override (per-site draw before cap)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=40,
        help="Cap total retrospective rows after sampling (default 40 for a quick run; 0 = no cap)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-test-date",
        type=str,
        default="2008-01-01",
        help="Earliest raw test measurement dates",
    )
    parser.add_argument(
        "--n-anchors",
        type=int,
        default=50,
        help="Passed to validate_runtime_parameters only (sampling uses --fraction)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path to write full summary JSON",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use config default parallel retrospective (default: single-threaded for stability)",
    )
    args = parser.parse_args()

    import config as cfg

    orig = {
        "seed": cfg.RANDOM_SEED,
        "frac": getattr(cfg, "TEST_SAMPLE_FRACTION", 0.20),
        "parallel": getattr(cfg, "ENABLE_PARALLEL", True),
        "min_date": getattr(cfg, "MIN_TEST_DATE", "2003-01-01"),
    }
    cfg.RANDOM_SEED = args.seed
    cfg.TEST_SAMPLE_FRACTION = args.fraction
    cfg.MIN_TEST_DATE = args.min_test_date
    if not args.parallel:
        cfg.ENABLE_PARALLEL = False

    try:
        from forecasting.raw_forecast_engine import RawForecastEngine

        engine = RawForecastEngine(validate_on_init=False)
        engine.random_seed = args.seed
        cap = args.max_samples if args.max_samples > 0 else None
        df = engine.run_retrospective_evaluation(
            task="regression",
            model_type="ensemble",
            n_anchors=args.n_anchors,
            min_test_date=args.min_test_date,
            max_test_samples=cap,
        )
    finally:
        cfg.RANDOM_SEED = orig["seed"]
        cfg.TEST_SAMPLE_FRACTION = orig["frac"]
        cfg.ENABLE_PARALLEL = orig["parallel"]
        cfg.MIN_TEST_DATE = orig["min_date"]

    if df is None or df.empty:
        print("ERROR: No retrospective results (empty dataframe).")
        return 1

    y = df["actual_da"].astype(float).values
    cap_setting = args.max_samples if args.max_samples > 0 else None

    out = {
        "settings": {
            "test_sample_fraction": args.fraction,
            "max_test_samples": cap_setting,
            "seed": args.seed,
            "min_test_date": args.min_test_date,
            "n_rows": len(df),
        },
        "pooled": {},
        "per_site": {},
    }

    specs = [
        ("naive", "naive_prediction"),
        ("linear", "predicted_da_linear"),
        ("ensemble", "ensemble_prediction"),
    ]
    for name, col in specs:
        if col not in df.columns:
            print(f"WARNING: missing column {col} for {name}")
            continue
        pred = pd.to_numeric(df[col], errors="coerce").values
        out["pooled"][name] = _metrics(y, pred)

    for site in sorted(df["site"].unique()):
        sub = df[df["site"] == site]
        yt = sub["actual_da"].astype(float).values
        out["per_site"][site] = {}
        for name, col in specs:
            if col not in sub.columns:
                continue
            yp = pd.to_numeric(sub[col], errors="coerce").values
            out["per_site"][site][name] = _metrics(yt, yp)

    print("\n=== Quick raw retrospective (leak-free) ===")
    print(
        f"fraction={args.fraction}, max_samples={cap_setting or 'none'}, "
        f"seed={args.seed}, min_test_date={args.min_test_date}, n={len(df)}"
    )
    print("\nPooled R² / MAE / n:")
    for name, col in specs:
        m = out["pooled"].get(name, {})
        print(
            f"  {name:<10} R2={m.get('r2', float('nan')):.4f}  MAE={m.get('mae', float('nan')):.3f}  n={m.get('n', 0)}"
        )

    print("\nPer-site R² (naive | linear | ensemble):")
    for site in sorted(out["per_site"].keys()):
        row = out["per_site"][site]
        parts = []
        for name in ("naive", "linear", "ensemble"):
            r2 = row.get(name, {}).get("r2", float("nan"))
            parts.append(f"{r2:.3f}")
        print(f"  {site:<16} {' | '.join(parts)}")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
