#!/usr/bin/env python3
"""
Chronological deterministic retrospective evaluation.

Unlike ``multi_seed_baseline.py`` (which samples ``TEST_SAMPLE_FRACTION`` of
real DA measurements at a given random seed and then filters by date),
this script uses **every** real DA measurement in the requested date range
as a test point. Output is a single deterministic number per metric per
site — no seed dimension, no sampling noise.

Why both protocols exist:

* Multi-seed bootstrap (``multi_seed_baseline.py``): gives a noise estimate
  via cross-seed dispersion. Useful for "how much would a different sample
  draw change the headline." 5 seeds × 20% sample → ~1200 anchors / seed,
  ~160 per seed in the post-2022 holdout window.

* Chronological deterministic (this script): uses 100% of real DA in the
  window, no sampling. Result is one number per metric. Row-level bootstrap
  CIs (``B=10,000``) quantify variance over which rows happen to be
  measured, but not over sampling choice. Cleaner for paper claims about
  generalization to new data; less suited for sampling-noise estimation.

Default windows match the rest of the project:

    validation = [2019-01-01, 2022-01-01)   # used for tuning objectives
    holdout    = [2022-01-01, 2024-01-01)   # the unbiased headline

Usage:
    python scripts/eval/chronological_eval.py
    python scripts/eval/chronological_eval.py --window 2022-01-01:2024-01-01
    python scripts/eval/chronological_eval.py --task regression --model ensemble
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    recall_score, precision_score, fbeta_score, f1_score,
)

# Repo root on sys.path so ``import config`` works from anywhere
try:
    from _repo import ensure_repo_root
    ensure_repo_root()
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config  # noqa: E402
from forecasting.raw_forecast_engine import RawForecastEngine  # noqa: E402


def _bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, metric_fn, B: int = 10_000,
                  alpha: float = 0.05, rng_seed: int = 42) -> tuple[float, float, float]:
    """Row-level percentile bootstrap CI for a regression metric."""
    rng = np.random.default_rng(rng_seed)
    n = len(y_true)
    if n < 5:
        return float("nan"), float("nan"), float("nan")
    samples = np.empty(B, dtype=float)
    for i in range(B):
        idx = rng.integers(0, n, size=n)
        samples[i] = metric_fn(y_true[idx], y_pred[idx])
    lo = float(np.percentile(samples, 100 * (alpha / 2)))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return float(metric_fn(y_true, y_pred)), lo, hi


def run_chronological(window_start: str, window_end: str, task: str, model_type: str,
                       outdir: Path) -> dict:
    """Run a deterministic retrospective over every real DA measurement in window."""
    print(f"\n=== chronological_eval: {window_start} to {window_end} "
          f"({task}, {model_type}) ===", flush=True)

    engine = RawForecastEngine(validate_on_init=False)
    # Patch the engine's anchor sampling: instead of random-fraction sampling,
    # use every real DA measurement in window. We do this by overriding
    # ``TEST_SAMPLE_FRACTION=1.0`` and then filtering by date after the run.
    # (Cleaner than reaching into engine internals.)
    saved = config.TEST_SAMPLE_FRACTION
    try:
        config.TEST_SAMPLE_FRACTION = 1.0
        results = engine.run_retrospective_evaluation(
            task=task, model_type=model_type,
            n_anchors=getattr(config, "N_RANDOM_ANCHORS", 9999),
            min_test_date=window_start,
        )
    finally:
        config.TEST_SAMPLE_FRACTION = saved

    if results is None or results.empty:
        raise RuntimeError("No retrospective results returned")
    results["date"] = pd.to_datetime(results["date"])
    in_window = (results["date"] >= window_start) & (results["date"] < window_end)
    sub = results[in_window].copy()

    out: dict = {
        "window": {"start": window_start, "end": window_end},
        "n_total_real_da_in_window": int(in_window.sum()),
        "task": task, "model_type": model_type,
        "per_site": {},
        "overall": {},
    }

    if task == "regression":
        for label, mask in [("overall", pd.Series([True] * len(sub), index=sub.index))] + [
            (f"site::{site}", sub["site"] == site) for site in sorted(sub["site"].unique())
        ]:
            ssub = sub[mask]
            if len(ssub) < 5:
                continue
            y, p = ssub["actual_da"].astype(float).values, ssub["predicted_da"].astype(float).values
            yt = (y > config.SPIKE_THRESHOLD).astype(int)
            yp = (p > config.SPIKE_REGRESSION_ALERT_THRESHOLD).astype(int)
            block = {
                "n": int(len(ssub)),
                "n_spikes": int(yt.sum()),
                "r2": _bootstrap_ci(y, p, r2_score),
                "mae": _bootstrap_ci(y, p, mean_absolute_error),
                "rmse": _bootstrap_ci(y, p, lambda a, b: float(np.sqrt(mean_squared_error(a, b)))),
            }
            if yt.sum() >= 1:
                block.update({
                    "spike_recall": float(recall_score(yt, yp, zero_division=0)),
                    "spike_precision": float(precision_score(yt, yp, zero_division=0)),
                    "spike_f2": float(fbeta_score(yt, yp, beta=2, zero_division=0)),
                })
            if label == "overall":
                out["overall"] = block
            else:
                out["per_site"][label.replace("site::", "")] = block

    outdir.mkdir(parents=True, exist_ok=True)
    win_tag = f"{window_start.replace('-', '')}_{window_end.replace('-', '')}"
    out_path = outdir / f"chronological_{task}_{model_type}_{win_tag}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    sub.to_parquet(outdir / f"chronological_{task}_{model_type}_{win_tag}.parquet", index=False)
    print(f"  Saved → {out_path}", flush=True)

    if task == "regression" and "overall" in out:
        ov = out["overall"]
        r2 = ov.get("r2", (None,))[0]
        mae = ov.get("mae", (None,))[0]
        n = ov.get("n", 0)
        ns = ov.get("n_spikes", 0)
        print(f"\n  N = {n} (spikes = {ns})", flush=True)
        if r2 is not None:
            print(f"  R²  = {r2:+.3f} [{ov['r2'][1]:+.3f}, {ov['r2'][2]:+.3f}]", flush=True)
            print(f"  MAE = {mae:.2f}  [{ov['mae'][1]:.2f}, {ov['mae'][2]:.2f}]", flush=True)
        if ns >= 1:
            print(f"  spike recall = {ov['spike_recall']:.3f}", flush=True)
            print(f"  spike F2     = {ov['spike_f2']:.3f}", flush=True)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--window", default="2022-01-01:2024-01-01",
                   help="Date range as 'YYYY-MM-DD:YYYY-MM-DD' (default: post-2022 holdout)")
    p.add_argument("--task", default="regression", choices=["regression", "classification"])
    p.add_argument("--model", default="ensemble")
    p.add_argument("--out-dir", default="eval_outputs/chronological")
    args = p.parse_args()

    start, end = args.window.split(":")
    run_chronological(start, end, args.task, args.model, Path(args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
