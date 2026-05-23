#!/usr/bin/env python3
"""
Validate tuned hyperparameters on the 2019+ temporal holdout (final unbiased test).

Runs the retrospective forecast with proposed_overrides.json applied, then
splits results into:
  - TUNING window (< 2019-01-01): the data Optuna saw.
  - HOLDOUT window (>= 2019-01-01): completely untouched by tuning.

A tuned config is considered "real" only if it improves on BOTH windows.
If it improves only on tuning and degrades on holdout → overfitting; discard.

Usage:
  # Baseline run (current per_site_models.py values):
  python scripts/eval/validate_tuned_on_holdout.py --label baseline

  # Tuned run (apply proposed_overrides.json):
  DATECT_HPARAM_OVERRIDE_JSON=proposed_overrides.json \
      python scripts/eval/validate_tuned_on_holdout.py --label tuned

  # Compare the two:
  python scripts/eval/validate_tuned_on_holdout.py --compare baseline tuned
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

try:
    from _repo import ensure_repo_root
    ensure_repo_root()
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Canonical split loaded from config/tuned_hyperparameters.json
from forecasting.tuned_config import get_eval_windows as _get_eval_windows
_w = _get_eval_windows()
VAL_START = pd.Timestamp(_w["validation_start"])
VAL_END   = pd.Timestamp(_w["validation_end"])  # holdout starts here


def run_eval(label: str, output_dir: str = "holdout_validation") -> str:
    import config
    from forecasting.raw_forecast_engine import RawForecastEngine

    engine = RawForecastEngine(validate_on_init=False)
    results_df = engine.run_retrospective_evaluation(
        task="regression", model_type="ensemble",
        n_anchors=getattr(config, "N_RANDOM_ANCHORS", 500),
        min_test_date="2008-01-01",
    )
    if results_df is None or results_df.empty:
        raise RuntimeError("No retrospective results")
    results_df["date"] = pd.to_datetime(results_df["date"])
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out = Path(output_dir) / f"{label}_predictions.parquet"
    results_df.to_parquet(out, index=False)
    print(f"  Saved {len(results_df)} rows → {out}")
    return str(out)


def metrics_table(predictions_path: str) -> dict:
    df = pd.read_parquet(predictions_path)
    df["date"] = pd.to_datetime(df["date"])

    out = {"per_site": {}, "overall": {}, "windows": {}}
    for window_name, mask in [
        ("pretrain_pre2019",  df["date"] < VAL_START),
        ("validation_2019_2022", (df["date"] >= VAL_START) & (df["date"] < VAL_END)),
        ("holdout_2022plus",  df["date"] >= VAL_END),
        ("all", pd.Series([True] * len(df), index=df.index)),
    ]:
        sub = df[mask]
        if len(sub) < 5:
            continue
        out["windows"][window_name] = {
            "r2": float(r2_score(sub["actual_da"], sub["predicted_da"])),
            "mae": float(mean_absolute_error(sub["actual_da"], sub["predicted_da"])),
            "n": int(len(sub)),
        }
        for site in sorted(sub["site"].unique()):
            ssub = sub[sub["site"] == site]
            if len(ssub) < 5:
                continue
            out["per_site"].setdefault(site, {})[window_name] = {
                "r2": float(r2_score(ssub["actual_da"], ssub["predicted_da"])),
                "mae": float(mean_absolute_error(ssub["actual_da"], ssub["predicted_da"])),
                "n": int(len(ssub)),
            }
    return out


def compare(baseline_path: str, tuned_path: str) -> int:
    b = metrics_table(baseline_path)
    t = metrics_table(tuned_path)

    print()
    print("=" * 78)
    print("WINDOW-LEVEL COMPARISON (baseline → tuned)")
    print("=" * 78)
    print(f"{'Window':<26} {'baseline R²':>12} {'tuned R²':>12} {'Δ R²':>10} {'N':>6}")
    print("-" * 78)
    for w in ("pretrain_pre2019", "validation_2019_2022", "holdout_2022plus", "all"):
        bw = b["windows"].get(w, {})
        tw = t["windows"].get(w, {})
        if not bw or not tw:
            continue
        d = tw["r2"] - bw["r2"]
        print(f"{w:<26} {bw['r2']:>12.4f} {tw['r2']:>12.4f} {d:>+10.4f} {bw['n']:>6d}")

    print()
    print("VERDICT (verdict uses validation 2019-2022 vs holdout 2022-2024)")
    print("-" * 78)
    bt = b["windows"].get("validation_2019_2022", {})
    tt = t["windows"].get("validation_2019_2022", {})
    bh = b["windows"].get("holdout_2022plus", {})
    th = t["windows"].get("holdout_2022plus", {})
    tune_delta = tt.get("r2", 0) - bt.get("r2", 0)
    hold_delta = th.get("r2", 0) - bh.get("r2", 0)

    if tune_delta > 0.005 and hold_delta > 0.005:
        print("  ✅ TUNED CONFIG IS REAL: improves on validation AND holdout.")
        print("     Recommend: merge proposed_overrides.json into per_site_models.py.")
    elif tune_delta > 0.005 and hold_delta < -0.005:
        print("  ❌ OVERFITTING: validation improved but holdout degraded.")
        print("     Discard tuned config. The tuning loop fit noise.")
    elif tune_delta > 0.005 and abs(hold_delta) <= 0.005:
        print("  🟡 NEUTRAL ON HOLDOUT: validation improved, holdout unchanged.")
        print("     Marginal value; review per-site detail before merging.")
    else:
        print("  ⚪ NO MEANINGFUL CHANGE on either window. Tuning didn't help.")

    print()
    print("PER-SITE DETAIL (holdout 2022-2024 only — the unbiased number)")
    print("-" * 78)
    rows = []
    for site in sorted(b["per_site"].keys()):
        bh_site = b["per_site"].get(site, {}).get("holdout_2022plus")
        th_site = t["per_site"].get(site, {}).get("holdout_2022plus")
        if not bh_site or not th_site:
            continue
        rows.append({
            "site": site,
            "baseline_R2": bh_site["r2"],
            "tuned_R2": th_site["r2"],
            "delta_R2": th_site["r2"] - bh_site["r2"],
            "N": bh_site["n"],
        })
    df = pd.DataFrame(rows).sort_values("delta_R2", ascending=False)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(df.to_string(index=False))
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--label", type=str, default=None,
                   help="Run eval and save predictions under this label")
    p.add_argument("--compare", nargs=2, metavar=("BASELINE", "TUNED"),
                   help="Compare two labels (uses holdout_validation/<label>_predictions.parquet)")
    p.add_argument("--output-dir", type=str, default="holdout_validation")
    args = p.parse_args()

    if args.compare:
        b = Path(args.output_dir) / f"{args.compare[0]}_predictions.parquet"
        t = Path(args.output_dir) / f"{args.compare[1]}_predictions.parquet"
        if not b.exists() or not t.exists():
            print(f"Missing: {b} or {t}")
            return 1
        return compare(str(b), str(t))

    if args.label:
        run_eval(args.label, args.output_dir)
        return 0

    print("Specify --label LABEL or --compare BASELINE TUNED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
