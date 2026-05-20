#!/usr/bin/env python3
"""
Aggregate per-site Optuna tuning results into a proposed per_site_models.py diff.

Reads tuning_results/<site_slug>/best_params.json (produced by
tune_per_site_hyperparams.py) and emits:
  1. A summary table (current R² vs tuned R²) — needs baseline to compare against.
  2. A Python snippet that can be pasted into per_site_models.py.
  3. A JSON file (proposed_overrides.json) usable as DATECT_HPARAM_OVERRIDE_JSON.

Usage:
  python scripts/eval/aggregate_tuning_results.py \
      --baseline paper_ablation_results.json \
      --tuning-root tuning_results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SITES = [
    "Copalis", "Kalaloch", "Twin Harbors", "Quinault", "Long Beach",
    "Clatsop Beach", "Coos Bay", "Cannon Beach", "Gold Beach", "Newport",
]


def slug(site: str) -> str:
    return site.lower().replace(" ", "_")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning-root", type=str, default="tuning_results")
    parser.add_argument("--baseline", type=str, default=None,
                        help="paper_ablation_results.json for current per-site R²")
    parser.add_argument("--out-json", type=str, default="proposed_overrides.json")
    args = parser.parse_args()

    root = Path(args.tuning_root)
    rows = []
    overrides = {}

    for site in SITES:
        f = root / slug(site) / "best_params.json"
        if not f.exists():
            print(f"  [skip] {site}: no best_params.json")
            continue
        rec = json.loads(f.read_text())
        rows.append({
            "site": site,
            "tuned_R2": rec["best_value_r2"],
            "tuned_MAE": rec["best_mae"],
            "n_test": rec["n_test_points"],
            "n_trials": rec["n_trials_completed"],
            "elapsed_min": round(rec["elapsed_seconds"] / 60, 1),
        })
        p = rec["best_params"]
        clip_max_raw = p.get("clip_max", "none")
        clip_max = None if clip_max_raw == "none" else float(clip_max_raw)
        overrides[site] = {
            "xgb_params": {
                "max_depth": p["xgb_max_depth"],
                "n_estimators": p["xgb_n_estimators"],
                "learning_rate": p["xgb_lr"],
                "min_child_weight": p["xgb_mcw"],
                "reg_alpha": p["xgb_alpha"],
                "reg_lambda": p["xgb_lambda"],
                "gamma": p["xgb_gamma"],
                "subsample": p["xgb_subsample"],
                "colsample_bytree": p["xgb_colsample"],
                "colsample_bylevel": p.get("xgb_colsample_bylevel", 0.8),
            },
            "rf_params": {
                "n_estimators": p["rf_n_estimators"],
                "max_depth": p["rf_max_depth"],
                "min_samples_split": p["rf_min_split"],
                "min_samples_leaf": p["rf_min_leaf"],
                "max_features": p["rf_max_features"],
            },
            "param_grid": [{
                "max_depth": p["xgb_max_depth"],
                "n_estimators": p["xgb_n_estimators"],
                "learning_rate": p["xgb_lr"],
                "min_child_weight": p["xgb_mcw"],
            }],
            "ensemble_weights": [p["w_xgb"], 1.0 - p["w_xgb"], 0.0],
            "prediction_clip_q": p["clip_q"],
            "prediction_clip_max": clip_max,
        }

    df = pd.DataFrame(rows)
    if args.baseline:
        bdata = json.loads(Path(args.baseline).read_text())
        bsite = bdata.get("baseline", {}).get("per_site", {})
        df["baseline_R2"] = df["site"].map(lambda s: bsite.get(s, {}).get("r2"))
        df["delta_R2"] = df["tuned_R2"] - df["baseline_R2"]
        df = df.sort_values("delta_R2", ascending=False)

    pd.set_option("display.float_format", "{:.4f}".format)
    print(df.to_string(index=False))
    Path(args.out_json).write_text(json.dumps(overrides, indent=2))
    print(f"\nWrote {args.out_json}  ({len(overrides)} sites)")
    print(f"To validate: DATECT_HPARAM_OVERRIDE_JSON={args.out_json} python scripts/eval/eval_paper_metrics.py")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
