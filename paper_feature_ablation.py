#!/usr/bin/env python3
"""
DATect Individual Feature Ablation Study

Tests the impact of removing individual environmental features and feature
groups that have low importance scores. This complements paper_ablation_study.py
which tested architectural components (lags, derived features, etc.) by testing
specific environmental data sources.

Each ablation drops a feature (or group) via DATECT_EXTRA_DROP_FEATURES and
runs the full retrospective evaluation in a subprocess.

Usage (run on Hyak):
    python3 paper_feature_ablation.py

Output: paper_feature_ablation_results.json
"""

import json
import os
import subprocess
import sys
import warnings

warnings.filterwarnings('ignore')


# ── Ablation experiments ──────────────────────────────────────────────────
# Each entry: (name, list of features to drop)
# Ordered from most suspicious (lowest importance) to less suspicious.

ABLATIONS = [
    # --- Individual satellite features ---
    ("No modis-k490 + k490_squared",
     ["modis-k490", "k490_squared"]),

    ("No fluor_efficiency",
     ["fluor_efficiency"]),

    ("No modis-chla",
     ["modis-chla"]),

    ("No chla-anom",
     ["chla-anom"]),

    ("No modis-flr",
     ["modis-flr"]),

    ("No sst-anom",
     ["sst-anom"]),

    # --- PN features ---
    ("No PN features (pn_log + pn_above_threshold)",
     ["pn_log", "pn_above_threshold"]),

    # --- Individual derived features ---
    ("No beuti_relaxation",
     ["beuti_relaxation"]),

    ("No beuti_squared",
     ["beuti_squared"]),

    ("No pdo_oni_phase",
     ["pdo_oni_phase"]),

    ("No mhw_flag",
     ["mhw_flag"]),

    # --- Compound: all satellite-derived + K490 + fluor ---
    ("No satellite-derived (chla-anom, sst-anom, modis-flr, modis-k490, k490_squared, fluor_efficiency)",
     ["chla-anom", "sst-anom", "modis-flr", "modis-k490", "k490_squared", "fluor_efficiency"]),

    # --- Rolling features (already partially in ZERO_IMPORTANCE) ---
    ("No rolling std features",
     ["raw_obs_roll_std_4", "raw_obs_roll_std_8", "raw_obs_roll_std_12"]),

    # --- Temporal: days_since_start (linear trend proxy) ---
    ("No days_since_start",
     ["days_since_start"]),

    # --- Compound: everything below importance 0.005 ---
    ("No low-importance features (all < 0.005 avg importance)",
     ["modis-k490", "k490_squared", "fluor_efficiency", "chla-anom",
      "pdo_oni_phase", "beuti_relaxation", "pn_above_threshold",
      "modis-chla"]),
]


# ── Subprocess script ─────────────────────────────────────────────────────
SUBPROCESS_SCRIPT = '''
import warnings
warnings.filterwarnings('ignore')

import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import config
print(f"Config: ZERO_IMPORTANCE len={len(config.ZERO_IMPORTANCE_FEATURES)}", file=sys.stderr)
print(f"Config: dropped extras = {config.ZERO_IMPORTANCE_FEATURES[-10:]}", file=sys.stderr)

from forecasting.raw_forecast_engine import RawForecastEngine
engine = RawForecastEngine(validate_on_init=False)

results_df = engine.run_retrospective_evaluation(
    task="regression",
    model_type="ensemble",
    n_anchors=getattr(config, "N_RANDOM_ANCHORS", 500),
    min_test_date="2008-01-01"
)

if results_df is None or results_df.empty:
    print(json.dumps({"error": "no results"}))
    sys.exit(1)

y_true = results_df["actual_da"].values
y_pred = results_df["predicted_da"].values

overall = {
    "r2": float(r2_score(y_true, y_pred)),
    "mae": float(mean_absolute_error(y_true, y_pred)),
    "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    "n": len(y_true),
}

per_site = {}
for site in sorted(results_df["site"].unique()):
    mask = results_df["site"] == site
    yt = results_df.loc[mask, "actual_da"].values
    yp = results_df.loc[mask, "predicted_da"].values
    per_site[site] = {
        "r2": float(r2_score(yt, yp)),
        "mae": float(mean_absolute_error(yt, yp)),
        "n": int(mask.sum()),
    }

print(json.dumps({"overall": overall, "per_site": per_site}))
'''


def run_ablation(name, features_to_drop):
    """Run a single feature ablation in a subprocess."""
    print(f"\n{'='*60}")
    print(f"ABLATION: {name}")
    print(f"  Dropping: {features_to_drop}")
    print(f"{'='*60}")

    env = os.environ.copy()
    env["DATECT_EXTRA_DROP_FEATURES"] = ",".join(features_to_drop)

    result = subprocess.run(
        [sys.executable, "-c", SUBPROCESS_SCRIPT],
        env=env,
        stdout=subprocess.PIPE,
        stderr=None,  # Show progress bars live
        text=True,
        timeout=7200,
    )

    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode})")
        return None

    try:
        lines = result.stdout.strip().split("\n")
        json_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith("{"):
                json_start = i
                break

        if json_start is None:
            print(f"  ERROR: No JSON in output")
            print(result.stdout[-500:])
            return None

        json_str = "\n".join(lines[json_start:])
        data = json.loads(json_str)

        overall = data['overall']
        print(f"  Overall: R²={overall['r2']:.4f}, MAE={overall['mae']:.2f}, N={overall['n']}")
        for site, m in data.get('per_site', {}).items():
            print(f"    {site:<18} R²={m['r2']:.4f}, MAE={m['mae']:.2f}, N={m['n']}")

        return data

    except (json.JSONDecodeError, KeyError) as e:
        print(f"  ERROR parsing output: {e}")
        print(result.stdout[-500:])
        return None


def main():
    # ── Baseline (no extra drops) ─────────────────────────────────────────
    print("\n" + "="*60)
    print("BASELINE: Full DATect (no extra drops)")
    print("="*60)

    env = os.environ.copy()
    env.pop("DATECT_EXTRA_DROP_FEATURES", None)

    result = subprocess.run(
        [sys.executable, "-c", SUBPROCESS_SCRIPT],
        env=env,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        timeout=7200,
    )

    baseline = None
    if result.returncode == 0:
        lines = result.stdout.strip().split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith("{"):
                json_str = "\n".join(lines[i:])
                baseline = json.loads(json_str)
                break

    if baseline:
        overall = baseline['overall']
        print(f"  Overall: R²={overall['r2']:.4f}, MAE={overall['mae']:.2f}, N={overall['n']}")
    else:
        print("  BASELINE FAILED — aborting")
        sys.exit(1)

    # ── Run all ablations ─────────────────────────────────────────────────
    all_results = {"baseline": baseline}

    for name, features in ABLATIONS:
        key = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        key = key.replace(",", "").replace("+", "and").replace("<", "lt")
        data = run_ablation(name, features)
        all_results[key] = data

    # ── Save results ──────────────────────────────────────────────────────
    with open('paper_feature_ablation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # ── Summary table ─────────────────────────────────────────────────────
    baseline_r2 = baseline['overall']['r2']
    baseline_mae = baseline['overall']['mae']

    print("\n" + "="*75)
    print("FEATURE ABLATION SUMMARY")
    print("="*75)
    print(f"{'Configuration':<55} {'R²':>7} {'ΔR²':>8} {'MAE':>7} {'ΔMAE':>7}")
    print("-"*75)
    print(f"{'Baseline (full DATect)':<55} {baseline_r2:>7.4f} {'---':>8} {baseline_mae:>7.2f} {'---':>7}")

    for name, features in ABLATIONS:
        key = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        key = key.replace(",", "").replace("+", "and").replace("<", "lt")
        result = all_results.get(key)
        if result is None:
            print(f"{name:<55} {'FAILED':>7}")
            continue
        r2 = result['overall']['r2']
        mae = result['overall']['mae']
        dr2 = r2 - baseline_r2
        dmae = mae - baseline_mae
        print(f"{name:<55} {r2:>7.4f} {dr2:>+8.4f} {mae:>7.2f} {dmae:>+7.2f}")

    print()

    # ── Flag features safe to remove ──────────────────────────────────────
    print("\nFEATURES SAFE TO REMOVE (|ΔR²| < 0.005 and ΔMAE < 0.10):")
    print("-"*60)
    for name, features in ABLATIONS:
        key = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        key = key.replace(",", "").replace("+", "and").replace("<", "lt")
        result = all_results.get(key)
        if result is None:
            continue
        r2 = result['overall']['r2']
        mae = result['overall']['mae']
        dr2 = r2 - baseline_r2
        dmae = mae - baseline_mae
        if abs(dr2) < 0.005 and dmae < 0.10:
            print(f"  ✓ {name}")
            print(f"    Features: {features}")
            print(f"    ΔR²={dr2:+.4f}, ΔMAE={dmae:+.2f}")

    print("\nFEATURES THAT HURT WHEN REMOVED (ΔR² < -0.005):")
    print("-"*60)
    for name, features in ABLATIONS:
        key = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        key = key.replace(",", "").replace("+", "and").replace("<", "lt")
        result = all_results.get(key)
        if result is None:
            continue
        r2 = result['overall']['r2']
        dr2 = r2 - baseline_r2
        if dr2 < -0.005:
            print(f"  ✗ {name}")
            print(f"    Features: {features}")
            print(f"    ΔR²={dr2:+.4f}")

    print(f"\nSaved to paper_feature_ablation_results.json")


if __name__ == "__main__":
    main()
