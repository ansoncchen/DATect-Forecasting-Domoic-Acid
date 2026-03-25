#!/usr/bin/env python3
"""
DATect Ablation Study for Paper

Runs 5 ablation experiments, each disabling one key component:
  1. No interpolated training (USE_INTERPOLATED_TRAINING=False)
  2. No per-site customization (USE_PER_SITE_MODELS=False)
  3. No observation-order lags (LAG_FEATURES=[])
  4. No derived features (drop MHW, BEUTI², PDO-ONI phase, PN tipping, fluor, K490²)
  5. No naive in ensemble (naive weight → 0, renormalize XGB+RF)

FIX: Uses environment variables instead of config mutations so loky workers
(which spawn fresh processes) inherit the correct settings. Each experiment
runs as a subprocess to guarantee a clean config import.

Usage (run on Hyak):
    python3 paper_ablation_study.py

Output: paper_ablation_results.json
"""

import json
import os
import subprocess
import sys
import warnings
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')


# ── Derived features to disable in ablation 4 ─────────────────────────────
DERIVED_FEATURES = [
    'mhw_flag', 'beuti_squared', 'beuti_relaxation',
    'pdo_oni_phase', 'fluor_efficiency', 'k490_squared',
    'pn_log', 'pn_above_threshold',
]


def run_ablation(name, env_overrides=None, naive_zero=False):
    """Run a single ablation experiment in a subprocess for clean config."""
    print(f"\n{'='*60}")
    print(f"ABLATION: {name}")
    print(f"{'='*60}")

    if env_overrides:
        for k, v in env_overrides.items():
            print(f"  ENV: {k}={v}")

    if naive_zero:
        print(f"  Naive weights zeroed (in-process mutation)")

    # For most ablations, use subprocess to get clean config from env vars.
    # For naive ablation, must mutate per_site_models in-process.
    if naive_zero:
        return _run_naive_ablation()
    else:
        return _run_subprocess_ablation(name, env_overrides or {})


def _run_subprocess_ablation(name, env_overrides):
    """Run ablation in a subprocess so config.py reads env vars fresh."""
    env = os.environ.copy()
    env.update(env_overrides)

    # Run the helper script
    result = subprocess.run(
        [sys.executable, "-c", SUBPROCESS_SCRIPT],
        env=env,
        stdout=subprocess.PIPE,
        stderr=None,  # Show progress bars live
        text=True,
        timeout=7200,  # 2 hour timeout per experiment
    )

    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode})")
        return None

    # Parse JSON output from subprocess
    try:
        # Find the JSON block in stdout
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

        # Print results
        overall = data['overall']
        print(f"  Overall: R²={overall['r2']:.4f}, MAE={overall['mae']:.2f}, N={overall['n']}")
        for site, m in data.get('per_site', {}).items():
            print(f"    {site:<18} R²={m['r2']:.4f}, MAE={m['mae']:.2f}, N={m['n']}")

        return data

    except (json.JSONDecodeError, KeyError) as e:
        print(f"  ERROR parsing output: {e}")
        print(result.stdout[-500:])
        return None


# Script that runs inside each subprocess
SUBPROCESS_SCRIPT = '''
import warnings
warnings.filterwarnings('ignore')

import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import config
# Print config state for debugging
print(f"Config: USE_INTERPOLATED_TRAINING={config.USE_INTERPOLATED_TRAINING}", file=sys.stderr)
print(f"Config: USE_PER_SITE_MODELS={config.USE_PER_SITE_MODELS}", file=sys.stderr)
print(f"Config: LAG_FEATURES={config.LAG_FEATURES}", file=sys.stderr)
print(f"Config: ZERO_IMPORTANCE len={len(config.ZERO_IMPORTANCE_FEATURES)}", file=sys.stderr)

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


def _run_naive_ablation():
    """Naive ablation requires in-process mutation of per_site_models dicts."""
    import config
    from forecasting import per_site_models as psm
    from forecasting.raw_forecast_engine import RawForecastEngine

    # Save originals
    orig_site_configs = deepcopy(psm.SITE_SPECIFIC_CONFIGS)
    orig_default = deepcopy(psm.DEFAULT_SITE_CONFIG)

    # Zero out naive weights, renormalize XGB+RF
    for site_name, site_cfg in psm.SITE_SPECIFIC_CONFIGS.items():
        weights = site_cfg.get('ensemble_weights')
        w_xgb, w_rf, w_naive = weights if weights is not None else (0.40, 0.40, 0.20)
        total = w_xgb + w_rf
        if total > 0:
            site_cfg['ensemble_weights'] = (w_xgb / total, w_rf / total, 0.0)
        else:
            site_cfg['ensemble_weights'] = (0.5, 0.5, 0.0)
    default_weights = psm.DEFAULT_SITE_CONFIG.get('ensemble_weights')
    w_xgb, w_rf, w_naive = default_weights if default_weights is not None else (0.45, 0.35, 0.20)
    total = w_xgb + w_rf
    psm.DEFAULT_SITE_CONFIG['ensemble_weights'] = (w_xgb / total, w_rf / total, 0.0)

    try:
        engine = RawForecastEngine(validate_on_init=False)
        results_df = engine.run_retrospective_evaluation(
            task="regression",
            model_type="ensemble",
            n_anchors=getattr(config, 'N_RANDOM_ANCHORS', 500),
            min_test_date="2008-01-01"
        )
    finally:
        # Restore originals
        psm.SITE_SPECIFIC_CONFIGS = orig_site_configs
        psm.DEFAULT_SITE_CONFIG = orig_default

    if results_df is None or results_df.empty:
        print("  ERROR: No results")
        return None

    y_true = results_df['actual_da'].values
    y_pred = results_df['predicted_da'].values

    overall = {
        'r2': float(r2_score(y_true, y_pred)),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'n': len(y_true),
    }

    per_site = {}
    for site in sorted(results_df['site'].unique()):
        mask = results_df['site'] == site
        yt = results_df.loc[mask, 'actual_da'].values
        yp = results_df.loc[mask, 'predicted_da'].values
        per_site[site] = {
            'r2': float(r2_score(yt, yp)),
            'mae': float(mean_absolute_error(yt, yp)),
            'n': int(mask.sum()),
        }

    print(f"  Overall: R²={overall['r2']:.4f}, MAE={overall['mae']:.2f}, N={overall['n']}")
    for site, m in per_site.items():
        print(f"    {site:<18} R²={m['r2']:.4f}, MAE={m['mae']:.2f}, N={m['n']}")

    return {'overall': overall, 'per_site': per_site}


def main():
    # ── Baseline ───────────────────────────────────────────────────────────
    baseline = run_ablation("Baseline (full DATect)")

    # ── Ablation 1: No interpolated training ───────────────────────────────
    abl_no_interp = run_ablation(
        "No interpolated training",
        env_overrides={"DATECT_USE_INTERPOLATED_TRAINING": "false"},
    )

    # ── Ablation 2: No per-site customization ──────────────────────────────
    abl_no_persite = run_ablation(
        "No per-site customization",
        env_overrides={"DATECT_USE_PER_SITE_MODELS": "false"},
    )

    # ── Ablation 3: No observation-order lags ──────────────────────────────
    abl_no_lags = run_ablation(
        "No observation-order lags",
        env_overrides={"DATECT_LAG_FEATURES": "none"},
    )

    # ── Ablation 4: No derived features ────────────────────────────────────
    abl_no_derived = run_ablation(
        "No derived features",
        env_overrides={"DATECT_EXTRA_DROP_FEATURES": ",".join(DERIVED_FEATURES)},
    )

    # ── Ablation 5: No naive in ensemble ───────────────────────────────────
    abl_no_naive = run_ablation(
        "No naive in ensemble",
        naive_zero=True,
    )

    # ── Compile results ───────────────────────────────────────────────────
    all_results = {
        'baseline': baseline,
        'no_interpolated_training': abl_no_interp,
        'no_per_site_customization': abl_no_persite,
        'no_observation_order_lags': abl_no_lags,
        'no_derived_features': abl_no_derived,
        'no_naive_in_ensemble': abl_no_naive,
    }

    with open('paper_ablation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)
    print(f"{'Configuration':<35} {'R²':>8} {'ΔR²':>8} {'MAE':>8} {'N':>6}")
    print("-" * 70)

    baseline_r2 = baseline['overall']['r2'] if baseline else 0

    for name, result in all_results.items():
        if result is None:
            print(f"{name:<35} {'FAILED':>8}")
            continue
        r2 = result['overall']['r2']
        mae = result['overall']['mae']
        n = result['overall']['n']
        delta = r2 - baseline_r2
        delta_str = f"{delta:+.3f}" if name != 'baseline' else "---"
        print(f"{name:<35} {r2:>8.3f} {delta_str:>8} {mae:>8.2f} {n:>6}")

    print()
    print("Saved to paper_ablation_results.json")


if __name__ == "__main__":
    main()
