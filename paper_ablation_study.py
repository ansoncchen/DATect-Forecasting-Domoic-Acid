#!/usr/bin/env python3
"""
DATect Ablation Study for Paper

Runs 5 ablation experiments, each disabling one key component:
  1. No interpolated training (USE_INTERPOLATED_TRAINING=False)
  2. No per-site customization (USE_PER_SITE_MODELS=False)
  3. No observation-order lags (LAG_FEATURES=[])
  4. No derived features (MHW, BEUTI², PDO-ONI phase, PN tipping, fluor, K490²)
  5. No naive in ensemble (naive weight → 0, renormalize XGB+RF)

FIX: Uses threading backend instead of loky (multiprocessing) so config
mutations are visible to parallel workers. Threads share process memory;
XGBoost/sklearn release the GIL during C-level computation, so threading
is both correct AND fast.

Usage (run on Hyak):
    python3 paper_ablation_study.py

Output: paper_ablation_results.json
"""

import json
import sys
import warnings
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')


def run_ablation(name, setup_fn, teardown_fn=None):
    """Run a single ablation experiment."""
    import config
    import backend.api as api_module

    print(f"\n{'='*60}")
    print(f"ABLATION: {name}")
    print(f"{'='*60}")

    # ── Use threading so config mutations are visible to workers ──
    # loky (default) spawns independent processes that import config fresh.
    # Threading shares the same process memory → mutations propagate.
    config.PARALLEL_BACKEND = "threading"

    # Apply config changes
    setup_fn(config)

    # Verify config state
    print(f"  Config: USE_INTERPOLATED_TRAINING={config.USE_INTERPOLATED_TRAINING}")
    print(f"  Config: USE_PER_SITE_MODELS={config.USE_PER_SITE_MODELS}")
    print(f"  Config: LAG_FEATURES={config.LAG_FEATURES}")
    print(f"  Config: ZERO_IMPORTANCE len={len(config.ZERO_IMPORTANCE_FEATURES)}")
    print(f"  Config: PARALLEL_BACKEND={config.PARALLEL_BACKEND}")

    # Force a completely fresh engine
    api_module.forecast_engine = None

    from forecasting.raw_forecast_engine import RawForecastEngine
    engine = RawForecastEngine(validate_on_init=False)

    n_anchors = getattr(config, 'N_RANDOM_ANCHORS', 500)

    results_df = engine.run_retrospective_evaluation(
        task="regression",
        model_type="ensemble",
        n_anchors=n_anchors,
        min_test_date="2008-01-01"
    )

    # Restore config
    config.PARALLEL_BACKEND = "loky"
    if teardown_fn:
        teardown_fn(config)

    if results_df is None or results_df.empty:
        print(f"  ERROR: No results for {name}")
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


# ── Derived features to disable ──────────────────────────────────────────────
DERIVED_FEATURES = [
    'mhw_flag', 'beuti_squared', 'beuti_relaxation',
    'pdo_oni_phase', 'fluor_efficiency', 'k490_squared',
    'pn_log', 'pn_above_threshold',
]


def main():
    import config

    # Save originals for restoration
    orig_interpolated = config.USE_INTERPOLATED_TRAINING
    orig_per_site = config.USE_PER_SITE_MODELS
    orig_lag_features = list(config.LAG_FEATURES)
    orig_zero_imp = list(config.ZERO_IMPORTANCE_FEATURES)

    # ── Baseline (full DATect, for comparison) ────────────────────────────────
    baseline = run_ablation(
        "Baseline (full DATect)",
        setup_fn=lambda c: None,
    )

    # ── Ablation 1: No interpolated training ──────────────────────────────────
    def setup_no_interp(c):
        c.USE_INTERPOLATED_TRAINING = False
    def teardown_no_interp(c):
        c.USE_INTERPOLATED_TRAINING = orig_interpolated

    abl_no_interp = run_ablation(
        "No interpolated training",
        setup_fn=setup_no_interp,
        teardown_fn=teardown_no_interp,
    )

    # ── Ablation 2: No per-site customization ─────────────────────────────────
    def setup_no_persite(c):
        c.USE_PER_SITE_MODELS = False
    def teardown_no_persite(c):
        c.USE_PER_SITE_MODELS = orig_per_site

    abl_no_persite = run_ablation(
        "No per-site customization",
        setup_fn=setup_no_persite,
        teardown_fn=teardown_no_persite,
    )

    # ── Ablation 3: No observation-order lags ─────────────────────────────────
    # FIX: USE_LAG_FEATURES (boolean) is never read by the engine.
    # Must clear config.LAG_FEATURES (the int list) AND patch the
    # RawForecastConfig dataclass default (frozen at import time).
    def setup_no_lags(c):
        c.LAG_FEATURES = []
        from forecasting.raw_data_forecaster import RawForecastConfig
        RawForecastConfig.__dataclass_fields__['lags'].default = ()

    def teardown_no_lags(c):
        c.LAG_FEATURES = list(orig_lag_features)
        from forecasting.raw_data_forecaster import RawForecastConfig
        RawForecastConfig.__dataclass_fields__['lags'].default = tuple(orig_lag_features)

    abl_no_lags = run_ablation(
        "No observation-order lags",
        setup_fn=setup_no_lags,
        teardown_fn=teardown_no_lags,
    )

    # ── Ablation 4: No derived features ───────────────────────────────────────
    def setup_no_derived(c):
        c.ZERO_IMPORTANCE_FEATURES = list(orig_zero_imp) + DERIVED_FEATURES
    def teardown_no_derived(c):
        c.ZERO_IMPORTANCE_FEATURES = list(orig_zero_imp)

    abl_no_derived = run_ablation(
        "No derived features",
        setup_fn=setup_no_derived,
        teardown_fn=teardown_no_derived,
    )

    # ── Ablation 5: No naive in ensemble ──────────────────────────────────────
    def setup_no_naive(c):
        from forecasting import per_site_models as psm
        c._orig_site_configs = deepcopy(psm.SITE_SPECIFIC_CONFIGS)
        for site_name, site_cfg in psm.SITE_SPECIFIC_CONFIGS.items():
            w_xgb, w_rf, w_naive = site_cfg.get('ensemble_weights', (0.40, 0.40, 0.20))
            total = w_xgb + w_rf
            if total > 0:
                site_cfg['ensemble_weights'] = (w_xgb / total, w_rf / total, 0.0)
            else:
                site_cfg['ensemble_weights'] = (0.5, 0.5, 0.0)
        c._orig_default_weights = psm.DEFAULT_SITE_CONFIG.get('ensemble_weights')
        psm.DEFAULT_SITE_CONFIG['ensemble_weights'] = (0.5625, 0.4375, 0.0)

    def teardown_no_naive(c):
        from forecasting import per_site_models as psm
        psm.SITE_SPECIFIC_CONFIGS = c._orig_site_configs
        if c._orig_default_weights is not None:
            psm.DEFAULT_SITE_CONFIG['ensemble_weights'] = c._orig_default_weights
        else:
            psm.DEFAULT_SITE_CONFIG.pop('ensemble_weights', None)

    abl_no_naive = run_ablation(
        "No naive in ensemble",
        setup_fn=setup_no_naive,
        teardown_fn=teardown_no_naive,
    )

    # ── Compile results ───────────────────────────────────────────────────────
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

    # ── Summary table ─────────────────────────────────────────────────────────
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
