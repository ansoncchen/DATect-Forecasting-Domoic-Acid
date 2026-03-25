#!/usr/bin/env python3
"""
Paired bootstrap test for model comparison.

Tests whether the ensemble significantly outperforms each competitor model
by bootstrapping the paired difference in R² (ΔR²).

For each bootstrap replicate, both models are evaluated on the same resampled
data, and ΔR² = R²(ensemble) − R²(competitor) is computed. The 95% CI and
empirical p-value (proportion of replicates where competitor wins) are reported.

Usage:
    python3 paper_paired_bootstrap.py

Output:
    paper_paired_bootstrap_results.json
    Console output with formatted results for paper text
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score

# ── Configuration ────────────────────────────────────────────────────────────
CACHE_DIR = Path("./cache_seed123/retrospective")
N_BOOTSTRAP = 10000
SEED = 42
OUTPUT_JSON = "paper_paired_bootstrap_results.json"

COMPETITORS = {
    "xgboost": "XGBoost",
    "rf": "Random Forest",
    "linear": "Ridge Regression",
    "naive": "Naïve Persistence",
}


def load_predictions(model_type: str) -> dict[str, list]:
    """Load predictions keyed by (date, site) for alignment."""
    json_path = CACHE_DIR / f"regression_{model_type}.json"
    with open(json_path) as f:
        data = json.load(f)
    return {
        (r["date"], r["site"]): r
        for r in data
        if r["actual_da"] is not None and r["predicted_da"] is not None
    }


def paired_bootstrap_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> dict:
    """
    Paired bootstrap test: H0: R²(A) = R²(B), H1: R²(A) > R²(B).

    Returns point estimate, 95% CI, and one-sided p-value for ΔR² = R²(A) − R²(B).
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    # Point estimates
    r2_a = r2_score(y_true, y_pred_a)
    r2_b = r2_score(y_true, y_pred_b)
    delta_r2 = r2_a - r2_b

    # Bootstrap
    deltas = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        if np.std(yt) == 0:
            continue
        r2_a_boot = r2_score(yt, y_pred_a[idx])
        r2_b_boot = r2_score(yt, y_pred_b[idx])
        deltas.append(r2_a_boot - r2_b_boot)

    deltas = np.array(deltas)

    # One-sided p-value: proportion of times B beats A
    p_value = np.mean(deltas <= 0)

    return {
        "r2_a": float(r2_a),
        "r2_b": float(r2_b),
        "delta_r2": float(delta_r2),
        "delta_r2_ci_lo": float(np.percentile(deltas, 2.5)),
        "delta_r2_ci_hi": float(np.percentile(deltas, 97.5)),
        "p_value": float(p_value),
        "n_boot": len(deltas),
        "n_samples": n,
    }


def main():
    print("=" * 70)
    print("Paired Bootstrap Test: Ensemble vs Competitors")
    print(f"B = {N_BOOTSTRAP:,} | Cache: {CACHE_DIR}")
    print("=" * 70)

    # Load ensemble predictions
    ens_data = load_predictions("ensemble")
    print(f"Ensemble: {len(ens_data)} predictions")

    results = {}

    for comp_key, comp_name in COMPETITORS.items():
        comp_data = load_predictions(comp_key)

        # Align predictions by (date, site)
        common_keys = sorted(set(ens_data.keys()) & set(comp_data.keys()))
        y_true = np.array([ens_data[k]["actual_da"] for k in common_keys])
        y_ens = np.array([ens_data[k]["predicted_da"] for k in common_keys])
        y_comp = np.array([comp_data[k]["predicted_da"] for k in common_keys])

        test = paired_bootstrap_test(y_true, y_ens, y_comp)
        results[comp_key] = {"display_name": comp_name, **test}

        sig = "***" if test["p_value"] < 0.001 else "**" if test["p_value"] < 0.01 else "*" if test["p_value"] < 0.05 else "n.s."
        print(f"\n  Ensemble vs {comp_name}:")
        print(f"    ΔR² = {test['delta_r2']:.4f} [{test['delta_r2_ci_lo']:.4f}, {test['delta_r2_ci_hi']:.4f}]")
        print(f"    p = {test['p_value']:.4f} ({sig})")
        print(f"    N = {test['n_samples']}")

    # Save results
    with open(OUTPUT_JSON, "w") as f:
        json.dump({"n_bootstrap": N_BOOTSTRAP, "seed": SEED, "comparisons": results}, f, indent=2)
    print(f"\nSaved: {OUTPUT_JSON}")

    # Summary for paper
    print("\n" + "=" * 70)
    print("PAPER TEXT (copy-paste):")
    print("=" * 70)
    xgb = results["xgboost"]
    print(f"Paired bootstrap test (B = {N_BOOTSTRAP:,}): the ensemble's advantage over")
    print(f"XGBoost (ΔR² = {xgb['delta_r2']:.3f} [{xgb['delta_r2_ci_lo']:.3f}, {xgb['delta_r2_ci_hi']:.3f}],")
    print(f"p = {xgb['p_value']:.3f}) ", end="")
    if xgb["p_value"] < 0.05:
        print("is statistically significant at α = 0.05.")
    else:
        print("is not statistically significant at α = 0.05,")
        print("reflecting the modest margin between the ensemble and its strongest component.")


if __name__ == "__main__":
    main()
