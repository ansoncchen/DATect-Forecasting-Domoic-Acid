#!/usr/bin/env python3
"""
Nested cross-validation for ensemble weight optimization.

Uses cached per-model predictions (XGBoost, RF, Naive) to find optimal
per-site ensemble weights via grid search with k-fold CV, then evaluates
on the independent test set (seed=123).

No model retraining needed — just re-blends existing predictions.

Outputs:
  - Console: comparison of manual vs CV-optimized weights per site
  - paper_nested_cv_results.json: full results for reproducibility
  - LaTeX table for paper appendix
"""

import json
import itertools
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import r2_score, mean_absolute_error


# ---------------------------------------------------------------------------
# Weight grid: all (w_xgb, w_rf, w_naive) combinations summing to 1.0
# at 0.05 resolution
# ---------------------------------------------------------------------------
def build_weight_grid(step: float = 0.05) -> list[tuple[float, float, float]]:
    """Generate all (w_xgb, w_rf, w_naive) with given step, summing to 1."""
    grid = []
    vals = np.arange(0.0, 1.0 + step / 2, step)
    for w1 in vals:
        for w2 in vals:
            w3 = 1.0 - w1 - w2
            if w3 >= -1e-9:
                grid.append((round(w1, 3), round(w2, 3), round(max(w3, 0.0), 3)))
    return grid


# ---------------------------------------------------------------------------
# Load cached predictions for a given seed
# ---------------------------------------------------------------------------
def load_predictions(cache_dir: str) -> pd.DataFrame:
    """Load XGB, RF, Naive, and Ensemble predictions and merge by (site, date)."""
    models = {
        "xgb": f"{cache_dir}/retrospective/regression_xgboost.json",
        "rf": f"{cache_dir}/retrospective/regression_rf.json",
        "naive": f"{cache_dir}/retrospective/regression_naive.json",
        "ensemble": f"{cache_dir}/retrospective/regression_ensemble.json",
    }

    frames = {}
    for name, path in models.items():
        with open(path) as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df = df[df["actual_da"].notna()].copy()
        df = df.rename(columns={
            "predicted_da": f"pred_{name}",
            "actual_da": "actual_da",
        })
        frames[name] = df[["site", "date", "actual_da", f"pred_{name}"]].copy()

    # Merge all models on (site, date)
    merged = frames["xgb"]
    for name in ["rf", "naive", "ensemble"]:
        merged = merged.merge(
            frames[name][["site", "date", f"pred_{name}"]],
            on=["site", "date"],
            how="inner",
        )
    return merged


# ---------------------------------------------------------------------------
# Find optimal weights for one site via grid search on given data
# ---------------------------------------------------------------------------
def optimize_weights_for_site(
    df_site: pd.DataFrame,
    weight_grid: list[tuple[float, float, float]],
    metric: str = "r2",
) -> tuple[tuple[float, float, float], float]:
    """Find weights that maximize R² (or minimize MAE) on df_site."""
    actual = df_site["actual_da"].values
    xgb = df_site["pred_xgb"].values
    rf = df_site["pred_rf"].values
    naive = df_site["pred_naive"].values

    best_score = -np.inf
    best_weights = (1 / 3, 1 / 3, 1 / 3)

    for w_xgb, w_rf, w_naive in weight_grid:
        pred = w_xgb * xgb + w_rf * rf + w_naive * naive
        if metric == "r2":
            score = r2_score(actual, pred)
        elif metric == "mae":
            score = -mean_absolute_error(actual, pred)  # negate so higher is better
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_weights = (w_xgb, w_rf, w_naive)

    return best_weights, best_score


# ---------------------------------------------------------------------------
# K-fold nested CV for one site
# ---------------------------------------------------------------------------
def nested_cv_site(
    df_site: pd.DataFrame,
    weight_grid: list[tuple[float, float, float]],
    k: int = 5,
    seed: int = 42,
    metric: str = "r2",
) -> dict:
    """
    K-fold nested CV for weight selection at one site.

    Inner loop: grid search on k-1 folds to find best weights.
    Outer loop: evaluate those weights on held-out fold.

    Returns dict with per-fold results and overall CV score.
    """
    n = len(df_site)
    if n < k:
        return None

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    folds = np.array_split(indices, k)

    fold_results = []
    all_cv_preds = np.full(n, np.nan)
    actual = df_site["actual_da"].values

    for fold_idx in range(k):
        test_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != fold_idx])

        # Inner loop: find best weights on training folds
        inner_best_weights, inner_best_score = optimize_weights_for_site(
            df_site.iloc[train_idx], weight_grid, metric=metric
        )

        # Outer loop: evaluate on held-out fold
        test_data = df_site.iloc[test_idx]
        test_pred = (
            inner_best_weights[0] * test_data["pred_xgb"].values
            + inner_best_weights[1] * test_data["pred_rf"].values
            + inner_best_weights[2] * test_data["pred_naive"].values
        )
        all_cv_preds[test_idx] = test_pred

        fold_r2 = r2_score(test_data["actual_da"].values, test_pred) if len(test_idx) > 1 else np.nan
        fold_mae = mean_absolute_error(test_data["actual_da"].values, test_pred)

        fold_results.append({
            "fold": fold_idx,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "inner_weights": inner_best_weights,
            "inner_score": inner_best_score,
            "outer_r2": fold_r2,
            "outer_mae": fold_mae,
        })

    # Overall CV score (predictions from all folds combined)
    valid_mask = ~np.isnan(all_cv_preds)
    cv_r2 = r2_score(actual[valid_mask], all_cv_preds[valid_mask])
    cv_mae = mean_absolute_error(actual[valid_mask], all_cv_preds[valid_mask])

    # Also get the "consensus" weights: optimize on ALL data (for comparison)
    full_best_weights, full_best_score = optimize_weights_for_site(
        df_site, weight_grid, metric=metric
    )

    # Most-selected weights across folds
    fold_weight_counts = defaultdict(int)
    for fr in fold_results:
        fold_weight_counts[fr["inner_weights"]] += 1
    majority_weights = max(fold_weight_counts, key=fold_weight_counts.get)
    majority_count = fold_weight_counts[majority_weights]

    return {
        "n_points": n,
        "k": k,
        "cv_r2": cv_r2,
        "cv_mae": cv_mae,
        "full_data_weights": full_best_weights,
        "full_data_r2": full_best_score,
        "majority_weights": majority_weights,
        "majority_count": majority_count,
        "fold_results": fold_results,
        "fold_weight_counts": dict(
            (str(k), v) for k, v in fold_weight_counts.items()
        ),
    }


# ---------------------------------------------------------------------------
# Evaluate a set of weights on a dataset
# ---------------------------------------------------------------------------
def evaluate_weights(df: pd.DataFrame, weights_by_site: dict) -> dict:
    """Evaluate per-site weights on a dataset. Returns per-site and overall metrics."""
    results = {}
    all_actual = []
    all_pred = []

    for site, site_df in df.groupby("site"):
        w = weights_by_site.get(site, (1 / 3, 1 / 3, 1 / 3))
        pred = (
            w[0] * site_df["pred_xgb"].values
            + w[1] * site_df["pred_rf"].values
            + w[2] * site_df["pred_naive"].values
        )
        actual = site_df["actual_da"].values
        r2 = r2_score(actual, pred)
        mae = mean_absolute_error(actual, pred)
        results[site] = {"r2": r2, "mae": mae, "n": len(actual)}
        all_actual.extend(actual)
        all_pred.extend(pred)

    results["Overall"] = {
        "r2": r2_score(all_actual, all_pred),
        "mae": mean_absolute_error(all_actual, all_pred),
        "n": len(all_actual),
    }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("NESTED CV ENSEMBLE WEIGHT OPTIMIZATION")
    print("=" * 80)

    # Current manual weights from per_site_models.py
    manual_weights = {
        "Copalis": (0.45, 0.00, 0.55),
        "Kalaloch": (0.00, 0.35, 0.65),
        "Twin Harbors": (0.30, 0.10, 0.60),
        "Quinault": (0.40, 0.15, 0.45),
        "Long Beach": (0.95, 0.00, 0.05),
        "Clatsop Beach": (0.95, 0.05, 0.00),
        "Coos Bay": (0.00, 1.00, 0.00),
        "Cannon Beach": (0.10, 0.75, 0.15),
        "Gold Beach": (1.00, 0.00, 0.00),
        "Newport": (1.00, 0.00, 0.00),
    }

    # Build weight grid
    weight_grid = build_weight_grid(step=0.05)
    print(f"\nWeight grid: {len(weight_grid)} combinations (step=0.05)")

    # Load development set (seed=42) and independent test set (seed=123)
    print("\nLoading cached predictions...")
    dev_df = load_predictions("cache")
    test_df = load_predictions("cache_seed123")
    print(f"  Development set (seed=42): {len(dev_df)} test points")
    print(f"  Independent test set (seed=123): {len(test_df)} test points")

    # ---------------------------------------------------------------------------
    # Run nested CV on development set
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STAGE 1: Nested 5-fold CV on development set (seed=42)")
    print("=" * 80)

    cv_results = {}
    cv_optimized_weights = {}
    sites = sorted(dev_df["site"].unique())

    for site in sites:
        site_df = dev_df[dev_df["site"] == site].reset_index(drop=True)
        result = nested_cv_site(site_df, weight_grid, k=5, seed=42)
        if result is None:
            print(f"  {site}: too few points for CV")
            cv_optimized_weights[site] = manual_weights.get(site, (1/3, 1/3, 1/3))
            continue

        cv_results[site] = result
        cv_optimized_weights[site] = result["full_data_weights"]

        mw = manual_weights.get(site, (1/3, 1/3, 1/3))
        cw = result["full_data_weights"]
        majw = result["majority_weights"]

        # Evaluate manual weights on dev set for comparison
        manual_pred = (
            mw[0] * site_df["pred_xgb"].values
            + mw[1] * site_df["pred_rf"].values
            + mw[2] * site_df["pred_naive"].values
        )
        manual_r2 = r2_score(site_df["actual_da"].values, manual_pred)

        print(f"\n  {site} (N={result['n_points']}):")
        print(f"    Manual weights:   ({mw[0]:.2f}, {mw[1]:.2f}, {mw[2]:.2f})  → dev R²={manual_r2:.3f}")
        print(f"    CV-optimal (full): ({cw[0]:.2f}, {cw[1]:.2f}, {cw[2]:.2f})  → dev R²={result['full_data_r2']:.3f}")
        print(f"    CV-optimal (majority {result['majority_count']}/{result['k']}): ({majw[0]:.2f}, {majw[1]:.2f}, {majw[2]:.2f})")
        print(f"    Nested CV R²:     {result['cv_r2']:.3f}  (unbiased estimate)")

        # Show fold stability
        fold_weights = [fr["inner_weights"] for fr in result["fold_results"]]
        unique_weights = set(fold_weights)
        if len(unique_weights) == 1:
            print(f"    Fold stability:   STABLE (all {result['k']} folds agree)")
        else:
            print(f"    Fold stability:   {len(unique_weights)} distinct weight sets across {result['k']} folds")
            for w, count in sorted(result["fold_weight_counts"].items(), key=lambda x: -x[1]):
                print(f"      {w}: {count}/{result['k']} folds")

    # ---------------------------------------------------------------------------
    # Evaluate on independent test set (seed=123)
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STAGE 2: Evaluation on independent test set (seed=123)")
    print("=" * 80)

    manual_eval = evaluate_weights(test_df, manual_weights)
    cv_eval = evaluate_weights(test_df, cv_optimized_weights)

    # Also try majority weights from CV
    majority_weights = {}
    for site in sites:
        if site in cv_results:
            majority_weights[site] = cv_results[site]["majority_weights"]
        else:
            majority_weights[site] = manual_weights.get(site, (1/3, 1/3, 1/3))
    majority_eval = evaluate_weights(test_df, majority_weights)

    # Equal weights baseline
    equal_weights = {site: (1/3, 1/3, 1/3) for site in sites}
    equal_eval = evaluate_weights(test_df, equal_weights)

    print(f"\n{'Site':<15} {'N':>5}  {'Manual R²':>10} {'CV-opt R²':>10} {'Majority R²':>10} {'Equal R²':>10}  {'Manual':>20} {'CV-optimal':>20}")
    print("-" * 120)
    for site in sites:
        n = manual_eval[site]["n"]
        mr2 = manual_eval[site]["r2"]
        cr2 = cv_eval[site]["r2"]
        majr2 = majority_eval[site]["r2"]
        er2 = equal_eval[site]["r2"]
        mw = manual_weights.get(site, (1/3, 1/3, 1/3))
        cw = cv_optimized_weights.get(site, (1/3, 1/3, 1/3))
        delta = cr2 - mr2
        marker = "  <<<" if abs(delta) > 0.02 else ""
        print(f"{site:<15} {n:>5}  {mr2:>10.3f} {cr2:>10.3f} {majr2:>10.3f} {er2:>10.3f}  ({mw[0]:.2f},{mw[1]:.2f},{mw[2]:.2f})  ({cw[0]:.2f},{cw[1]:.2f},{cw[2]:.2f}){marker}")

    print("-" * 120)
    n = manual_eval["Overall"]["n"]
    print(f"{'Overall':<15} {n:>5}  {manual_eval['Overall']['r2']:>10.3f} {cv_eval['Overall']['r2']:>10.3f} {majority_eval['Overall']['r2']:>10.3f} {equal_eval['Overall']['r2']:>10.3f}")

    print(f"\n{'':>22} {'Manual MAE':>10} {'CV-opt MAE':>10} {'Majority MAE':>10} {'Equal MAE':>10}")
    print("-" * 80)
    for site in sites:
        print(f"{site:<15} {'':>5}  {manual_eval[site]['mae']:>10.2f} {cv_eval[site]['mae']:>10.2f} {majority_eval[site]['mae']:>10.2f} {equal_eval[site]['mae']:>10.2f}")
    print("-" * 80)
    print(f"{'Overall':<15} {'':>5}  {manual_eval['Overall']['mae']:>10.2f} {cv_eval['Overall']['mae']:>10.2f} {majority_eval['Overall']['mae']:>10.2f} {equal_eval['Overall']['mae']:>10.2f}")

    # ---------------------------------------------------------------------------
    # Generate LaTeX table
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("LaTeX table: Manual vs CV-optimized weights")
    print("=" * 80)
    print()
    print(r"\begin{table}[H]")
    print(r"\caption{Comparison of manually tuned and nested-CV-optimized ensemble weights. CV-optimal weights determined by 5-fold cross-validation on the development set (seed = 42, $n = 1{,}177$), then evaluated on the independent test set (seed = 123, $n = 2{,}181$). $\Delta R^2$ is CV-optimal minus manual.}")
    print(r"\label{tab:cv-weights}")
    print(r"\small")
    print(r"\begin{tabular}{l ccc ccc rr r}")
    print(r"\toprule")
    print(r" & \multicolumn{3}{c}{\textbf{Manual weights}} & \multicolumn{3}{c}{\textbf{CV-optimal weights}} & & & \\")
    print(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    print(r"\textbf{Site} & $w_{\text{XGB}}$ & $w_{\text{RF}}$ & $w_{\text{N}}$ & $w_{\text{XGB}}$ & $w_{\text{RF}}$ & $w_{\text{N}}$ & \textbf{Manual $R^2$} & \textbf{CV $R^2$} & $\boldsymbol{\Delta R^2}$ \\")
    print(r"\midrule")

    for site in sites:
        mw = manual_weights.get(site, (1/3, 1/3, 1/3))
        cw = cv_optimized_weights.get(site, (1/3, 1/3, 1/3))
        mr2 = manual_eval[site]["r2"]
        cr2 = cv_eval[site]["r2"]
        delta = cr2 - mr2
        sign = "+" if delta >= 0 else ""
        print(f"{site:<15} & {mw[0]:.2f} & {mw[1]:.2f} & {mw[2]:.2f} & {cw[0]:.2f} & {cw[1]:.2f} & {cw[2]:.2f} & {mr2:.3f} & {cr2:.3f} & {sign}{delta:.3f} \\\\")

    print(r"\midrule")
    mr2 = manual_eval["Overall"]["r2"]
    cr2 = cv_eval["Overall"]["r2"]
    delta = cr2 - mr2
    sign = "+" if delta >= 0 else ""
    print(f"{'Overall':<15} & --- & --- & --- & --- & --- & --- & {mr2:.3f} & {cr2:.3f} & {sign}{delta:.3f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # ---------------------------------------------------------------------------
    # Save full results
    # ---------------------------------------------------------------------------
    output = {
        "description": "Nested CV ensemble weight optimization results",
        "weight_grid_step": 0.05,
        "weight_grid_size": len(weight_grid),
        "dev_set": {"seed": 42, "n": len(dev_df)},
        "test_set": {"seed": 123, "n": len(test_df)},
        "manual_weights": {k: list(v) for k, v in manual_weights.items()},
        "cv_optimized_weights": {k: list(v) for k, v in cv_optimized_weights.items()},
        "majority_weights": {k: list(v) for k, v in majority_weights.items()},
        "test_set_evaluation": {
            "manual": {k: v for k, v in manual_eval.items()},
            "cv_optimized": {k: v for k, v in cv_eval.items()},
            "majority": {k: v for k, v in majority_eval.items()},
            "equal": {k: v for k, v in equal_eval.items()},
        },
        "per_site_cv_results": {},
    }

    for site, result in cv_results.items():
        output["per_site_cv_results"][site] = {
            "n_points": result["n_points"],
            "k": result["k"],
            "cv_r2": result["cv_r2"],
            "cv_mae": result["cv_mae"],
            "full_data_weights": list(result["full_data_weights"]),
            "full_data_r2": result["full_data_r2"],
            "majority_weights": list(result["majority_weights"]),
            "majority_count": result["majority_count"],
            "fold_weight_counts": result["fold_weight_counts"],
            "fold_results": [
                {
                    "fold": fr["fold"],
                    "n_train": fr["n_train"],
                    "n_test": fr["n_test"],
                    "inner_weights": list(fr["inner_weights"]),
                    "outer_r2": fr["outer_r2"] if not np.isnan(fr["outer_r2"]) else None,
                    "outer_mae": fr["outer_mae"],
                }
                for fr in result["fold_results"]
            ],
        }

    with open("paper_nested_cv_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to paper_nested_cv_results.json")


if __name__ == "__main__":
    main()
