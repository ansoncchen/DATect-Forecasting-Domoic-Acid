#!/usr/bin/env python3
"""
Comprehensive analysis: Why do manual weights outperform CV-optimized weights?

Investigates:
1. Oracle weights (best possible on test set) — what's the ceiling?
2. Distribution shift between dev (seed=42) and test (seed=123) sets
3. Temporal CV (respecting time order) vs random CV
4. Leave-one-year-out CV
5. Per-site component model quality (is the problem the models, not weights?)
6. Weight sensitivity analysis (how flat is the R² surface?)
7. Stacking (train a meta-learner instead of fixed weights)
8. Bootstrap stability of optimal weights
9. Optimizing on MAE instead of R²
10. What if we use seed=123 overlapping points only?
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import RidgeCV
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def load_predictions(cache_dir: str) -> pd.DataFrame:
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
        df = df.rename(columns={"predicted_da": f"pred_{name}"})
        frames[name] = df[["site", "date", "actual_da", f"pred_{name}"]].copy()

    merged = frames["xgb"]
    for name in ["rf", "naive", "ensemble"]:
        merged = merged.merge(
            frames[name][["site", "date", f"pred_{name}"]],
            on=["site", "date"], how="inner",
        )
    merged["date"] = pd.to_datetime(merged["date"])
    return merged


def build_weight_grid(step=0.05):
    grid = []
    vals = np.arange(0.0, 1.0 + step / 2, step)
    for w1 in vals:
        for w2 in vals:
            w3 = 1.0 - w1 - w2
            if w3 >= -1e-9:
                grid.append((round(w1, 3), round(w2, 3), round(max(w3, 0.0), 3)))
    return grid


def blend(df, w):
    return w[0] * df["pred_xgb"].values + w[1] * df["pred_rf"].values + w[2] * df["pred_naive"].values


def find_best_weights(df, grid, metric="r2"):
    actual = df["actual_da"].values
    best_score, best_w = -np.inf, (1/3, 1/3, 1/3)
    for w in grid:
        pred = blend(df, w)
        s = r2_score(actual, pred) if metric == "r2" else -mean_absolute_error(actual, pred)
        if s > best_score:
            best_score, best_w = s, w
    return best_w, best_score


MANUAL_WEIGHTS = {
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


def eval_weights(df, weights_by_site):
    all_a, all_p = [], []
    per_site = {}
    for site, sdf in df.groupby("site"):
        w = weights_by_site.get(site, (1/3, 1/3, 1/3))
        p = blend(sdf, w)
        a = sdf["actual_da"].values
        per_site[site] = {"r2": r2_score(a, p), "mae": mean_absolute_error(a, p), "n": len(a)}
        all_a.extend(a); all_p.extend(p)
    per_site["Overall"] = {"r2": r2_score(all_a, all_p), "mae": mean_absolute_error(all_a, all_p), "n": len(all_a)}
    return per_site


# ===========================================================================
def main():
    print("=" * 90)
    print("COMPREHENSIVE WEIGHT ANALYSIS")
    print("=" * 90)

    grid = build_weight_grid(0.05)
    fine_grid = build_weight_grid(0.01)
    dev = load_predictions("cache")
    test = load_predictions("cache_seed123")
    sites = sorted(dev["site"].unique())

    # -----------------------------------------------------------------------
    # 1. ORACLE WEIGHTS — best possible on test set (upper bound)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("1. ORACLE WEIGHTS (optimized directly on seed=123 test set)")
    print("   This is the CEILING — impossible to achieve without cheating.")
    print("=" * 90)

    oracle_weights = {}
    for site in sites:
        sdf = test[test["site"] == site]
        w, score = find_best_weights(sdf, fine_grid)
        oracle_weights[site] = w

    oracle_eval = eval_weights(test, oracle_weights)
    manual_eval = eval_weights(test, MANUAL_WEIGHTS)

    print(f"\n{'Site':<15} {'N':>5}  {'Manual R²':>10} {'Oracle R²':>10} {'Gap':>8}  {'Manual wts':>20} {'Oracle wts':>20}")
    print("-" * 100)
    for site in sites:
        mw = MANUAL_WEIGHTS[site]
        ow = oracle_weights[site]
        mr = manual_eval[site]["r2"]
        orc = oracle_eval[site]["r2"]
        print(f"{site:<15} {manual_eval[site]['n']:>5}  {mr:>10.3f} {orc:>10.3f} {orc-mr:>8.3f}  ({mw[0]:.2f},{mw[1]:.2f},{mw[2]:.2f})  ({ow[0]:.2f},{ow[1]:.2f},{ow[2]:.2f})")
    print("-" * 100)
    print(f"{'Overall':<15} {manual_eval['Overall']['n']:>5}  {manual_eval['Overall']['r2']:>10.3f} {oracle_eval['Overall']['r2']:>10.3f} {oracle_eval['Overall']['r2']-manual_eval['Overall']['r2']:>8.3f}")

    # -----------------------------------------------------------------------
    # 2. DISTRIBUTION SHIFT ANALYSIS
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("2. DISTRIBUTION SHIFT: Dev (seed=42) vs Test (seed=123)")
    print("=" * 90)

    for site in sites:
        dev_s = dev[dev["site"] == site]
        test_s = test[test["site"] == site]
        overlap = pd.merge(dev_s, test_s, on=["site", "date"], suffixes=("_dev", "_test"))

        print(f"\n  {site}: dev N={len(dev_s)}, test N={len(test_s)}, overlap N={len(overlap)}")
        print(f"    Dev  actual DA: mean={dev_s['actual_da'].mean():.1f}, std={dev_s['actual_da'].std():.1f}, "
              f"median={dev_s['actual_da'].median():.1f}, max={dev_s['actual_da'].max():.0f}")
        print(f"    Test actual DA: mean={test_s['actual_da'].mean():.1f}, std={test_s['actual_da'].std():.1f}, "
              f"median={test_s['actual_da'].median():.1f}, max={test_s['actual_da'].max():.0f}")

        # Fraction of high-DA points
        dev_high = (dev_s["actual_da"] > 20).mean()
        test_high = (test_s["actual_da"] > 20).mean()
        print(f"    Fraction DA>20: dev={dev_high:.3f}, test={test_high:.3f}")

        # Per-model R² on dev vs test
        for model in ["xgb", "rf", "naive"]:
            dev_r2 = r2_score(dev_s["actual_da"], dev_s[f"pred_{model}"])
            test_r2 = r2_score(test_s["actual_da"], test_s[f"pred_{model}"])
            print(f"    {model:>5} R²: dev={dev_r2:.3f}, test={test_r2:.3f}, shift={test_r2-dev_r2:+.3f}")

    # -----------------------------------------------------------------------
    # 3. WEIGHT SENSITIVITY / R² SURFACE FLATNESS
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("3. WEIGHT SENSITIVITY: How flat is the R² surface?")
    print("   Shows R² range for all weights within 0.01 of the optimum.")
    print("=" * 90)

    for site in sites:
        sdf_dev = dev[dev["site"] == site]
        sdf_test = test[test["site"] == site]

        # Compute R² for all weights on both sets
        dev_scores = []
        test_scores = []
        for w in grid:
            dev_scores.append((w, r2_score(sdf_dev["actual_da"], blend(sdf_dev, w))))
            test_scores.append((w, r2_score(sdf_test["actual_da"], blend(sdf_test, w))))

        dev_scores.sort(key=lambda x: -x[1])
        test_scores.sort(key=lambda x: -x[1])

        best_dev_r2 = dev_scores[0][1]
        best_test_r2 = test_scores[0][1]

        # How many weight combos are within 0.01 R² of optimum?
        near_opt_dev = [s for s in dev_scores if s[1] >= best_dev_r2 - 0.01]
        near_opt_test = [s for s in test_scores if s[1] >= best_test_r2 - 0.01]

        # What's the rank of dev-optimal weights on the test set?
        dev_best_w = dev_scores[0][0]
        test_rank = next(i for i, (w, _) in enumerate(test_scores) if w == dev_best_w) + 1

        # What's the R² of dev-optimal weights on test set?
        dev_best_on_test = r2_score(sdf_test["actual_da"], blend(sdf_test, dev_best_w))

        print(f"\n  {site}:")
        print(f"    Dev-optimal: {dev_scores[0][0]} → dev R²={best_dev_r2:.3f}")
        print(f"    Test-optimal: {test_scores[0][0]} → test R²={best_test_r2:.3f}")
        print(f"    Dev-optimal weights on test: R²={dev_best_on_test:.3f} (rank {test_rank}/{len(grid)})")
        print(f"    Near-optimal (within 0.01 R²): dev={len(near_opt_dev)}/{len(grid)}, test={len(near_opt_test)}/{len(grid)}")

        # Are the near-optimal regions overlapping?
        near_dev_set = set(s[0] for s in near_opt_dev)
        near_test_set = set(s[0] for s in near_opt_test)
        overlap = near_dev_set & near_test_set
        print(f"    Overlap of near-optimal regions: {len(overlap)} weight combos")
        mw = MANUAL_WEIGHTS[site]
        manual_in_near_dev = mw in near_dev_set
        manual_in_near_test = mw in near_test_set
        print(f"    Manual weights in near-optimal: dev={manual_in_near_dev}, test={manual_in_near_test}")

    # -----------------------------------------------------------------------
    # 4. TEMPORAL CV (respecting time order)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("4. TEMPORAL CV: Train on earlier years, validate on later years")
    print("   Split dev set at median date per site.")
    print("=" * 90)

    temporal_cv_weights = {}
    for site in sites:
        sdf = dev[dev["site"] == site].sort_values("date")
        n = len(sdf)
        mid = n // 2
        train_half = sdf.iloc[:mid]
        val_half = sdf.iloc[mid:]

        # Find best weights on first half, evaluate on second half
        w_early, score_early = find_best_weights(train_half, grid)
        val_pred = blend(val_half, w_early)
        val_r2 = r2_score(val_half["actual_da"], val_pred)

        # Also try: find best on second half (more recent, might match test better)
        w_late, score_late = find_best_weights(val_half, grid)

        temporal_cv_weights[site] = w_early  # Use early-half-optimized

        mw = MANUAL_WEIGHTS[site]
        manual_val_r2 = r2_score(val_half["actual_da"], blend(val_half, mw))

        print(f"\n  {site} (N={n}, split at {sdf.iloc[mid]['date'].strftime('%Y-%m-%d')}):")
        print(f"    Early-half optimal: {w_early} → early R²={score_early:.3f}, late-half R²={val_r2:.3f}")
        print(f"    Late-half optimal:  {w_late} → late R²={score_late:.3f}")
        print(f"    Manual weights on late half: R²={manual_val_r2:.3f}")

    temporal_eval = eval_weights(test, temporal_cv_weights)
    print(f"\n  Temporal CV weights on test set (seed=123):")
    print(f"    Overall R²={temporal_eval['Overall']['r2']:.3f} (manual={manual_eval['Overall']['r2']:.3f})")

    # -----------------------------------------------------------------------
    # 5. LEAVE-ONE-YEAR-OUT CV on dev set
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("5. LEAVE-ONE-YEAR-OUT CV on dev set")
    print("=" * 90)

    loyo_weights = {}
    for site in sites:
        sdf = dev[dev["site"] == site].copy()
        sdf["year"] = sdf["date"].dt.year
        years = sorted(sdf["year"].unique())

        if len(years) < 3:
            print(f"  {site}: only {len(years)} years, skipping")
            loyo_weights[site] = MANUAL_WEIGHTS[site]
            continue

        fold_weights = []
        all_preds = []
        all_actuals = []

        for yr in years:
            train = sdf[sdf["year"] != yr]
            val = sdf[sdf["year"] == yr]
            if len(train) < 5 or len(val) < 2:
                continue
            w, _ = find_best_weights(train, grid)
            fold_weights.append(w)
            pred = blend(val, w)
            all_preds.extend(pred)
            all_actuals.extend(val["actual_da"].values)

        if all_preds:
            loyo_r2 = r2_score(all_actuals, all_preds)
            # Average weights across folds
            avg_w = tuple(round(np.mean([fw[i] for fw in fold_weights]), 2) for i in range(3))
            # Renormalize
            total = sum(avg_w)
            if total > 0:
                avg_w = tuple(round(w/total, 2) for w in avg_w)
            loyo_weights[site] = avg_w

            # Weight stability
            w_std = [np.std([fw[i] for fw in fold_weights]) for i in range(3)]
            print(f"  {site}: LOYO R²={loyo_r2:.3f}, avg weights=({avg_w[0]:.2f},{avg_w[1]:.2f},{avg_w[2]:.2f}), "
                  f"weight std=({w_std[0]:.2f},{w_std[1]:.2f},{w_std[2]:.2f}), {len(years)} years")
        else:
            loyo_weights[site] = MANUAL_WEIGHTS[site]

    loyo_eval = eval_weights(test, loyo_weights)
    print(f"\n  LOYO weights on test set: Overall R²={loyo_eval['Overall']['r2']:.3f} (manual={manual_eval['Overall']['r2']:.3f})")

    # -----------------------------------------------------------------------
    # 6. STACKING (Ridge meta-learner)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("6. STACKING: Ridge regression meta-learner (trained on dev, eval on test)")
    print("=" * 90)

    stacking_weights = {}
    for site in sites:
        dev_s = dev[dev["site"] == site]
        test_s = test[test["site"] == site]

        X_dev = dev_s[["pred_xgb", "pred_rf", "pred_naive"]].values
        y_dev = dev_s["actual_da"].values

        # Ridge with non-negative constraint approximation
        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], fit_intercept=False)
        ridge.fit(X_dev, y_dev)
        coefs = ridge.coef_

        # Normalize to sum to 1 (only positive coefs)
        coefs = np.maximum(coefs, 0)
        total = coefs.sum()
        if total > 0:
            coefs = coefs / total
        else:
            coefs = np.array([1/3, 1/3, 1/3])

        stacking_weights[site] = tuple(round(c, 3) for c in coefs)

        X_test = test_s[["pred_xgb", "pred_rf", "pred_naive"]].values
        y_test = test_s["actual_da"].values

        stack_pred = ridge.predict(X_test)
        stack_r2 = r2_score(y_test, stack_pred)
        norm_pred = blend(test_s, stacking_weights[site])
        norm_r2 = r2_score(y_test, norm_pred)

        mw = MANUAL_WEIGHTS[site]
        manual_r2 = r2_score(y_test, blend(test_s, mw))

        print(f"  {site}: Ridge coefs=({coefs[0]:.2f},{coefs[1]:.2f},{coefs[2]:.2f}), "
              f"alpha={ridge.alpha_}, ridge R²={stack_r2:.3f}, normalized R²={norm_r2:.3f}, manual R²={manual_r2:.3f}")

    stacking_eval = eval_weights(test, stacking_weights)
    print(f"\n  Stacking weights on test: Overall R²={stacking_eval['Overall']['r2']:.3f} (manual={manual_eval['Overall']['r2']:.3f})")

    # -----------------------------------------------------------------------
    # 7. BOOTSTRAP STABILITY of optimal weights
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("7. BOOTSTRAP STABILITY: How stable are 'optimal' weights?")
    print("   1000 bootstrap samples of dev set → optimal weights each time")
    print("=" * 90)

    rng = np.random.RandomState(42)
    n_boot = 1000

    for site in sites:
        sdf = dev[dev["site"] == site]
        n = len(sdf)
        boot_weights = []

        for _ in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            boot_df = sdf.iloc[idx]
            w, _ = find_best_weights(boot_df, grid)
            boot_weights.append(w)

        w_xgb = [bw[0] for bw in boot_weights]
        w_rf = [bw[1] for bw in boot_weights]
        w_naive = [bw[2] for bw in boot_weights]

        # How often does bootstrap agree with manual?
        mw = MANUAL_WEIGHTS[site]
        exact_match = sum(1 for bw in boot_weights if bw == mw) / n_boot

        print(f"  {site} (N={n}):")
        print(f"    w_XGB:   mean={np.mean(w_xgb):.2f} ± {np.std(w_xgb):.2f}  [95% CI: {np.percentile(w_xgb,2.5):.2f}, {np.percentile(w_xgb,97.5):.2f}]  manual={mw[0]:.2f}")
        print(f"    w_RF:    mean={np.mean(w_rf):.2f} ± {np.std(w_rf):.2f}  [95% CI: {np.percentile(w_rf,2.5):.2f}, {np.percentile(w_rf,97.5):.2f}]  manual={mw[1]:.2f}")
        print(f"    w_Naive: mean={np.mean(w_naive):.2f} ± {np.std(w_naive):.2f}  [95% CI: {np.percentile(w_naive,2.5):.2f}, {np.percentile(w_naive,97.5):.2f}]  manual={mw[2]:.2f}")
        print(f"    Manual weights exact match: {exact_match*100:.1f}% of bootstraps")

        # Most common bootstrap weight
        from collections import Counter
        top3 = Counter(boot_weights).most_common(3)
        print(f"    Top bootstrap weights: {', '.join(f'{w}({c/n_boot*100:.1f}%)' for w,c in top3)}")

    # -----------------------------------------------------------------------
    # 8. MAE-OPTIMIZED WEIGHTS
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("8. MAE-OPTIMIZED WEIGHTS (instead of R²)")
    print("=" * 90)

    mae_weights = {}
    for site in sites:
        sdf = dev[dev["site"] == site]
        w, score = find_best_weights(sdf, grid, metric="mae")
        mae_weights[site] = w
        print(f"  {site}: MAE-optimal={w}, R²-optimal={find_best_weights(sdf, grid, metric='r2')[0]}")

    mae_eval = eval_weights(test, mae_weights)
    print(f"\n  MAE-optimized on test: R²={mae_eval['Overall']['r2']:.3f}, MAE={mae_eval['Overall']['mae']:.2f}")
    print(f"  Manual on test:        R²={manual_eval['Overall']['r2']:.3f}, MAE={manual_eval['Overall']['mae']:.2f}")

    # -----------------------------------------------------------------------
    # 9. FINE GRID (0.01 step) optimization
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("9. FINE GRID (step=0.01, 5151 combos) vs COARSE (step=0.05, 231)")
    print("=" * 90)

    fine_weights = {}
    for site in sites:
        sdf = dev[dev["site"] == site]
        w_coarse, s_coarse = find_best_weights(sdf, grid)
        w_fine, s_fine = find_best_weights(sdf, fine_grid)
        fine_weights[site] = w_fine
        if w_coarse != w_fine:
            print(f"  {site}: coarse={w_coarse}(R²={s_coarse:.4f}), fine={w_fine}(R²={s_fine:.4f})")
        else:
            print(f"  {site}: same at both resolutions: {w_coarse}")

    fine_eval = eval_weights(test, fine_weights)
    print(f"\n  Fine-grid on test: R²={fine_eval['Overall']['r2']:.3f} (coarse={eval_weights(test, {s: find_best_weights(dev[dev['site']==s], grid)[0] for s in sites})['Overall']['r2']:.3f}, manual={manual_eval['Overall']['r2']:.3f})")

    # -----------------------------------------------------------------------
    # 10. COMPONENT MODEL QUALITY COMPARISON
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("10. COMPONENT MODEL QUALITY: Does blending even help?")
    print("    Per-site R² of individual models on test set (seed=123)")
    print("=" * 90)

    print(f"\n{'Site':<15} {'N':>5}  {'XGB R²':>8} {'RF R²':>8} {'Naive R²':>8} {'Ens(manual)':>12} {'Best single':>12}")
    print("-" * 80)
    for site in sites:
        sdf = test[test["site"] == site]
        a = sdf["actual_da"].values
        xr2 = r2_score(a, sdf["pred_xgb"])
        rr2 = r2_score(a, sdf["pred_rf"])
        nr2 = r2_score(a, sdf["pred_naive"])
        er2 = manual_eval[site]["r2"]
        best = max(xr2, rr2, nr2)
        best_name = ["XGB", "RF", "Naive"][[xr2, rr2, nr2].index(best)]
        beats_best = "✓" if er2 > best else "✗"
        print(f"{site:<15} {len(a):>5}  {xr2:>8.3f} {rr2:>8.3f} {nr2:>8.3f} {er2:>12.3f} {best:>8.3f} ({best_name}) {beats_best}")
    print("-" * 80)
    # Overall
    xr2_all = r2_score(test["actual_da"], test["pred_xgb"])
    rr2_all = r2_score(test["actual_da"], test["pred_rf"])
    nr2_all = r2_score(test["actual_da"], test["pred_naive"])
    print(f"{'Overall':<15} {len(test):>5}  {xr2_all:>8.3f} {rr2_all:>8.3f} {nr2_all:>8.3f} {manual_eval['Overall']['r2']:>12.3f}")

    # -----------------------------------------------------------------------
    # 11. THE KEY QUESTION: Correlation between dev-optimal and test-optimal
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("11. DEV vs TEST OPTIMAL WEIGHT CORRELATION")
    print("    For each site: rank all 231 weight combos by R² on dev, see rank on test")
    print("=" * 90)

    for site in sites:
        dev_s = dev[dev["site"] == site]
        test_s = test[test["site"] == site]

        dev_ranked = []
        test_ranked = []
        for w in grid:
            dev_ranked.append((w, r2_score(dev_s["actual_da"], blend(dev_s, w))))
            test_ranked.append((w, r2_score(test_s["actual_da"], blend(test_s, w))))

        dev_ranked.sort(key=lambda x: -x[1])
        test_ranked.sort(key=lambda x: -x[1])

        dev_rank_map = {w: i+1 for i, (w, _) in enumerate(dev_ranked)}
        test_rank_map = {w: i+1 for i, (w, _) in enumerate(test_ranked)}

        # Rank correlation
        from scipy.stats import spearmanr
        dev_ranks = [dev_rank_map[w] for w in [x[0] for x in dev_ranked]]
        test_ranks_aligned = [test_rank_map[w] for w in [x[0] for x in dev_ranked]]
        rho, pval = spearmanr(dev_ranks, test_ranks_aligned)

        # Where does dev #1 rank on test?
        dev_best_w = dev_ranked[0][0]
        test_rank_of_dev_best = test_rank_map[dev_best_w]

        # Where does manual rank on each?
        mw = MANUAL_WEIGHTS[site]
        manual_dev_rank = dev_rank_map.get(mw, "N/A")
        manual_test_rank = test_rank_map.get(mw, "N/A")

        print(f"  {site}: Spearman ρ={rho:.3f} (p={pval:.1e})")
        print(f"    Dev-best {dev_best_w} → test rank {test_rank_of_dev_best}/{len(grid)}")
        print(f"    Manual {mw} → dev rank {manual_dev_rank}, test rank {manual_test_rank}")

    # -----------------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("SUMMARY: All weight strategies on independent test set (seed=123)")
    print("=" * 90)

    strategies = {
        "Manual (current)": manual_eval,
        "5-fold CV on dev": eval_weights(test, {s: find_best_weights(dev[dev["site"]==s], grid)[0] for s in sites}),
        "Temporal CV": temporal_eval,
        "Leave-one-year-out": loyo_eval,
        "Stacking (Ridge)": stacking_eval,
        "MAE-optimized": mae_eval,
        "Fine grid (0.01)": fine_eval,
        "Equal (1/3,1/3,1/3)": eval_weights(test, {s: (1/3,1/3,1/3) for s in sites}),
        "Oracle (cheating)": oracle_eval,
    }

    print(f"\n{'Strategy':<25} {'Overall R²':>12} {'Overall MAE':>12}")
    print("-" * 55)
    for name, ev in strategies.items():
        print(f"{name:<25} {ev['Overall']['r2']:>12.3f} {ev['Overall']['mae']:>12.2f}")


if __name__ == "__main__":
    main()
