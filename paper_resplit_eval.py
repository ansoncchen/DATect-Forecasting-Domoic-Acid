#!/usr/bin/env python3
"""
DATect Independent Test Set Evaluation (seed=123, fraction=0.40)

Runs the full retrospective evaluation with a different random seed and
larger sample fraction than was used during per-site configuration tuning
(seed=42, fraction=0.20). This ensures reported metrics are independent
of configuration selection.

Also computes bootstrap 95% CIs on the results.

Usage (run on Hyak):
    python3 paper_resplit_eval.py

Output:
    cache_seed123/retrospective/*.parquet   (cached results)
    paper_resplit_results.json              (metrics + bootstrap CIs)
    paper_resplit_latex.txt                 (copy-paste LaTeX table rows)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# ── Configuration overrides ──────────────────────────────────────────────────
EVAL_SEED = 123
EVAL_FRACTION = 0.40
CACHE_DIR = Path("./cache_seed123")
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42


def bootstrap_metrics(y_true, y_pred, n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """Compute bootstrap 95% CIs for R², MAE, RMSE."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    r2s, maes, rmses = [], [], []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yt, yp = y_true[idx], y_pred[idx]
        if np.std(yt) == 0:
            continue
        r2s.append(r2_score(yt, yp))
        maes.append(mean_absolute_error(yt, yp))
        rmses.append(np.sqrt(mean_squared_error(yt, yp)))

    return {
        'r2': float(r2_score(y_true, y_pred)),
        'r2_ci_lo': float(np.percentile(r2s, 2.5)),
        'r2_ci_hi': float(np.percentile(r2s, 97.5)),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'mae_ci_lo': float(np.percentile(maes, 2.5)),
        'mae_ci_hi': float(np.percentile(maes, 97.5)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'rmse_ci_lo': float(np.percentile(rmses, 2.5)),
        'rmse_ci_hi': float(np.percentile(rmses, 97.5)),
        'n': n,
    }


def run_evaluation(task, model_type):
    """Run retrospective evaluation with seed=123, fraction=0.40."""
    import config

    # Override seed and fraction
    original_seed = config.RANDOM_SEED
    original_fraction = config.TEST_SAMPLE_FRACTION
    config.RANDOM_SEED = EVAL_SEED
    config.TEST_SAMPLE_FRACTION = EVAL_FRACTION

    try:
        from backend.api import get_forecast_engine, clean_for_json

        # Clear cached engine to pick up new config
        import backend.api as api_module
        if hasattr(api_module, '_forecast_engine'):
            api_module._forecast_engine = None

        engine = get_forecast_engine()
        n_anchors = getattr(config, 'N_RANDOM_ANCHORS', 500)

        print(f"  Running {task} + {model_type} (seed={EVAL_SEED}, fraction={EVAL_FRACTION})...")

        results_df = engine.run_retrospective_evaluation(
            task=task,
            model_type=model_type,
            n_anchors=n_anchors,
            min_test_date="2008-01-01"
        )

        if results_df is not None and not results_df.empty:
            # Save to cache
            cache_subdir = CACHE_DIR / "retrospective"
            cache_subdir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_subdir / f"{task}_{model_type}"

            results_df.to_parquet(f"{cache_file}.parquet", index=False)

            # Also save JSON
            results_json = []
            for _, row in results_df.iterrows():
                record = {
                    'date': row['date'].strftime('%Y-%m-%d') if pd.notnull(row.get('date')) else None,
                    'site': row['site'],
                    'actual_da': clean_for_json(row['actual_da']) if 'actual_da' in row and pd.notnull(row.get('actual_da')) else None,
                    'predicted_da': clean_for_json(row['predicted_da']) if 'predicted_da' in row and pd.notnull(row.get('predicted_da')) else None,
                    'actual_category': clean_for_json(row['actual_category']) if 'actual_category' in row and pd.notnull(row.get('actual_category')) else None,
                    'predicted_category': clean_for_json(row['predicted_category']) if 'predicted_category' in row and pd.notnull(row.get('predicted_category')) else None,
                }
                results_json.append(record)

            with open(f"{cache_file}.json", 'w') as f:
                json.dump(results_json, f, default=str, indent=2)

            print(f"    Saved {len(results_df)} predictions to {cache_file}")
            return results_df
        else:
            print(f"    ERROR: No results")
            return None

    finally:
        # Restore original config
        config.RANDOM_SEED = original_seed
        config.TEST_SAMPLE_FRACTION = original_fraction


def compute_classification_metrics(results_df):
    """Compute per-category classification metrics."""
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

    y_true = results_df['actual_category'].values
    y_pred = results_df['predicted_category'].values

    # Overall accuracy
    acc = accuracy_score(y_true, y_pred)

    # Per-category
    labels = [0, 1, 2, 3]
    label_names = ['Low', 'Moderate', 'High', 'Extreme']
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)

    categories = {}
    for i, name in enumerate(label_names):
        categories[name] = {
            'n': int(support[i]),
            'precision': float(prec[i]),
            'recall': float(rec[i]),
            'f1': float(f1[i]),
        }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        'accuracy': float(acc),
        'categories': categories,
        'confusion_matrix': cm.tolist(),
    }


def format_ci(val, lo, hi, decimals=2):
    """Format as 0.81 [0.54, 0.93]"""
    lo_str = f"{lo:.{decimals}f}" if lo >= 0 else f"$-${abs(lo):.{decimals}f}"
    hi_str = f"{hi:.{decimals}f}"
    return f"{val:.{decimals+1}f} [{lo_str}, {hi_str}]"


def main():
    start_time = datetime.now()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"DATect Independent Test Set Evaluation")
    print(f"  Seed: {EVAL_SEED} (independent of config-tuning seed=42)")
    print(f"  Sample fraction: {EVAL_FRACTION} (2x development fraction)")
    print("=" * 70)

    # ── Run all evaluation combinations ───────────────────────────────────
    combinations = [
        ("regression", "ensemble"),
        ("regression", "xgboost"),
        ("regression", "rf"),
        ("regression", "naive"),
        ("regression", "linear"),
        ("classification", "ensemble"),
        ("classification", "naive"),
        ("classification", "logistic"),
    ]

    all_dfs = {}
    for task, model_type in combinations:
        df = run_evaluation(task, model_type)
        if df is not None:
            all_dfs[(task, model_type)] = df

    # ── Compute bootstrap CIs for regression models ───────────────────────
    print("\n" + "=" * 70)
    print("BOOTSTRAP CONFIDENCE INTERVALS (B=2000)")
    print("=" * 70)

    regression_results = {}
    for model_type in ['ensemble', 'xgboost', 'rf', 'naive', 'linear']:
        key = ('regression', model_type)
        if key not in all_dfs:
            continue
        df = all_dfs[key]
        y_true = df['actual_da'].values
        y_pred = df['predicted_da'].values

        overall = bootstrap_metrics(y_true, y_pred)
        per_site = {}
        for site in sorted(df['site'].unique()):
            mask = df['site'] == site
            per_site[site] = bootstrap_metrics(
                df.loc[mask, 'actual_da'].values,
                df.loc[mask, 'predicted_da'].values,
            )

        regression_results[model_type] = {'overall': overall, 'per_site': per_site}
        print(f"\n  {model_type}: R²={overall['r2']:.3f} [{overall['r2_ci_lo']:.3f}, {overall['r2_ci_hi']:.3f}], "
              f"MAE={overall['mae']:.2f} [{overall['mae_ci_lo']:.2f}, {overall['mae_ci_hi']:.2f}], N={overall['n']}")

    # ── Compute classification metrics ────────────────────────────────────
    classification_results = {}
    for model_type in ['ensemble', 'naive', 'logistic']:
        key = ('classification', model_type)
        if key not in all_dfs:
            continue
        classification_results[model_type] = compute_classification_metrics(all_dfs[key])

    # ── Per-site accuracy for ensemble ────────────────────────────────────
    if ('classification', 'ensemble') in all_dfs:
        clf_df = all_dfs[('classification', 'ensemble')]
        per_site_acc = {}
        for site in sorted(clf_df['site'].unique()):
            mask = clf_df['site'] == site
            correct = (clf_df.loc[mask, 'actual_category'] == clf_df.loc[mask, 'predicted_category']).sum()
            total = mask.sum()
            per_site_acc[site] = float(correct / total)
    else:
        per_site_acc = {}

    # ── Save all results ──────────────────────────────────────────────────
    output = {
        'config': {
            'seed': EVAL_SEED,
            'fraction': EVAL_FRACTION,
            'n_bootstrap': N_BOOTSTRAP,
        },
        'regression': regression_results,
        'classification': classification_results,
        'per_site_accuracy': per_site_acc,
    }

    with open('paper_resplit_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # ── Generate LaTeX table rows ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("LATEX TABLE ROWS (copy into paper)")
    print("=" * 70)

    site_order = [
        'Twin Harbors', 'Copalis', 'Kalaloch', 'Quinault', 'Long Beach',
        'Clatsop Beach', 'Coos Bay', 'Newport', 'Gold Beach', 'Cannon Beach',
    ]

    tier_map = {
        'Twin Harbors': 'High-skill', 'Copalis': 'High-skill', 'Kalaloch': 'High-skill',
        'Quinault': 'High-skill', 'Long Beach': 'High-skill',
        'Clatsop Beach': 'Moderate', 'Coos Bay': 'Moderate', 'Newport': 'Moderate',
        'Gold Beach': 'Low-skill', 'Cannon Beach': 'Low-skill',
    }

    if 'ensemble' in regression_results:
        ens = regression_results['ensemble']

        latex_lines = []
        with open('paper_resplit_latex.txt', 'w') as f:
            f.write("% Per-site results table rows (seed=123, fraction=0.40)\n")
            f.write("% Generated by paper_resplit_eval.py\n\n")

            prev_tier = None
            for site in site_order:
                if site not in ens['per_site']:
                    continue
                r = ens['per_site'][site]
                acc = per_site_acc.get(site, 0)
                tier = tier_map[site]

                if prev_tier and tier != prev_tier:
                    f.write("\\addlinespace\n")
                prev_tier = tier

                # Format R² CI
                r2_lo = f"${-abs(r['r2_ci_lo']):.2f}$" if r['r2_ci_lo'] < 0 else f"{r['r2_ci_lo']:.2f}"
                r2_hi = f"{r['r2_ci_hi']:.2f}"
                if r['r2'] >= 0.6:
                    r2_str = f"\\textbf{{{r['r2']:.3f}}} [{r2_lo}, {r2_hi}]"
                elif r['r2'] < 0:
                    r2_str = f"$-${abs(r['r2']):.3f} [{r2_lo}, {r2_hi}]"
                else:
                    r2_str = f"{r['r2']:.3f} [{r2_lo}, {r2_hi}]"

                # Format MAE CI
                mae_str = f"{r['mae']:.2f} [{r['mae_ci_lo']:.2f}, {r['mae_ci_hi']:.2f}]"

                line = f"{site:<18} & {r['n']} & {r2_str} & {mae_str} & {r['rmse']:.2f} & {acc:.3f} & {tier} \\\\"
                f.write(line + "\n")
                print(line)

            # Overall
            r = ens['overall']
            overall_acc = classification_results.get('ensemble', {}).get('accuracy', 0)
            r2_lo = f"{r['r2_ci_lo']:.2f}"
            r2_hi = f"{r['r2_ci_hi']:.2f}"
            f.write("\\midrule\n")
            line = f"\\textbf{{Overall}} & \\textbf{{{r['n']:,}}} & \\textbf{{{r['r2']:.3f}}} [{r2_lo}, {r2_hi}] & \\textbf{{{r['mae']:.2f}}} [{r['mae_ci_lo']:.2f}, {r['mae_ci_hi']:.2f}] & \\textbf{{{r['rmse']:.2f}}} & \\textbf{{{overall_acc:.3f}}} & --- \\\\"
            f.write(line + "\n")
            print(line)

        print(f"\nSaved to paper_resplit_latex.txt")

    # ── Model comparison table ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("MODEL COMPARISON TABLE (Table 5 in paper)")
    print("=" * 70)
    print(f"{'Model':<35} {'R²':>8} {'R² CI':>18} {'MAE':>8} {'RMSE':>8} {'N':>6}")
    print("-" * 85)
    for model_type in ['ensemble', 'linear', 'rf', 'xgboost', 'naive']:
        if model_type not in regression_results:
            continue
        r = regression_results[model_type]['overall']
        name = {
            'ensemble': 'Ensemble (XGB+RF+Naive)',
            'linear': 'Ridge Regression (Linear)',
            'rf': 'Random Forest',
            'xgboost': 'XGBoost',
            'naive': 'Naive Persistence',
        }[model_type]
        print(f"{name:<35} {r['r2']:>8.3f} [{r['r2_ci_lo']:.3f}, {r['r2_ci_hi']:.3f}] {r['mae']:>8.2f} {r['rmse']:>8.2f} {r['n']:>6}")

    # ── Classification table ──────────────────────────────────────────────
    if 'ensemble' in classification_results:
        print("\n" + "=" * 70)
        print("CLASSIFICATION TABLE (Table 7 in paper)")
        print("=" * 70)
        clf = classification_results['ensemble']
        print(f"Overall accuracy: {clf['accuracy']:.3f}")
        print(f"{'Category':<12} {'N':>5} {'Prec':>8} {'Recall':>8} {'F1':>8}")
        print("-" * 45)
        for cat in ['Low', 'Moderate', 'High', 'Extreme']:
            c = clf['categories'][cat]
            print(f"{cat:<12} {c['n']:>5} {c['precision']:>8.2f} {c['recall']:>8.2f} {c['f1']:>8.2f}")

        print("\nConfusion matrix:")
        cm = np.array(clf['confusion_matrix'])
        labels = ['Low', 'Mod', 'High', 'Ext']
        print(f"{'':>12} {'Low':>6} {'Mod':>6} {'High':>6} {'Ext':>6}")
        for i, label in enumerate(labels):
            print(f"{label:>12} {cm[i,0]:>6} {cm[i,1]:>6} {cm[i,2]:>6} {cm[i,3]:>6}")

    elapsed = datetime.now() - start_time
    print(f"\n{'='*70}")
    print(f"Complete in {elapsed}")
    print(f"Results: paper_resplit_results.json")
    print(f"LaTeX:   paper_resplit_latex.txt")
    print(f"Cache:   {CACHE_DIR}/retrospective/")


if __name__ == "__main__":
    main()
