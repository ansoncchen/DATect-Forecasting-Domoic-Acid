"""
eval_paper_metrics.py — DATect Paper Metrics Computation
=========================================================

Computes all regression and classification metrics needed for the DATect
paper tables:
  - Table 1: Overall model comparison (R², MAE, RMSE with bootstrap CIs)
  - Table 2: Per-site ensemble metrics (R², MAE, RMSE, accuracy)
  - Table 3: 4-category classification (precision, recall, F1 per category)
  - Appendix: Weight robustness comparison

Usage:
    python3 eval_paper_metrics.py [--seed 123] [--force-rerun] \
        [--output-dir eval_results/paper_metrics]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# DA category thresholds: Low < 5, Moderate 5-20, High 20-40, Extreme >= 40
# ---------------------------------------------------------------------------
DA_BINS = [-np.inf, 5.0, 20.0, 40.0, np.inf]
DA_LABELS = ["Low", "Moderate", "High", "Extreme"]
DA_LABEL_INDICES = [0, 1, 2, 3]


def da_category(values: np.ndarray) -> np.ndarray:
    """Map DA values (µg/g) to 0=Low, 1=Moderate, 2=High, 3=Extreme."""
    cats = np.digitize(values, bins=[5.0, 20.0, 40.0])  # bins=[5,20,40] -> 0,1,2,3
    return cats


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def bootstrap_metric(
    actual: np.ndarray,
    predicted: np.ndarray,
    metric_fn,
    B: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Return (ci_lo, ci_hi) percentile bootstrap 95% CI for metric_fn."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(actual)
    scores = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, size=n)
        scores[i] = metric_fn(actual[idx], predicted[idx])
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def bootstrap_delta_r2(
    actual: np.ndarray,
    pred_ensemble: np.ndarray,
    pred_other: np.ndarray,
    B: int = 10000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """
    Paired bootstrap for delta R² = R²(ensemble) - R²(other).
    Returns (observed_delta, ci_lo, ci_hi).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(actual)
    obs_delta = r2_score(actual, pred_ensemble) - r2_score(actual, pred_other)
    deltas = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, size=n)
        a = actual[idx]
        e = pred_ensemble[idx]
        o = pred_other[idx]
        deltas[i] = r2_score(a, e) - r2_score(a, o)
    return (
        float(obs_delta),
        float(np.percentile(deltas, 2.5)),
        float(np.percentile(deltas, 97.5)),
    )


# ---------------------------------------------------------------------------
# Load or run retrospective data
# ---------------------------------------------------------------------------

# Map model identifiers used in this script to the actual run key that the
# engine uses and the prediction column to read from the saved parquet.
#
#   run_key: value passed as model_type to run_retrospective_evaluation
#   pred_col: column in the saved parquet that holds predictions for this model
#
# Note: The engine runs XGB, RF, Naive and Ensemble in a single pass when
#       model_type='xgboost' (raw column name 'predicted_da' == XGB).
#       'ensemble' is also in that same run (column 'ensemble_prediction').
#       'rf' is in that run as 'predicted_da_rf'.
#       'naive' is in that run as 'naive_prediction'.
#       'linear' requires a separate run with model_type='linear'.

MODEL_RUN_MAP = {
    "ensemble": {"run_key": "xgboost", "pred_col": "ensemble_prediction"},
    "xgboost":  {"run_key": "xgboost", "pred_col": "predicted_da"},
    "rf":       {"run_key": "xgboost", "pred_col": "predicted_da_rf"},
    "naive":    {"run_key": "xgboost", "pred_col": "naive_prediction"},
    "linear":   {"run_key": "linear",  "pred_col": "predicted_da"},
}


def get_retro_path(retro_dir: str, run_key: str) -> str:
    return os.path.join(retro_dir, f"retro_regression_{run_key}.parquet")


def load_or_run_retro(
    run_key: str,
    retro_dir: str,
    seed: int,
    sample_fraction: float,
    force_rerun: bool,
) -> pd.DataFrame | None:
    """Load a cached retrospective parquet or run a fresh evaluation."""
    cache_path = get_retro_path(retro_dir, run_key)
    os.makedirs(retro_dir, exist_ok=True)

    if os.path.exists(cache_path) and not force_rerun:
        print(f"  Loading cached {run_key} results from {cache_path}")
        df = pd.read_parquet(cache_path)
        return df

    print(f"  Running retrospective evaluation for run_key='{run_key}' "
          f"(seed={seed}, fraction={sample_fraction})...")
    print("  (This may take 15-60 minutes on Hyak)")

    try:
        # Patch config before importing the engine so seed/fraction take effect
        import config as cfg
        original_seed = cfg.RANDOM_SEED
        original_frac = getattr(cfg, "TEST_SAMPLE_FRACTION", 0.20)

        cfg.RANDOM_SEED = seed
        cfg.TEST_SAMPLE_FRACTION = sample_fraction

        try:
            from forecasting.raw_forecast_engine import RawForecastEngine
            engine = RawForecastEngine()
            engine.random_seed = seed  # ensure instance also uses new seed
            results_df = engine.run_retrospective_evaluation(
                task="regression",
                model_type=run_key,
            )
        finally:
            cfg.RANDOM_SEED = original_seed
            cfg.TEST_SAMPLE_FRACTION = original_frac

        if results_df is not None and not results_df.empty:
            results_df.to_parquet(cache_path, index=False)
            print(f"  Saved {len(results_df)} results to {cache_path}")
        return results_df

    except Exception as exc:
        print(f"  WARNING: Could not run retrospective for '{run_key}': {exc}")
        return None


def get_model_predictions(
    model_name: str,
    retro_dir: str,
    seed: int,
    sample_fraction: float,
    force_rerun: bool,
) -> pd.DataFrame | None:
    """
    Return a DataFrame with columns [site, date, anchor_date, actual_da,
    predicted_da] for the given model, loading or running the parent run.
    """
    cfg = MODEL_RUN_MAP[model_name]
    run_key = cfg["run_key"]
    pred_col = cfg["pred_col"]

    df = load_or_run_retro(run_key, retro_dir, seed, sample_fraction, force_rerun)
    if df is None or df.empty:
        return None

    # Normalise date column name (engine renames test_date -> date)
    if "date" not in df.columns and "test_date" in df.columns:
        df = df.rename(columns={"test_date": "date"})

    if pred_col not in df.columns:
        print(f"  WARNING: column '{pred_col}' not found in parquet for run '{run_key}'. "
              f"Available: {list(df.columns)}")
        return None

    out = df[["site", "date", "anchor_date", "actual_da", pred_col]].copy()
    out = out.rename(columns={pred_col: "predicted_da"})
    out = out.dropna(subset=["actual_da", "predicted_da"])
    return out


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def compute_regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    rng: np.random.Generator,
    bootstrap_b: int = 2000,
    compute_ci: bool = True,
) -> dict:
    """Compute R², MAE, RMSE with optional 95% bootstrap CIs."""
    r2 = float(r2_score(actual, predicted))
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(math.sqrt(mean_squared_error(actual, predicted)))

    result = {"r2": r2, "mae": mae, "rmse": rmse, "n": int(len(actual))}

    if compute_ci and len(actual) >= 10:
        r2_lo, r2_hi = bootstrap_metric(actual, predicted, r2_score, B=bootstrap_b, rng=rng)
        mae_lo, mae_hi = bootstrap_metric(actual, predicted, mean_absolute_error, B=bootstrap_b, rng=rng)
        result.update({
            "r2_ci_lo": r2_lo, "r2_ci_hi": r2_hi,
            "mae_ci_lo": mae_lo, "mae_ci_hi": mae_hi,
        })
    else:
        result.update({"r2_ci_lo": None, "r2_ci_hi": None,
                       "mae_ci_lo": None, "mae_ci_hi": None})
    return result


# ---------------------------------------------------------------------------
# 4-category classification helpers
# ---------------------------------------------------------------------------

def compute_4cat_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """4-category accuracy based on DA thresholds applied to continuous values."""
    actual_cat = da_category(actual)
    pred_cat = da_category(predicted)
    return float(accuracy_score(actual_cat, pred_cat))


def compute_classification_table(actual: np.ndarray, predicted: np.ndarray) -> pd.DataFrame:
    """
    Per-category and overall precision/recall/F1 for 4 DA risk categories.
    Returns a DataFrame with one row per category + one 'Overall' row.
    """
    actual_cat = da_category(actual)
    pred_cat = da_category(predicted)

    rows = []
    for i, label in enumerate(DA_LABELS):
        mask_actual = actual_cat == i
        n_actual = int(mask_actual.sum())
        if n_actual == 0:
            rows.append({
                "category": label,
                "N_actual": 0,
                "precision": None,
                "recall": None,
                "f1": None,
            })
            continue
        prec = float(precision_score(actual_cat, pred_cat, labels=[i], average="macro", zero_division=0))
        rec  = float(recall_score(actual_cat, pred_cat, labels=[i], average="macro", zero_division=0))
        f1v  = float(f1_score(actual_cat, pred_cat, labels=[i], average="macro", zero_division=0))
        rows.append({
            "category": label,
            "N_actual": n_actual,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1v, 4),
        })

    # Overall (macro)
    overall_acc = float(accuracy_score(actual_cat, pred_cat))
    overall_prec = float(precision_score(actual_cat, pred_cat, average="macro", zero_division=0))
    overall_rec  = float(recall_score(actual_cat, pred_cat, average="macro", zero_division=0))
    overall_f1   = float(f1_score(actual_cat, pred_cat, average="macro", zero_division=0))
    rows.append({
        "category": "Overall (macro)",
        "N_actual": int(len(actual)),
        "precision": round(overall_prec, 4),
        "recall": round(overall_rec, 4),
        "f1": round(overall_f1, 4),
        "accuracy": round(overall_acc, 4),
    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def fmt_ci(lo, hi, decimals=3):
    if lo is None or hi is None:
        return "       n/a      "
    fmt = f".{decimals}f"
    return f"[{lo:{fmt}}, {hi:{fmt}}]"


def print_table1(table1_df: pd.DataFrame, delta_r2_dict: dict):
    """Print Table 1: Overall model comparison."""
    print("\n" + "=" * 90)
    print("TABLE 1 — Overall Model Comparison (seed=123, 40% independent test set)")
    print("=" * 90)
    header = (f"  {'Model':<12s}  {'N':>5s}  {'R²':>7s}  {'R² 95% CI':<20s}  "
              f"{'MAE':>7s}  {'MAE 95% CI':<20s}  {'RMSE':>7s}")
    print(header)
    print("  " + "-" * 86)
    for _, row in table1_df.iterrows():
        r2_ci = fmt_ci(row.get("r2_ci_lo"), row.get("r2_ci_hi"))
        mae_ci = fmt_ci(row.get("mae_ci_lo"), row.get("mae_ci_hi"))
        print(f"  {row['model']:<12s}  {int(row['n']):>5d}  "
              f"{row['r2']:>7.4f}  {r2_ci:<20s}  "
              f"{row['mae']:>7.3f}  {mae_ci:<20s}  "
              f"{row['rmse']:>7.3f}")

    print("\n  Paired bootstrap ΔR² vs ensemble (B=10000):")
    print(f"  {'Competitor':<12s}  {'ΔR²':>8s}  {'95% CI':<22s}")
    print("  " + "-" * 46)
    for model, vals in delta_r2_dict.items():
        if vals is None:
            print(f"  {model:<12s}  {'n/a':>8s}")
            continue
        delta, lo, hi = vals
        ci = fmt_ci(lo, hi)
        print(f"  {model:<12s}  {delta:>8.4f}  {ci:<22s}")


def print_table2(table2_df: pd.DataFrame):
    """Print Table 2: Per-site ensemble metrics."""
    print("\n" + "=" * 110)
    print("TABLE 2 — Per-Site Ensemble Metrics (seed=123, 40% test set)")
    print("=" * 110)
    header = (f"  {'Site':<18s}  {'N':>4s}  {'R²':>7s}  {'R² 95% CI':<20s}  "
              f"{'MAE':>7s}  {'MAE 95% CI':<20s}  {'RMSE':>7s}  {'4-cat Acc':>9s}")
    print(header)
    print("  " + "-" * 106)
    for _, row in table2_df.iterrows():
        r2_ci = fmt_ci(row.get("r2_ci_lo"), row.get("r2_ci_hi"))
        mae_ci = fmt_ci(row.get("mae_ci_lo"), row.get("mae_ci_hi"))
        acc = f"{row['cat4_acc']:.3f}" if row.get("cat4_acc") is not None else "  n/a"
        r2v = f"{row['r2']:.4f}" if row.get("r2") is not None else "   n/a"
        maev = f"{row['mae']:.3f}" if row.get("mae") is not None else "  n/a"
        rmsev = f"{row['rmse']:.3f}" if row.get("rmse") is not None else "  n/a"
        print(f"  {row['site']:<18s}  {int(row['n']):>4d}  "
              f"{r2v:>7s}  {r2_ci:<20s}  "
              f"{maev:>7s}  {mae_ci:<20s}  "
              f"{rmsev:>7s}  {acc:>9s}")


def print_table3(table3_df: pd.DataFrame):
    """Print Table 3: 4-category classification metrics."""
    print("\n" + "=" * 70)
    print("TABLE 3 — 4-Category Classification Metrics (Ensemble, seed=123)")
    print("=" * 70)
    header = f"  {'Category':<18s}  {'N_actual':>8s}  {'Precision':>9s}  {'Recall':>7s}  {'F1':>7s}"
    print(header)
    print("  " + "-" * 60)
    for _, row in table3_df.iterrows():
        prec = f"{row['precision']:.4f}" if row.get("precision") is not None else "   n/a"
        rec  = f"{row['recall']:.4f}"    if row.get("recall")    is not None else "   n/a"
        f1v  = f"{row['f1']:.4f}"        if row.get("f1")        is not None else "   n/a"
        acc_str = ""
        if "accuracy" in row and row["accuracy"] is not None:
            acc_str = f"  (accuracy={row['accuracy']:.4f})"
        print(f"  {row['category']:<18s}  {int(row['N_actual']):>8d}  "
              f"{prec:>9s}  {rec:>7s}  {f1v:>7s}{acc_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute all paper metrics for DATect (Tables 1–3 + Appendix)"
    )
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed for independent test set (default: 123)")
    parser.add_argument("--sample-fraction", type=float, default=0.40,
                        help="Fraction of valid points to sample per site (default: 0.40)")
    parser.add_argument("--force-rerun", action="store_true",
                        help="Force re-run retrospective evaluations even if parquets exist")
    parser.add_argument("--output-dir", default="eval_results/paper_metrics",
                        help="Output directory for CSV/JSON results")
    args = parser.parse_args()

    seed = args.seed
    sample_fraction = args.sample_fraction
    force_rerun = args.force_rerun
    output_dir = args.output_dir

    seed_suffix = f"_seed{seed}" if seed != 42 else ""
    retro_dir = os.path.join("eval_results", f"retro{seed_suffix}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(retro_dir, exist_ok=True)

    rng = np.random.default_rng(42)  # fixed RNG for reproducible bootstrap

    print(f"\nDATect Paper Metrics — seed={seed}, fraction={sample_fraction}")
    print(f"Retrospective cache dir: {retro_dir}")
    print(f"Output dir:              {output_dir}")
    print()

    # -----------------------------------------------------------------------
    # Step 1: Load or run retrospective data for all models
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Loading / running retrospective evaluations")
    print("=" * 60)

    MODEL_NAMES = ["ensemble", "xgboost", "rf", "naive", "linear"]
    model_data: dict[str, pd.DataFrame | None] = {}

    for model_name in MODEL_NAMES:
        print(f"\nModel: {model_name}")
        df = get_model_predictions(
            model_name, retro_dir, seed, sample_fraction, force_rerun
        )
        if df is not None:
            print(f"  {len(df)} test points loaded for '{model_name}'")
        else:
            print(f"  WARNING: no data available for '{model_name}'")
        model_data[model_name] = df

    ensemble_df = model_data.get("ensemble")
    if ensemble_df is None or ensemble_df.empty:
        print("\nERROR: Ensemble data is required but not available. "
              "Run on Hyak first or check parquet files.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 2: Table 1 — Overall regression metrics with bootstrap CIs
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Computing Table 1 (overall regression metrics)")
    print("=" * 60)

    table1_rows = []
    for model_name in MODEL_NAMES:
        df = model_data.get(model_name)
        if df is None or df.empty:
            print(f"  Skipping {model_name} (no data)")
            continue

        actual    = df["actual_da"].values.astype(float)
        predicted = df["predicted_da"].values.astype(float)

        metrics = compute_regression_metrics(actual, predicted, rng=rng, bootstrap_b=2000)
        metrics["model"] = model_name
        table1_rows.append(metrics)
        print(f"  {model_name:<12s}  N={metrics['n']:>4d}  R²={metrics['r2']:.4f}  "
              f"MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}")

    table1_df = pd.DataFrame(table1_rows)[
        ["model", "n", "r2", "r2_ci_lo", "r2_ci_hi", "mae", "mae_ci_lo", "mae_ci_hi", "rmse"]
    ]

    # Paired bootstrap delta R² vs ensemble
    print("\n  Computing paired bootstrap delta R² (B=10000)...")
    delta_r2_dict: dict[str, tuple | None] = {}
    ens_actual    = ensemble_df["actual_da"].values.astype(float)
    ens_predicted = ensemble_df["predicted_da"].values.astype(float)

    for model_name in MODEL_NAMES:
        if model_name == "ensemble":
            continue
        df = model_data.get(model_name)
        if df is None or df.empty:
            delta_r2_dict[model_name] = None
            continue

        # Align on shared (site, date) pairs for fair paired comparison
        merged = ensemble_df[["site", "date", "actual_da", "predicted_da"]].merge(
            df[["site", "date", "predicted_da"]].rename(
                columns={"predicted_da": "predicted_other"}
            ),
            on=["site", "date"],
            how="inner",
        )
        if len(merged) < 10:
            delta_r2_dict[model_name] = None
            continue

        a  = merged["actual_da"].values.astype(float)
        pe = merged["predicted_da"].values.astype(float)
        po = merged["predicted_other"].values.astype(float)
        delta, lo, hi = bootstrap_delta_r2(a, pe, po, B=10000, rng=rng)
        delta_r2_dict[model_name] = (delta, lo, hi)
        print(f"    vs {model_name:<10s}  ΔR²={delta:+.4f}  95%CI [{lo:+.4f}, {hi:+.4f}]")

    # -----------------------------------------------------------------------
    # Step 3: Table 2 — Per-site ensemble metrics
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Computing Table 2 (per-site ensemble metrics)")
    print("=" * 60)

    import config as cfg
    all_sites = list(cfg.SITES.keys())

    table2_rows = []
    for site in all_sites:
        site_df = ensemble_df[ensemble_df["site"] == site].copy()
        n = len(site_df)
        if n == 0:
            print(f"  {site:<20s}  no data")
            continue

        actual    = site_df["actual_da"].values.astype(float)
        predicted = site_df["predicted_da"].values.astype(float)

        compute_ci = n >= 5
        metrics = compute_regression_metrics(
            actual, predicted, rng=rng, bootstrap_b=2000, compute_ci=compute_ci
        )
        cat4_acc = compute_4cat_accuracy(actual, predicted)

        row = {
            "site": site,
            "n": n,
            "r2": metrics["r2"],
            "r2_ci_lo": metrics.get("r2_ci_lo"),
            "r2_ci_hi": metrics.get("r2_ci_hi"),
            "mae": metrics["mae"],
            "mae_ci_lo": metrics.get("mae_ci_lo"),
            "mae_ci_hi": metrics.get("mae_ci_hi"),
            "rmse": metrics["rmse"],
            "cat4_acc": cat4_acc,
        }
        table2_rows.append(row)
        print(f"  {site:<20s}  N={n:>4d}  R²={metrics['r2']:.4f}  "
              f"MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}  "
              f"4-cat-acc={cat4_acc:.3f}")

    table2_df = pd.DataFrame(table2_rows)

    # -----------------------------------------------------------------------
    # Step 4: Table 3 — 4-category classification metrics (ensemble)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Computing Table 3 (4-category classification)")
    print("=" * 60)

    ens_actual    = ensemble_df["actual_da"].values.astype(float)
    ens_predicted = ensemble_df["predicted_da"].values.astype(float)
    table3_df = compute_classification_table(ens_actual, ens_predicted)

    for _, row in table3_df.iterrows():
        print(f"  {row['category']:<20s}  N={int(row['N_actual']):>5d}  "
              f"P={row.get('precision', 'n/a')!r}  "
              f"R={row.get('recall', 'n/a')!r}  "
              f"F1={row.get('f1', 'n/a')!r}")

    # -----------------------------------------------------------------------
    # Step 5: Appendix — Weight robustness comparison
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Weight robustness comparison (Appendix)")
    print("=" * 60)

    # For now, only the 'manual' strategy (current per_site_models.py weights)
    # is always available via the ensemble results already loaded.
    weight_rows = []
    manual_metrics = compute_regression_metrics(
        ens_actual, ens_predicted, rng=rng, bootstrap_b=2000
    )
    manual_metrics["strategy"] = "manual (per_site_models.py)"
    weight_rows.append(manual_metrics)
    print(f"  manual  R²={manual_metrics['r2']:.4f}  MAE={manual_metrics['mae']:.3f}  "
          f"RMSE={manual_metrics['rmse']:.3f}")

    # Check for alternative weight strategies in eval_results/
    alt_candidates = []
    for fname in sorted(os.listdir("eval_results")) if os.path.isdir("eval_results") else []:
        if fname.startswith("retro") and fname != f"retro{seed_suffix}":
            alt_dir = os.path.join("eval_results", fname)
            alt_path = os.path.join(alt_dir, "retro_regression_xgboost.parquet")
            if os.path.exists(alt_path):
                alt_candidates.append((fname, alt_path))

    for fname, alt_path in alt_candidates:
        try:
            alt_df = pd.read_parquet(alt_path)
            if "ensemble_prediction" in alt_df.columns and "actual_da" in alt_df.columns:
                alt_actual    = alt_df["actual_da"].dropna().values.astype(float)
                alt_predicted = alt_df["ensemble_prediction"].dropna().values.astype(float)
                min_len = min(len(alt_actual), len(alt_predicted))
                if min_len > 10:
                    alt_metrics = compute_regression_metrics(
                        alt_actual[:min_len], alt_predicted[:min_len], rng=rng, bootstrap_b=2000
                    )
                    alt_metrics["strategy"] = fname
                    weight_rows.append(alt_metrics)
                    print(f"  {fname:<30s}  R²={alt_metrics['r2']:.4f}  "
                          f"MAE={alt_metrics['mae']:.3f}  RMSE={alt_metrics['rmse']:.3f}")
        except Exception as exc:
            print(f"  WARNING: Could not load {alt_path}: {exc}")

    appendix_df = pd.DataFrame(weight_rows)

    # -----------------------------------------------------------------------
    # Print formatted paper tables
    # -----------------------------------------------------------------------
    print_table1(table1_df, delta_r2_dict)
    print_table2(table2_df)
    print_table3(table3_df)

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Saving outputs...")
    print("=" * 60)

    # Table CSVs
    t1_path = os.path.join(output_dir, "table1_model_comparison.csv")
    t2_path = os.path.join(output_dir, "table2_per_site.csv")
    t3_path = os.path.join(output_dir, "table3_classification.csv")

    table1_df.to_csv(t1_path, index=False)
    table2_df.to_csv(t2_path, index=False)
    table3_df.to_csv(t3_path, index=False)
    print(f"  Saved {t1_path}")
    print(f"  Saved {t2_path}")
    print(f"  Saved {t3_path}")

    # JSON with all values including CIs
    def _safe(v):
        if v is None:
            return None
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        return v

    def _df_to_records(df: pd.DataFrame) -> list[dict]:
        return [{k: _safe(v) for k, v in row.items()} for row in df.to_dict("records")]

    delta_r2_serializable = {}
    for k, v in delta_r2_dict.items():
        if v is None:
            delta_r2_serializable[k] = None
        else:
            delta_r2_serializable[k] = {
                "delta_r2": _safe(v[0]),
                "ci_lo": _safe(v[1]),
                "ci_hi": _safe(v[2]),
            }

    paper_metrics = {
        "meta": {
            "seed": seed,
            "sample_fraction": sample_fraction,
            "n_ensemble_test_points": int(len(ensemble_df)),
            "bootstrap_b_ci": 2000,
            "bootstrap_b_delta": 10000,
            "bootstrap_rng_seed": 42,
            "da_category_thresholds": {"Low": "<5", "Moderate": "5-20",
                                        "High": "20-40", "Extreme": ">=40"},
        },
        "table1_model_comparison": _df_to_records(table1_df),
        "table1_delta_r2_vs_ensemble": delta_r2_serializable,
        "table2_per_site": _df_to_records(table2_df),
        "table3_classification": _df_to_records(table3_df),
        "appendix_weight_robustness": _df_to_records(appendix_df),
    }

    json_path = os.path.join(output_dir, "paper_metrics.json")
    with open(json_path, "w") as f:
        json.dump(paper_metrics, f, indent=2, default=str)
    print(f"  Saved {json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
