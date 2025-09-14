"""
Hyperparameter Search for XGBoost
=================================

Runs a lightweight, leak-free random search over key XGBoost
hyperparameters using the existing retrospective evaluation engine.

Usage:
  python3 -m forecasting.hyperparam_search --task regression --trials 50 --anchors 500

Notes:
- Uses the same temporal safeguards as the main pipeline
- Evaluates across randomly sampled anchor points per site
- Selects best params by REGRESSION metrics only (R², spike F1, MAE)
- Saves best params to cache/hyperparams/regression_xgboost_best.json
- Also mirrors best regression params to cache/hyperparams/classification_xgboost_best.json
 - First evaluates a BASELINE using current config; each trial is compared to baseline
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

import config
from .forecast_engine import ForecastEngine


def _sample_loguniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def sample_params(task: str, rng: np.random.Generator) -> Dict:
    """Sample a candidate hyperparameter set for XGBoost Random Forest."""
    params = {
        "n_estimators": int(rng.integers(50, 501)),         # 50–500 trees in forest
        "max_depth": int(rng.integers(3, 15)),              # 3–15 (deeper for RF)
        "subsample": float(rng.uniform(0.6, 1.0)),          # 0.6–1.0
        "colsample_bynode": float(rng.uniform(0.6, 1.0)),   # 0.6–1.0 (RF uses bynode)
        "min_child_weight": int(rng.integers(1, 11)),       # 1–10
        "reg_alpha": float(rng.uniform(0.0, 0.5)),          # 0.0–0.5
        "reg_lambda": float(rng.uniform(0.1, 2.0)),         # 0.1–2.0
        "tree_method": "hist",
    }
    if task == "classification":
        params["eval_metric"] = "logloss"
    return params


def evaluate_params_regression(params: Dict, n_anchors: int, min_test_date: str,
                               w_r2: float, w_f1: float, w_mae: float) -> Tuple[float, Dict]:
    """
    Evaluate params using REGRESSION metrics only and compute a composite score.

    Objective (minimized) = - (w_r2 * R2 + w_f1 * F1_spike - w_mae * MAE_norm)
    where MAE_norm = MAE / IQR(actual_da) with safe fallbacks.
    """
    # Backup and override regression params
    original = dict(getattr(config, "XGB_REGRESSION_PARAMS", {}))
    config.XGB_REGRESSION_PARAMS.update(params)

    try:
        engine = ForecastEngine(validate_on_init=False)
        df = engine.run_retrospective_evaluation(
            task="regression", model_type="xgboost", n_anchors=n_anchors, min_test_date=min_test_date,
            model_params_override=params
        )

        if df is None or df.empty:
            return (float("inf"), {"status": "no_results"})

        import pandas as pd
        from sklearn.metrics import r2_score, mean_absolute_error, f1_score

        valid = df.dropna(subset=["actual_da", "predicted_da"]) if isinstance(df, pd.DataFrame) else engine.results_df
        if valid is None or valid.empty:
            return (float("inf"), {"status": "no_valid_pairs"})

        y_true = valid["actual_da"].values
        y_pred = valid["predicted_da"].values

        # Core metrics
        r2 = float(r2_score(y_true, y_pred))
        mae = float(mean_absolute_error(y_true, y_pred))
        # Spike-based F1 using configured threshold
        thr = getattr(config, "SPIKE_THRESHOLD", 20.0)
        y_true_bin = (y_true > thr).astype(int)
        y_pred_bin = (y_pred > thr).astype(int)
        f1 = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))

        # Normalize MAE by IQR to make it comparable
        q25, q75 = np.percentile(y_true, [25, 75])
        iqr = max(q75 - q25, 1e-6)
        mae_norm = float(mae / iqr)
        # Clip to reduce outlier domination
        mae_norm = float(min(mae_norm, 5.0))

        # Normalize weights (robust to user-provided values)
        ws = np.array([w_r2, w_f1, w_mae], dtype=float)
        if ws.sum() <= 0:
            ws = np.array([0.45, 0.45, 0.10])
        ws = ws / ws.sum()
        w_r2, w_f1, w_mae = ws.tolist()

        composite = (w_r2 * r2) + (w_f1 * f1) - (w_mae * mae_norm)
        obj = -float(composite)

        metrics = {
            "r2": r2,
            "f1_spike": f1,
            "mae": mae,
            "mae_norm": mae_norm,
            "composite": composite,
            "n": int(len(valid)),
            "weights": {"w_r2": w_r2, "w_f1": w_f1, "w_mae": w_mae},
        }
        return (obj, metrics)
    finally:
        config.XGB_REGRESSION_PARAMS.update(original)


def run_search(task: str, trials: int, anchors: int, min_test_date: str, seed: int,
               w_r2: float, w_f1: float, w_mae: float,
               mirror_classification: bool = True) -> Dict:
    rng = np.random.default_rng(seed)

    # 1) Evaluate BASELINE (current config)
    baseline_params = dict(getattr(config, "XGB_REGRESSION_PARAMS", {}))
    base_obj, base_metrics = evaluate_params_regression(baseline_params, anchors, min_test_date, w_r2, w_f1, w_mae)
    print(
        f"[BASELINE] obj={base_obj:.6f} composite={base_metrics.get('composite'):.4f} "
        f"r2={base_metrics.get('r2'):.4f} f1={base_metrics.get('f1_spike'):.4f} mae={base_metrics.get('mae'):.4f}"
    )

    best_obj = base_obj
    best_params: Dict | None = dict(baseline_params)
    best_metrics: Dict | None = dict(base_metrics)

    for t in range(1, trials + 1):
        params = sample_params("regression", rng)
        obj, metrics = evaluate_params_regression(params, anchors, min_test_date, w_r2, w_f1, w_mae)
        improved = obj < base_obj  # lower objective is better
        dr2 = float(metrics.get('r2', 0.0) - base_metrics.get('r2', 0.0))
        df1 = float(metrics.get('f1_spike', 0.0) - base_metrics.get('f1_spike', 0.0))
        dcomp = float(metrics.get('composite', 0.0) - base_metrics.get('composite', 0.0))
        dmae = float(metrics.get('mae', 0.0) - base_metrics.get('mae', 0.0))
        print(
            f"[TRIAL {t:03d}] obj={obj:.6f} Δcomp={dcomp:+.4f} Δr2={dr2:+.4f} Δf1={df1:+.4f} Δmae={dmae:+.4f} "
            f"improved={improved} params={{'n_estimators': {params['n_estimators']}, 'max_depth': {params['max_depth']}, 'learning_rate': {params['learning_rate']:.4f}}}"
        )

        if obj < best_obj:
            best_obj, best_params, best_metrics = obj, params, metrics

    result = {
        "task": "regression",  # selection is always based on regression metrics
        "objective": "-(w_r2*r2 + w_f1*f1 - w_mae*mae_norm)",
        "baseline_objective": base_obj,
        "best_objective": best_obj,
        "improved_over_baseline": best_obj < base_obj,
        "best_params": best_params,
        "best_metrics": best_metrics,
        "baseline_params": baseline_params,
        "baseline_metrics": base_metrics,
        "trials": trials,
        "anchors": anchors,
        "min_test_date": min_test_date,
        "random_seed": seed,
    }

    # Save to cache
    out_dir = Path("cache") / "hyperparams"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Always save regression best
    out_path = out_dir / "regression_xgboost_best.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"[INFO] Saved best regression params to {out_path}")

    # Mirror to classification JSON for convenience (copy-over policy)
    if mirror_classification and best_params is not None:
        cls_copy = {
            "task": "classification",
            "source": "copied_from_regression",
            "best_params": best_params,
            "note": "Use these for classification; keep eval_metric='logloss' if desired.",
        }
        cls_path = out_dir / "classification_xgboost_best.json"
        with cls_path.open("w") as f:
            json.dump(cls_copy, f, indent=2)
        print(f"[INFO] Mirrored params to {cls_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Random search XGBoost hyperparameters with leak-free evaluation")
    parser.add_argument("--task", choices=["regression", "classification"], default="regression",
                        help="Used only for output naming/compat; selection always uses regression metrics")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--anchors", type=int, default=120)
    parser.add_argument("--min-test-date", dest="min_test_date", default="2008-01-01")
    parser.add_argument("--seed", type=int, default=getattr(config, "RANDOM_SEED", 42))
    # Defaults reflect: R² most important, F1 slightly less, MAE least
    parser.add_argument("--w-r2", type=float, default=0.50, help="Weight for R² (maximize)")
    parser.add_argument("--w-f1", type=float, default=0.35, help="Weight for spike F1 (maximize)")
    parser.add_argument("--w-mae", type=float, default=0.15, help="Weight for normalized MAE (minimize)")
    parser.add_argument("--no-mirror-classification", action="store_true",
                        help="Do not write classification JSON copy of regression-best params")
    parser.add_argument("--write-config", action="store_true",
                        help="Write best regression params to config.XGB_REGRESSION_PARAMS and mirror to XGB_CLASSIFICATION_PARAMS")
    args = parser.parse_args()

    # Search on regression metrics only
    res = run_search(
        args.task, args.trials, args.anchors, args.min_test_date, args.seed,
        args.w_r2, args.w_f1, args.w_mae,
        mirror_classification=(not args.no_mirror_classification),
    )

    # Optionally write back into config.py
    if args.write_config and res.get("best_params"):
        best = res["best_params"]
        # Prepare classification params by adding eval_metric if not present
        cls_params = dict(best)
        cls_params.setdefault("eval_metric", "logloss")

        # Update the in-memory config
        config.XGB_REGRESSION_PARAMS.update(best)
        config.XGB_CLASSIFICATION_PARAMS.update(cls_params)

        # Persist to config.py (simple text replacement on dict literals)
        cfg_path = Path(__file__).resolve().parents[1] / "config.py"
        try:
            text = cfg_path.read_text()
            import re
            def dict_to_literal(d: Dict) -> str:
                # Deterministic order for readability
                keys = sorted(d.keys())
                inner = ",\n    ".join([f"\"{k}\": {json.dumps(d[k])}" for k in keys])
                return "{\n    " + inner + "\n}"

            new_reg = f"XGB_REGRESSION_PARAMS = {dict_to_literal(best)}"
            new_cls = f"XGB_CLASSIFICATION_PARAMS = {dict_to_literal(cls_params)}"

            text = re.sub(r"XGB_REGRESSION_PARAMS\s*=\s*\{[\s\S]*?\}\n", new_reg + "\n", text)
            text = re.sub(r"XGB_CLASSIFICATION_PARAMS\s*=\s*\{[\s\S]*?\}\n", new_cls + "\n", text)
            cfg_path.write_text(text)
            print(f"[INFO] Updated config.py with best params for regression and classification")
        except Exception as e:
            print(f"[WARN] Could not update config.py automatically: {e}")


if __name__ == "__main__":
    main()
