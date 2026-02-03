"""
Optuna Hyperparameter Optimization for XGBoost
==============================================

Usage:
  python3 -m forecasting.hyperparam_optuna --trials 100 --anchors 500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import optuna

import config
from .forecast_engine import ForecastEngine


def evaluate_params_regression(params: Dict, n_anchors: int, min_test_date: str,
                               w_r2: float, w_f1: float, w_mae: float) -> Dict:
    engine = ForecastEngine(validate_on_init=False)
    df = engine.run_retrospective_evaluation(
        task="regression", model_type="xgboost", n_anchors=n_anchors, min_test_date=min_test_date,
        model_params_override=params
    )

    if df is None or df.empty:
        return {"status": "no_results"}

    import pandas as pd
    from sklearn.metrics import r2_score, mean_absolute_error, f1_score

    valid = df.dropna(subset=["actual_da", "predicted_da"]) if isinstance(df, pd.DataFrame) else engine.results_df
    if valid is None or valid.empty:
        return {"status": "no_valid_pairs"}

    y_true = valid["actual_da"].values
    y_pred = valid["predicted_da"].values

    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    thr = getattr(config, "SPIKE_THRESHOLD", 20.0)
    y_true_bin = (y_true > thr).astype(int)
    y_pred_bin = (y_pred > thr).astype(int)
    f1 = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))

    q25, q75 = np.percentile(y_true, [25, 75])
    iqr = max(q75 - q25, 1e-6)
    mae_norm = float(min(mae / iqr, 5.0))

    ws = np.array([w_r2, w_f1, w_mae], dtype=float)
    if ws.sum() <= 0:
        ws = np.array([0.45, 0.45, 0.10])
    ws = ws / ws.sum()
    w_r2, w_f1, w_mae = ws.tolist()

    composite = (w_r2 * r2) + (w_f1 * f1) - (w_mae * mae_norm)

    return {
        "r2": r2,
        "f1_spike": f1,
        "mae": mae,
        "mae_norm": mae_norm,
        "composite": composite,
        "n": int(len(valid)),
        "weights": {"w_r2": w_r2, "w_f1": w_f1, "w_mae": w_mae},
    }


def run_optuna_search(trials: int, anchors: int, min_test_date: str, seed: int,
                      w_r2: float, w_f1: float, w_mae: float) -> Dict:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            "gamma": trial.suggest_float("gamma", 0.0, 0.4),
            "tree_method": "hist",
        }

        metrics = evaluate_params_regression(params, anchors, min_test_date, w_r2, w_f1, w_mae)
        if "composite" not in metrics:
            return float("-inf")

        trial.set_user_attr("metrics", metrics)
        return metrics["composite"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials)

    best_trial = study.best_trial
    best_params = best_trial.params
    best_metrics = best_trial.user_attrs.get("metrics", {})

    result = {
        "task": "regression",
        "objective": "(w_r2*r2 + w_f1*f1 - w_mae*mae_norm)",
        "best_params": best_params,
        "best_metrics": best_metrics,
        "trials": trials,
        "anchors": anchors,
        "min_test_date": min_test_date,
        "random_seed": seed,
    }

    out_dir = Path("cache") / "hyperparams"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "regression_xgboost_optuna_best.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)

    print(f"[INFO] Saved Optuna best params to {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Optuna search for XGBoost hyperparameters")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--anchors", type=int, default=120)
    parser.add_argument("--min-test-date", dest="min_test_date", default="2008-01-01")
    parser.add_argument("--seed", type=int, default=getattr(config, "RANDOM_SEED", 42))
    parser.add_argument("--w-r2", type=float, default=0.50)
    parser.add_argument("--w-f1", type=float, default=0.35)
    parser.add_argument("--w-mae", type=float, default=0.15)
    args = parser.parse_args()

    run_optuna_search(
        trials=args.trials,
        anchors=args.anchors,
        min_test_date=args.min_test_date,
        seed=args.seed,
        w_r2=args.w_r2,
        w_f1=args.w_f1,
        w_mae=args.w_mae,
    )


if __name__ == "__main__":
    main()
