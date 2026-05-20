#!/usr/bin/env python3
"""
Per-site hyperparameter tuning via Optuna (Task 12).

Searches XGBoost params, RF params, and ensemble weights for ONE site,
maximising per-site retrospective R² on raw DA. Designed to run as a Slurm
array job (one site per array task) so all 10 sites tune in parallel.

Mechanism:
  - Each trial writes a temporary JSON file with the candidate hyperparams.
  - Sets DATECT_HPARAM_OVERRIDE_JSON env var to that path.
  - Invokes a fresh subprocess that imports the engine (which re-reads
    per_site_models.py and applies the override via the new env-var hook).
  - Subprocess runs retrospective forecast at THIS site only and prints R².
  - Optuna's TPE sampler proposes next trial.

Best params per site are saved to:
  tuning_results/<site_slug>/best_params.json
  tuning_results/<site_slug>/study.db        (sqlite for resumability)

Usage (Hyak array job):
  sbatch --array=0-9 scripts/eval/tune_per_site_hyperparams.sbatch

Usage (single site, local smoke test):
  python scripts/eval/tune_per_site_hyperparams.py --site "Twin Harbors" --n-trials 3
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    from _repo import ensure_repo_root  # noqa
    ensure_repo_root()
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# Site indices for Slurm array tasks (--array=0-9 maps to these)
SITES = [
    "Copalis", "Kalaloch", "Twin Harbors", "Quinault", "Long Beach",
    "Clatsop Beach", "Coos Bay", "Cannon Beach", "Gold Beach", "Newport",
]


SUBPROCESS_SCRIPT = '''
import warnings; warnings.filterwarnings("ignore")
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

import config
from forecasting.raw_forecast_engine import RawForecastEngine

site = os.environ["TUNE_SITE"]
# THREE-WINDOW SPLIT:
#   TRAIN:    pre-anchor data (engine-enforced, anchor = test_date - 7)
#   VAL:      [TUNE_VAL_START, TUNE_VAL_END)  ← Optuna's objective
#   HOLDOUT:  [TUNE_VAL_END, ...]             ← reserved, never seen by tuning
val_start = pd.Timestamp(os.environ.get("TUNE_VAL_START", "2019-01-01"))
val_end   = pd.Timestamp(os.environ.get("TUNE_VAL_END",   "2022-01-01"))
engine = RawForecastEngine(validate_on_init=False)
# Restrict min_test_date to val_start so engine only computes forecasts in
# the validation window (we discard everything outside it anyway). Saves
# ~60% of compute per trial vs the default 2008-01-01.
results_df = engine.run_retrospective_evaluation(
    task="regression", model_type="ensemble",
    n_anchors=getattr(config, "N_RANDOM_ANCHORS", 500),
    min_test_date=val_start.strftime("%Y-%m-%d"),
)
if results_df is None or results_df.empty:
    print(json.dumps({"r2": -999.0, "mae": 999.0, "n": 0}))
    sys.exit(0)
# Filter to VALIDATION WINDOW: this site, [val_start, val_end)
results_df["date"] = pd.to_datetime(results_df["date"])
sub = results_df[(results_df["site"] == site)
                 & (results_df["date"] >= val_start)
                 & (results_df["date"] <  val_end)]
if len(sub) < 10:
    print(json.dumps({"r2": -999.0, "mae": 999.0, "n": len(sub)}))
    sys.exit(0)
print(json.dumps({
    "r2": float(r2_score(sub["actual_da"], sub["predicted_da"])),
    "mae": float(mean_absolute_error(sub["actual_da"], sub["predicted_da"])),
    "n": int(len(sub)),
    "val_window": [val_start.strftime("%Y-%m-%d"), val_end.strftime("%Y-%m-%d")],
}))
'''


def slug(site: str) -> str:
    return site.lower().replace(" ", "_")


def evaluate_trial(site: str, params: dict, timeout: int = 2700) -> dict:
    """Write params to JSON, run subprocess, return {r2, mae, n}."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({site: params}, f)
        json_path = f.name
    try:
        env = os.environ.copy()
        env["DATECT_HPARAM_OVERRIDE_JSON"] = json_path
        env["TUNE_SITE"] = site
        result = subprocess.run(
            [sys.executable, "-c", SUBPROCESS_SCRIPT],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=timeout,
        )
        if result.returncode != 0:
            return {"r2": -999.0, "mae": 999.0, "n": 0}
        # Extract last JSON line from stdout
        for line in reversed(result.stdout.strip().split("\n")):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        return {"r2": -999.0, "mae": 999.0, "n": 0}
    finally:
        os.unlink(json_path)


def make_objective(site: str):
    import optuna

    def objective(trial: "optuna.Trial") -> float:
        xgb_params = {
            "max_depth": trial.suggest_int("xgb_max_depth", 2, 5),
            "n_estimators": trial.suggest_int("xgb_n_estimators", 50, 300, step=50),
            "learning_rate": trial.suggest_float("xgb_lr", 0.01, 0.1, log=True),
            "min_child_weight": trial.suggest_int("xgb_mcw", 5, 20),
            "reg_alpha": trial.suggest_float("xgb_alpha", 0.1, 3.0, log=True),
            "reg_lambda": trial.suggest_float("xgb_lambda", 1.0, 15.0, log=True),
            "gamma": trial.suggest_float("xgb_gamma", 0.1, 3.0, log=True),
            "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("xgb_colsample", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("xgb_colsample_bylevel", 0.5, 1.0),
        }
        rf_params = {
            "n_estimators": trial.suggest_int("rf_n_estimators", 100, 500, step=100),
            "max_depth": trial.suggest_int("rf_max_depth", 4, 12),
            "min_samples_split": trial.suggest_int("rf_min_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("rf_min_leaf", 1, 10),
            "max_features": trial.suggest_float("rf_max_features", 0.3, 1.0),
        }
        w_xgb = trial.suggest_float("w_xgb", 0.0, 1.0)
        clip_q = trial.suggest_float("clip_q", 0.90, 0.99)
        # clip_max: hard ceiling — None or 60–120 µg/g (sites Kalaloch & Cannon Beach
        # currently use 80.0). Treat as categorical so Optuna can pick "no ceiling".
        clip_max_choice = trial.suggest_categorical("clip_max", ["none", 60.0, 80.0, 100.0, 120.0])
        clip_max = None if clip_max_choice == "none" else float(clip_max_choice)

        # Single-grid override (skip nested per-anchor tuning during search)
        param_grid = [{
            "max_depth": xgb_params["max_depth"],
            "n_estimators": xgb_params["n_estimators"],
            "learning_rate": xgb_params["learning_rate"],
            "min_child_weight": xgb_params["min_child_weight"],
        }]

        params = {
            "xgb_params": xgb_params,
            "rf_params": rf_params,
            "param_grid": param_grid,
            "ensemble_weights": [w_xgb, 1.0 - w_xgb, 0.0],
            "prediction_clip_q": clip_q,
            "prediction_clip_max": clip_max,
        }
        result = evaluate_trial(site, params)
        trial.set_user_attr("mae", result["mae"])
        trial.set_user_attr("n", result["n"])
        return result["r2"]

    return objective


def tune_site(site: str, n_trials: int, output_root: Path) -> None:
    import optuna
    out_dir = output_root / slug(site)
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{out_dir}/study.db"

    study = optuna.create_study(
        study_name=f"tune_{slug(site)}",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    print(f"[{site}] starting {n_trials} trials (existing: {len(study.trials)})")
    t0 = time.time()
    study.optimize(make_objective(site), n_trials=n_trials, show_progress_bar=False)
    elapsed = time.time() - t0

    best = study.best_trial
    print(f"[{site}] DONE in {elapsed/60:.1f} min  best R²={best.value:.4f}  "
          f"MAE={best.user_attrs.get('mae', '?')}  N={best.user_attrs.get('n', '?')}")

    (out_dir / "best_params.json").write_text(json.dumps({
        "site": site,
        "best_value_r2": best.value,
        "best_mae": best.user_attrs.get("mae"),
        "n_test_points": best.user_attrs.get("n"),
        "n_trials_completed": len(study.trials),
        "best_params": best.params,
        "elapsed_seconds": elapsed,
    }, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-site hyperparameter tuning (Task 12)")
    parser.add_argument("--site", type=str, default=None,
                        help="Site to tune. If omitted, uses $SLURM_ARRAY_TASK_ID to pick from SITES.")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Optuna trials per site (default: 30)")
    parser.add_argument("--output-root", type=str, default="tuning_results",
                        help="Output directory (default: tuning_results/)")
    args = parser.parse_args()

    if args.site is None:
        idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        if idx >= len(SITES):
            print(f"SLURM_ARRAY_TASK_ID={idx} out of range")
            return 1
        args.site = SITES[idx]

    tune_site(args.site, args.n_trials, Path(args.output_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
