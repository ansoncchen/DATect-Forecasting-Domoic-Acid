#!/usr/bin/env python3
"""
Spike classifier hyperparameter tuning (Task 13).

Tunes SPIKE_CLASSIFIER_PARAMS (9 XGB-classifier hyperparameters) to maximise
spike-event recall while keeping precision above a floor. Objective = F2
(recall-weighted) on DA > 20 µg/g events, evaluated against held-out raw DA.

Mechanism mirrors tune_per_site_hyperparams.py:
  - Each trial writes candidate params to a temp JSON.
  - Sets DATECT_SPIKE_CLASSIFIER_JSON env var to that path.
  - Subprocess imports the engine (which re-reads config.py + the new env hook
    that overrides SPIKE_CLASSIFIER_PARAMS), runs retrospective forecast across
    ALL sites (spike events are sparse, so pooled is the right unit).

This is a SINGLE study (not per-site) because spike events are rare (~5-10%
of test points) and per-site sample sizes are too small for stable
classifier tuning. The classifier sees pooled training across all sites.

Outputs:
  spike_tuning_results/study.db
  spike_tuning_results/best_params.json

Usage (Hyak):
  sbatch scripts/eval/tune_spike_classifier.sbatch

Usage (local smoke test):
  python scripts/eval/tune_spike_classifier.py --n-trials 3
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


SUBPROCESS_SCRIPT = '''
import warnings; warnings.filterwarnings("ignore")
import os, json, sys
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score

import config
from forecasting.raw_forecast_engine import RawForecastEngine

# Validation cutoff: tune only on pre-2019 spikes. 2019+ is reserved for
# final holdout reporting.
val_cutoff = pd.Timestamp(os.environ.get("TUNE_VAL_CUTOFF", "2019-01-01"))
engine = RawForecastEngine(validate_on_init=False)
results_df = engine.run_retrospective_evaluation(
    task="regression", model_type="ensemble",
    n_anchors=getattr(config, "N_RANDOM_ANCHORS", 500),
    min_test_date="2008-01-01",
)
if results_df is None or results_df.empty:
    print(json.dumps({"f2": -1.0, "recall": 0.0, "precision": 0.0, "n_spikes": 0}))
    sys.exit(0)

# Filter to tuning window
results_df["date"] = pd.to_datetime(results_df["date"])
results_df = results_df[results_df["date"] < val_cutoff]

# Spike label = actual DA > SPIKE_THRESHOLD; prediction = spike_alert column
spike_thresh = config.SPIKE_THRESHOLD
y_true = (results_df["actual_da"] > spike_thresh).astype(int).values
# Prefer the explicit spike_alert column if available; else fall back to
# predicted_da > SPIKE_REGRESSION_ALERT_THRESHOLD.
if "spike_alert" in results_df.columns:
    y_pred = results_df["spike_alert"].fillna(0).astype(int).values
else:
    y_pred = (results_df["predicted_da"] > getattr(config, "SPIKE_REGRESSION_ALERT_THRESHOLD", 12.0)).astype(int).values

n_spikes = int(y_true.sum())
if n_spikes < 5:
    print(json.dumps({"f2": -1.0, "recall": 0.0, "precision": 0.0, "n_spikes": n_spikes}))
    sys.exit(0)

precision = float(precision_score(y_true, y_pred, zero_division=0))
recall = float(recall_score(y_true, y_pred, zero_division=0))
f2 = float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0))
print(json.dumps({"f2": f2, "recall": recall, "precision": precision, "n_spikes": n_spikes}))
'''


def evaluate_trial(params: dict, timeout: int = 1800) -> dict:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(params, f)
        json_path = f.name
    try:
        env = os.environ.copy()
        env["DATECT_SPIKE_CLASSIFIER_JSON"] = json_path
        result = subprocess.run(
            [sys.executable, "-c", SUBPROCESS_SCRIPT],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=timeout,
        )
        if result.returncode != 0:
            return {"f2": -1.0, "recall": 0.0, "precision": 0.0, "n_spikes": 0}
        for line in reversed(result.stdout.strip().split("\n")):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        return {"f2": -1.0, "recall": 0.0, "precision": 0.0, "n_spikes": 0}
    finally:
        os.unlink(json_path)


def make_objective():
    import optuna

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 3.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        }
        # Also tune the alert probability threshold (operating point on the PR curve)
        prob_thresh = trial.suggest_float("spike_alert_prob_threshold", 0.05, 0.30)
        params["_spike_alert_prob_threshold"] = prob_thresh

        result = evaluate_trial(params)
        trial.set_user_attr("recall", result["recall"])
        trial.set_user_attr("precision", result["precision"])
        trial.set_user_attr("n_spikes", result["n_spikes"])
        return result["f2"]

    return objective


def main() -> int:
    parser = argparse.ArgumentParser(description="Spike classifier tuning (Task 13)")
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--output-root", type=str, default="spike_tuning_results")
    args = parser.parse_args()

    import optuna

    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{out_dir}/study.db"

    study = optuna.create_study(
        study_name="spike_classifier_tune",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    print(f"Starting {args.n_trials} trials (existing: {len(study.trials)})")
    t0 = time.time()
    study.optimize(make_objective(), n_trials=args.n_trials, show_progress_bar=False)
    elapsed = time.time() - t0

    best = study.best_trial
    print(f"DONE in {elapsed/60:.1f} min  best F2={best.value:.4f}  "
          f"recall={best.user_attrs.get('recall')}  precision={best.user_attrs.get('precision')}")

    (out_dir / "best_params.json").write_text(json.dumps({
        "best_f2": best.value,
        "best_recall": best.user_attrs.get("recall"),
        "best_precision": best.user_attrs.get("precision"),
        "n_spikes": best.user_attrs.get("n_spikes"),
        "n_trials_completed": len(study.trials),
        "best_params": best.params,
        "elapsed_seconds": elapsed,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
