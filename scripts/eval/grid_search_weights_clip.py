#!/usr/bin/env python3
"""
Constrained grid search over the THREE high-leverage hyperparameters per site:
  - ensemble_weights (XGB vs RF blend): w_xgb ∈ {0.0, 0.25, 0.5, 0.75, 1.0}
  - prediction_clip_q:                 ∈ {0.95, 0.97, 0.99}
  - prediction_clip_max:               ∈ {None, 60, 80, 100, 120}

Why these three? The prior stability study showed |ΔR²| < 0.001 from RF
hyperparameter perturbations and similarly tiny effects from XGB regularization
moves. The dimensions that ACTUALLY moved the score were:
  - ensemble_weights swap (ΔR² = -0.028 when forced to wrong model)
  - clipping relaxation  (ΔR² = -0.030 when clip removed)
  - per-site customization (ΔR² = -0.103 when removed entirely)

Optuna's full 18-dim search overfit catastrophically (val +0.072, holdout
-0.157, see §18 of OAD_INTEGRATION_RESULTS.md). This grid search uses just
the 3 demonstrably-impactful dims, eliminating most of the overfitting
surface area.

CHRONOLOGICAL 3-FOLD CV within the val window:
  Fold A: train ≤ 2018-12-31, test 2019-01-01 to 2019-12-31
  Fold B: train ≤ 2019-12-31, test 2020-01-01 to 2020-12-31
  Fold C: train ≤ 2020-12-31, test 2021-01-01 to 2021-12-31

Each config's score = mean_R² - 0.5*std_R² across the 3 folds (penalizes
high-variance configs). The HOLDOUT 2022-2024 window is NEVER touched here.

Compute estimate: 75 configs × 3 folds × 10 sites ≈ 2250 evals × ~30s each
≈ 18 hr sequential, ~2 hr if run as Slurm array (10 sites × 75/3 = 25 evals
per task, parallel).

Usage:
  python scripts/eval/grid_search_weights_clip.py --site "Twin Harbors"
  # Or via slurm array:
  sbatch scripts/eval/grid_search_weights_clip.sbatch  # --array=0-9
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    from _repo import ensure_repo_root
    ensure_repo_root()
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


SITES = [
    "Copalis", "Kalaloch", "Twin Harbors", "Quinault", "Long Beach",
    "Clatsop Beach", "Coos Bay", "Cannon Beach", "Gold Beach", "Newport",
]

# The 3-dim grid — DELIBERATELY small to limit overfitting room
W_XGB_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
CLIP_Q_GRID = [0.95, 0.97, 0.99]
CLIP_MAX_GRID = [None, 60.0, 80.0, 100.0, 120.0]

# Chronological CV folds. Derived from config/tuned_hyperparameters.json
# eval_windows.validation_start..validation_end (default: [2019, 2022)),
# split into yearly sub-folds. Holdout window stays untouched by construction.
def _build_folds():
    from forecasting.tuned_config import get_eval_windows
    w = get_eval_windows()
    start_year = int(w["validation_start"].split("-")[0])
    end_year   = int(w["validation_end"].split("-")[0])
    return [(f"{y}-01-01", f"{y+1}-01-01") for y in range(start_year, end_year)]

FOLDS = _build_folds()


SUBPROCESS_SCRIPT = '''
import warnings; warnings.filterwarnings("ignore")
import os, sys, json
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

import config
from forecasting.raw_forecast_engine import RawForecastEngine

site = os.environ["GRID_SITE"]
fold_start = pd.Timestamp(os.environ["FOLD_START"])
fold_end   = pd.Timestamp(os.environ["FOLD_END"])

engine = RawForecastEngine(validate_on_init=False)
# Only evaluate within this fold's test window — saves compute by skipping
# years we don't score. (Inner per-anchor tuning stays at whatever the env says.)
results_df = engine.run_retrospective_evaluation(
    task="regression", model_type="ensemble",
    n_anchors=getattr(config, "N_RANDOM_ANCHORS", 500),
    min_test_date=fold_start.strftime("%Y-%m-%d"),
)
if results_df is None or results_df.empty:
    print(json.dumps({"r2": -999.0, "mae": 999.0, "n": 0}))
    sys.exit(0)
results_df["date"] = pd.to_datetime(results_df["date"])
sub = results_df[
    (results_df["site"] == site)
    & (results_df["date"] >= fold_start)
    & (results_df["date"] <  fold_end)
]
if len(sub) < 5:
    print(json.dumps({"r2": -999.0, "mae": 999.0, "n": len(sub)}))
    sys.exit(0)
print(json.dumps({
    "r2":  float(r2_score(sub["actual_da"], sub["predicted_da"])),
    "mae": float(mean_absolute_error(sub["actual_da"], sub["predicted_da"])),
    "n":   int(len(sub)),
}))
'''


def evaluate_config_fold(site: str, params: dict, fold_start: str, fold_end: str,
                          timeout: int = 1800) -> dict:
    """Run the engine once with the given config on one chronological fold."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({site: params}, f)
        json_path = f.name
    try:
        env = os.environ.copy()
        env["DATECT_HPARAM_OVERRIDE_JSON"] = json_path
        env["DATECT_MIN_TRAINING_FOR_TUNING"] = "99999"  # skip inner tuning
        env["GRID_SITE"] = site
        env["FOLD_START"] = fold_start
        env["FOLD_END"] = fold_end
        result = subprocess.run(
            [sys.executable, "-c", SUBPROCESS_SCRIPT],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=timeout,
        )
        if result.returncode != 0:
            return {"r2": -999.0, "mae": 999.0, "n": 0}
        for line in reversed(result.stdout.strip().split("\n")):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        return {"r2": -999.0, "mae": 999.0, "n": 0}
    finally:
        os.unlink(json_path)


def grid_search_one_site(site: str, out_dir: Path) -> None:
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{site.replace(' ', '_').lower()}_grid.json"

    # Resume from existing JSON if present
    completed = {}
    if log_path.exists():
        prev = json.loads(log_path.read_text())
        completed = {tuple(k.split("|")): v for k, v in prev.get("evals", {}).items()}
        print(f"[{site}] Resuming with {len(completed)} prior evals.", flush=True)

    grid = list(itertools.product(W_XGB_GRID, CLIP_Q_GRID, CLIP_MAX_GRID))
    print(f"[{site}] grid size: {len(grid)} configs × {len(FOLDS)} folds "
          f"= {len(grid) * len(FOLDS)} total evals", flush=True)
    t0 = time.time()

    evals = {}
    for i, (w_xgb, clip_q, clip_max) in enumerate(grid):
        params = {
            "ensemble_weights": [w_xgb, 1.0 - w_xgb, 0.0],
            "prediction_clip_q": clip_q,
            "prediction_clip_max": clip_max,
        }
        fold_r2s = []
        fold_ns = []
        for fold_start, fold_end in FOLDS:
            key = (f"{w_xgb}", f"{clip_q}", f"{clip_max}", fold_start)
            if key in completed:
                res = completed[key]
            else:
                res = evaluate_config_fold(site, params, fold_start, fold_end)
                completed[key] = res
                # Persist intermediate results
                serializable = {"|".join(k): v for k, v in completed.items()}
                log_path.write_text(json.dumps({"site": site, "evals": serializable}, indent=2))
            fold_r2s.append(res["r2"])
            fold_ns.append(res["n"])
        cfg_key = f"w_xgb={w_xgb}|clip_q={clip_q}|clip_max={clip_max}"
        evals[cfg_key] = {
            "fold_r2": fold_r2s,
            "fold_n":  fold_ns,
            "mean_r2": float(np.mean([r for r in fold_r2s if r > -100])),
            "std_r2":  float(np.std([r for r in fold_r2s if r > -100])),
            "robust_score": float(
                np.mean([r for r in fold_r2s if r > -100])
                - 0.5 * np.std([r for r in fold_r2s if r > -100])
            ),
        }
        if (i + 1) % 5 == 0 or i == 0:
            print(f"[{site}] config {i+1}/{len(grid)} done, elapsed {(time.time()-t0)/60:.1f} min", flush=True)

    # Select winner by robust_score
    ranked = sorted(evals.items(), key=lambda kv: -kv[1]["robust_score"])
    winner_key, winner_stats = ranked[0]

    summary = {
        "site": site,
        "n_configs_evaluated": len(evals),
        "winner_config": winner_key,
        "winner_stats": winner_stats,
        "top_5": [
            {"config": k, **v} for k, v in ranked[:5]
        ],
        "elapsed_seconds": time.time() - t0,
        "fold_definitions": [{"start": s, "end": e} for s, e in FOLDS],
    }
    (out_dir / f"{site.replace(' ', '_').lower()}_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"\n[{site}] WINNER: {winner_key}", flush=True)
    print(f"  mean R²={winner_stats['mean_r2']:.4f}  "
          f"std={winner_stats['std_r2']:.4f}  "
          f"robust_score={winner_stats['robust_score']:.4f}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--site", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="grid_search_results")
    args = p.parse_args()
    if args.site is None:
        idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        if idx >= len(SITES):
            print(f"SLURM_ARRAY_TASK_ID={idx} out of range")
            return 1
        args.site = SITES[idx]
    grid_search_one_site(args.site, Path(args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
