#!/usr/bin/env python3
"""
Multi-seed bootstrap baseline R²/MAE — get genuine CIs for the headline metrics.

The single-seed R² values currently quoted (0.17 to 0.49 across runs) span 0.32
R² units, far exceeding any tuning effect we've measured. This is the wrong
unit for scientific reporting. Run the standard retrospective with 5 seeds and
compute bootstrap CIs.

Each task = one seed × one retrospective evaluation, saved to a per-seed parquet.
Aggregator (separate, run locally) computes bootstrap CIs from the 5 parquets.

Usage:
  python scripts/eval/multi_seed_baseline.py --seed 42
  # Or via slurm array (--array=0-4 → seeds 42, 43, 44, 45, 46):
  sbatch scripts/eval/multi_seed_baseline.sbatch
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from _repo import ensure_repo_root
    ensure_repo_root()
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


SUBPROCESS_SCRIPT = '''
import warnings; warnings.filterwarnings("ignore")
import os, sys
import pandas as pd

import config
seed = int(os.environ["BOOTSTRAP_SEED"])
config.RANDOM_SEED = seed

from forecasting.raw_forecast_engine import RawForecastEngine
engine = RawForecastEngine(validate_on_init=False)
results_df = engine.run_retrospective_evaluation(
    task="regression", model_type="ensemble",
    n_anchors=getattr(config, "N_RANDOM_ANCHORS", 500),
    min_test_date="2008-01-01",
)
if results_df is None or results_df.empty:
    print("ERROR: no results")
    sys.exit(1)
results_df.to_parquet(os.environ["OUT_PATH"], index=False)
print(f"saved {len(results_df)} rows to {os.environ['OUT_PATH']}")
'''


def run_one_seed(seed: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"baseline_seed{seed}_predictions.parquet"
    if out_path.exists():
        print(f"[seed {seed}] already exists: {out_path}")
        return
    print(f"[seed {seed}] starting retrospective...", flush=True)
    t0 = time.time()
    env = os.environ.copy()
    env["BOOTSTRAP_SEED"] = str(seed)
    env["OUT_PATH"] = str(out_path.resolve())
    env["DATECT_MIN_TRAINING_FOR_TUNING"] = "99999"  # match holdout config for consistency
    result = subprocess.run(
        [sys.executable, "-c", SUBPROCESS_SCRIPT],
        env=env, stdout=subprocess.PIPE, stderr=None,
        text=True, timeout=7200,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"[seed {seed}] FAILED ({elapsed/60:.1f} min)")
        return
    print(f"[seed {seed}] done in {elapsed/60:.1f} min")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-dir", type=str, default="multi_seed_results")
    args = p.parse_args()
    if args.seed is None:
        # Array task: seeds 42, 43, 44, 45, 46
        idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        args.seed = 42 + idx
    run_one_seed(args.seed, Path(args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
