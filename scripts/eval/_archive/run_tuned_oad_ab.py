#!/usr/bin/env python3
"""
Task 14: Post-tuning OAD A/B comparison.

Re-runs the OAD vs no-OAD comparison using the TUNED per-site hyperparameters
(proposed_overrides.json from Task 12). Tests whether OAD's null effect under
hand-tuned hparams persists when hyperparameters are properly optimized.

Each array task runs ONE configuration:
  task 0 = tuned + OAD             (DATECT_HPARAM_OVERRIDE_JSON only)
  task 1 = tuned + no_oad_features (HPARAM override + DATECT_EXTRA_DROP_FEATURES)

Saves to partial_tuned_ab/<name>.json. Both windows (val 2019-2022 and
holdout 2022-2024) are computed by analyze_tuned_oad_ab.py.
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

from scripts.eval.paper_ablation_study import SUBPROCESS_SCRIPT
from forecasting.oad_features import OAD_FEATURES_ALL as OAD_FEATURES


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--overrides", type=str, default="proposed_overrides.json")
    parser.add_argument("--out-dir", type=str, default="partial_tuned_ab")
    args = parser.parse_args()

    if args.index is None:
        args.index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

    overrides_abs = str(Path(args.overrides).resolve())
    if not Path(overrides_abs).exists():
        print(f"FATAL: {overrides_abs} does not exist. Run Task 12 + aggregate first.")
        return 1

    if args.index == 0:
        name = "tuned_plus_oad"
        env_over = {"DATECT_HPARAM_OVERRIDE_JSON": overrides_abs}
    elif args.index == 1:
        name = "tuned_no_oad"
        env_over = {
            "DATECT_HPARAM_OVERRIDE_JSON": overrides_abs,
            "DATECT_EXTRA_DROP_FEATURES": ",".join(OAD_FEATURES),
        }
    else:
        print(f"--index {args.index} not in [0, 1]")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\nTUNED A/B: {name}  seed={args.seed}\n{'='*60}", flush=True)
    for k, v in env_over.items():
        print(f"  ENV: {k}={v}", flush=True)

    env = os.environ.copy()
    env.update(env_over)
    env["DATECT_ABLATION_SEED"] = str(args.seed)

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "-c", SUBPROCESS_SCRIPT],
        env=env, stdout=subprocess.PIPE, stderr=None,
        text=True, timeout=7200,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"FAILED (exit {result.returncode}) after {elapsed/60:.1f} min", flush=True)
        out = {"name": name, "error": f"exit {result.returncode}", "elapsed_seconds": elapsed}
    else:
        lines = result.stdout.strip().split("\n")
        json_str = None
        for i, line in enumerate(lines):
            if line.strip().startswith("{"):
                json_str = "\n".join(lines[i:])
                break
        if json_str is None:
            out = {"name": name, "error": "no_json", "stdout_tail": result.stdout[-1000:]}
        else:
            data = json.loads(json_str)
            data["name"] = name
            data["elapsed_seconds"] = elapsed
            ov = data["overall"]
            print(f"  Overall: R²={ov['r2']:.4f}, MAE={ov['mae']:.2f}, N={ov['n']}  "
                  f"({elapsed/60:.1f} min)", flush=True)
            out = data

    (out_dir / f"{name}.json").write_text(json.dumps(out, indent=2))
    print(f"  Wrote {out_dir / (name + '.json')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
