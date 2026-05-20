#!/usr/bin/env python3
"""
Run ONE ablation experiment as a standalone subprocess and save its result
to partial_ablation/<name>.json. Designed for Slurm array jobs where each
array task runs one ablation (preemption-resistant: max ~30 min per task).

The 6 ablations and their env vars match paper_ablation_study.py exactly.

Usage:
  python scripts/eval/run_one_ablation.py --index 0   # baseline
  python scripts/eval/run_one_ablation.py --index 5   # no_oad_features
  # OR via slurm array:
  python scripts/eval/run_one_ablation.py  # uses $SLURM_ARRAY_TASK_ID
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

# Import the SUBPROCESS_SCRIPT and constants from paper_ablation_study.py
from scripts.eval.paper_ablation_study import (
    SUBPROCESS_SCRIPT, DERIVED_FEATURES, OAD_FEATURES,
)


# Same definitions as paper_ablation_study.py main(), keyed by stable name
ABLATIONS = [
    ("baseline", {}),
    ("no_interpolated_training", {"DATECT_USE_INTERPOLATED_TRAINING": "false"}),
    ("no_per_site_customization", {"DATECT_USE_PER_SITE_MODELS": "false"}),
    ("no_observation_order_lags", {"DATECT_LAG_FEATURES": "none"}),
    ("no_derived_features", {"DATECT_EXTRA_DROP_FEATURES": ",".join(DERIVED_FEATURES)}),
    ("no_oad_features", {"DATECT_EXTRA_DROP_FEATURES": ",".join(OAD_FEATURES)}),
    ("with_oad_on_small_n", {"DATECT_OAD_ON_SMALL_N": "true"}),
]


def run_one(name: str, env_overrides: dict, seed: int, out_dir: Path) -> dict:
    print(f"\n{'='*60}\nABLATION: {name}  seed={seed}\n{'='*60}", flush=True)
    for k, v in env_overrides.items():
        print(f"  ENV: {k}={v}", flush=True)

    env = os.environ.copy()
    env.update(env_overrides)
    env["DATECT_ABLATION_SEED"] = str(seed)

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "-c", SUBPROCESS_SCRIPT],
        env=env, stdout=subprocess.PIPE, stderr=None,
        text=True, timeout=7200,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode}) after {elapsed/60:.1f} min", flush=True)
        out = {"name": name, "error": f"exit {result.returncode}", "elapsed_seconds": elapsed}
    else:
        lines = result.stdout.strip().split("\n")
        json_str = None
        for i, line in enumerate(lines):
            if line.strip().startswith("{"):
                json_str = "\n".join(lines[i:])
                break
        if json_str is None:
            print(f"  ERROR: no JSON in output (elapsed {elapsed/60:.1f} min)", flush=True)
            out = {"name": name, "error": "no_json", "stdout_tail": result.stdout[-1000:]}
        else:
            data = json.loads(json_str)
            data["name"] = name
            data["elapsed_seconds"] = elapsed
            ov = data["overall"]
            print(f"  Overall: R²={ov['r2']:.4f}, MAE={ov['mae']:.2f}, N={ov['n']}  "
                  f"({elapsed/60:.1f} min)", flush=True)
            out = data

    out_path = out_dir / f"{name}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  Wrote {out_path}", flush=True)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=int, default=None,
                   help="Ablation index (0-6). If omitted, uses $SLURM_ARRAY_TASK_ID")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out-dir", type=str, default="partial_ablation")
    args = p.parse_args()

    if args.index is None:
        args.index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if args.index < 0 or args.index >= len(ABLATIONS):
        print(f"--index {args.index} out of range [0, {len(ABLATIONS)})")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name, env_over = ABLATIONS[args.index]
    run_one(name, env_over, args.seed, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
