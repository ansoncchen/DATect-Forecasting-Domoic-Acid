#!/usr/bin/env python3
"""
Merge partial_ablation/<name>.json files into paper_ablation_results.json.

Run after the run_one_ablation.sbatch array completes:
  python scripts/eval/merge_ablation_partials.py
  # -> writes paper_ablation_results.json with all ablations keyed by name
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ABLATION_ORDER = [
    "baseline",
    "no_interpolated_training",
    "no_per_site_customization",
    "no_observation_order_lags",
    "no_derived_features",
    "no_oad_features",
    "with_oad_on_small_n",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=str, default="partial_ablation")
    p.add_argument("--out", type=str, default="paper_ablation_results.json")
    args = p.parse_args()

    out = {}
    missing = []
    for name in ABLATION_ORDER:
        f = Path(args.in_dir) / f"{name}.json"
        if not f.exists():
            missing.append(name)
            continue
        data = json.loads(f.read_text())
        if "error" in data:
            print(f"  {name}: ERROR — {data['error']}")
            out[name] = None
        else:
            ov = data.get("overall", {})
            print(f"  {name}: R²={ov.get('r2', '?'):.4f}, MAE={ov.get('mae', '?'):.2f}, N={ov.get('n', '?')}")
            out[name] = data

    if missing:
        print(f"\nMissing ablations: {missing}")

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}  ({len(out)}/{len(ABLATION_ORDER)} ablations present)")

    base = out.get("baseline")
    if base:
        base_r2 = base["overall"]["r2"]
        print(f"\n{'Configuration':<35} {'R2':>8} {'dR2':>8} {'MAE':>8}")
        print("-" * 60)
        for name in ABLATION_ORDER:
            r = out.get(name)
            if r is None:
                print(f"{name:<35} {'MISSING':>8}")
                continue
            r2 = r["overall"]["r2"]
            mae = r["overall"]["mae"]
            d = "---" if name == "baseline" else f"{r2 - base_r2:+.4f}"
            print(f"{name:<35} {r2:>8.4f} {d:>8} {mae:>8.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
