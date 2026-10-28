#!/usr/bin/env python3
"""Compare evaluation outputs from two golden-baseline runs.

Behavior-neutral commits must reproduce a golden run exactly. This script
checks every prediction, table and metrics file the golden job writes, and
for files that differ prints the headline-metric deltas so behavior-changing
commits can quote them.

Usage:
    python scripts/eval/compare_golden.py <baseline_dir> <candidate_dir>

Exit code 0 means every file is identical.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

GOLDEN_GLOBS = [
    "eval_outputs/chronological/chronological_regression_ensemble_*.parquet",
    "eval_outputs/chronological/chronological_regression_ensemble_*.json",
    "golden_multi_seed/*.parquet",
    "eval_results/retro_seed123/*.parquet",
    "eval_results/paper_metrics/*.csv",
    "eval_results/paper_metrics/paper_metrics.json",
    "eval_results/spike_metrics_seed123/*.csv",
]


def _collect(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for g in GOLDEN_GLOBS for p in root.glob(g)}


def _frame_diff(a: pd.DataFrame, b: pd.DataFrame) -> str | None:
    try:
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    except AssertionError as exc:
        return str(exc).splitlines()[0]
    return None


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _chronological_summary(path: Path) -> dict:
    overall = json.loads(path.read_text())["overall"]
    return {k: (v[0] if isinstance(v, list) else v) for k, v in overall.items()}


def compare(base: Path, cand: Path) -> int:
    base_files, cand_files = _collect(base), _collect(cand)
    failures = identical = 0

    for missing in sorted(base_files ^ cand_files):
        side = "candidate" if missing in base_files else "baseline"
        print(f"MISSING in {side}: {missing}")
        failures += 1

    for rel in sorted(base_files & cand_files):
        a, b = base / rel, cand / rel
        if rel.endswith(".json"):
            same = json.loads(a.read_text()) == json.loads(b.read_text())
            reason = None if same else "JSON content differs"
        else:
            reason = _frame_diff(_read(a), _read(b))
        if reason is None:
            identical += 1
            print(f"identical  {rel}")
            continue
        failures += 1
        print(f"DIFFERENT  {rel}: {reason}")
        if "chronological" in rel and rel.endswith(".json"):
            sa, sb = _chronological_summary(a), _chronological_summary(b)
            for key in sorted(set(sa) | set(sb)):
                va, vb = sa.get(key), sb.get(key)
                if va != vb:
                    print(f"    {key}: {va} -> {vb}")

    print(f"\n{identical} identical, {failures} differing or missing")
    return 0 if failures == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("baseline", type=Path)
    p.add_argument("candidate", type=Path)
    args = p.parse_args()
    return compare(args.baseline, args.candidate)


if __name__ == "__main__":
    sys.exit(main())
