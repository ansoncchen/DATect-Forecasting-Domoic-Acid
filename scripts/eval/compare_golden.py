#!/usr/bin/env python3
"""Compare evaluation outputs from two golden-baseline runs.

Behavior-neutral commits must reproduce a golden run exactly. This script
checks every prediction, table and metrics file the golden job writes, and
for files that differ prints the headline-metric deltas so behavior-changing
commits can quote them.

Prediction parquets must match exactly. Metric tables (CSV/JSON) may also
match within a relative tolerance of 1e-12, reported as rounding: they are
derived numbers, and algebraically identical formulas (e.g. F2 from counts vs
from precision and recall) can differ in the last bit.

Usage:
    python scripts/eval/compare_golden.py <baseline_dir> <candidate_dir>

Exit code 0 means every file is identical (or, for metric tables, equal within rounding).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

GOLDEN_GLOBS = [
    "results/chronological/chronological_regression_ensemble_*.parquet",
    "results/chronological/chronological_regression_ensemble_*.json",
    "golden_multi_seed/*.parquet",
    "results/retro_seed123/*.parquet",
    "results/paper_metrics/*.csv",
    "results/paper_metrics/paper_metrics.json",
    "results/spike_metrics_seed123/*.csv",
]

# Golden runs of commits before the results/ consolidation wrote these paths
# (legacy prefix -> current prefix; first match wins, so order matters).
LEGACY_PREFIXES = {
    "eval_outputs/chronological/": "results/chronological/",
    "eval_results/": "results/",
}


def _to_legacy(rel: str) -> str:
    for old, new in LEGACY_PREFIXES.items():
        if rel.startswith(new):
            return old + rel[len(new):]
    return rel


def _to_current(rel: str) -> str:
    for old, new in LEGACY_PREFIXES.items():
        if rel.startswith(old):
            return new + rel[len(old):]
    return rel


def _collect(root: Path) -> dict[str, Path]:
    """Map current-layout relative paths to files under ``root`` (either layout)."""
    found = {}
    for pattern in GOLDEN_GLOBS:
        for g in {pattern, _to_legacy(pattern)}:
            for p in root.glob(g):
                found[_to_current(str(p.relative_to(root)))] = p
    return found


METRIC_RTOL = 1e-12


def _frame_diff(a: pd.DataFrame, b: pd.DataFrame, exact: bool) -> str | None:
    try:
        if exact:
            pd.testing.assert_frame_equal(a, b, check_exact=True)
        else:
            pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=METRIC_RTOL, atol=0)
    except AssertionError as exc:
        return str(exc).splitlines()[0]
    return None


def _json_close(x, y) -> bool:
    if isinstance(x, dict) and isinstance(y, dict):
        return x.keys() == y.keys() and all(_json_close(x[k], y[k]) for k in x)
    if isinstance(x, list) and isinstance(y, list):
        return len(x) == len(y) and all(_json_close(i, j) for i, j in zip(x, y))
    if isinstance(x, float) and isinstance(y, float):
        return x == y or abs(x - y) <= METRIC_RTOL * max(abs(x), abs(y))
    return x == y


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _chronological_summary(path: Path) -> dict:
    overall = json.loads(path.read_text())["overall"]
    return {k: (v[0] if isinstance(v, list) else v) for k, v in overall.items()}


def compare(base: Path, cand: Path) -> int:
    base_files, cand_files = _collect(base), _collect(cand)
    failures = identical = rounding = 0

    for missing in sorted(base_files.keys() ^ cand_files.keys()):
        side = "candidate" if missing in base_files else "baseline"
        print(f"MISSING in {side}: {missing}")
        failures += 1

    for rel in sorted(base_files.keys() & cand_files.keys()):
        a, b = base_files[rel], cand_files[rel]
        if rel.endswith(".json"):
            ja, jb = json.loads(a.read_text()), json.loads(b.read_text())
            exact = ja == jb
            reason = None if _json_close(ja, jb) else "JSON content differs"
        else:
            fa, fb = _read(a), _read(b)
            reason = _frame_diff(fa, fb, exact=True)
            exact = reason is None
            if not exact and not rel.endswith(".parquet"):  # predictions get no tolerance
                reason = _frame_diff(fa, fb, exact=False)
        if reason is None:
            if exact:
                identical += 1
                print(f"identical  {rel}")
            else:
                rounding += 1
                print(f"rounding   {rel} (equal within rtol={METRIC_RTOL:g})")
            continue
        failures += 1
        print(f"DIFFERENT  {rel}: {reason}")
        if "chronological" in rel and rel.endswith(".json"):
            sa, sb = _chronological_summary(a), _chronological_summary(b)
            for key in sorted(set(sa) | set(sb)):
                va, vb = sa.get(key), sb.get(key)
                if va != vb:
                    print(f"    {key}: {va} -> {vb}")

    print(f"\n{identical} identical, {rounding} equal within rounding, {failures} differing or missing")
    return 0 if failures == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("baseline", type=Path)
    p.add_argument("candidate", type=Path)
    args = p.parse_args()
    return compare(args.baseline, args.candidate)


if __name__ == "__main__":
    sys.exit(main())
