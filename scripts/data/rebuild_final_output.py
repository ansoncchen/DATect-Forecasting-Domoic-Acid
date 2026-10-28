#!/usr/bin/env python3
"""Rebuild data/processed/final_output.parquet offline, without downloading anything.

dataset-creation.py rebuilds everything from scratch, which re-downloads every
MODIS satellite chunk and the PDO / ONI / BEUTI series from NOAA ERDDAP (slow
and unreliable). This script keeps every environmental column of the existing
parquet exactly as it is (climate indices, streamflow, BEUTI, the 7 satellite
columns) and recomputes only the columns derived from files in this repo, with
the same functions dataset-creation.py uses:

  da, pn   weekly toxin and Pseudo-nitzschia from data/raw/{da,pn}-input
           (process_da / process_pn, then compile_da_pn's biological decay fill)
  oad_*    OAD features from data/processed/oad_scores.parquet and
           oad_cloud_fractions.parquet (add_oad_features)

Run it after changing measurement parsing, the decay fill, or the OAD inputs.
Recompute only what changed: OAD features use np.polyfit, whose last bits vary
across machines (~1e-15), so rebuilding them without reason adds noise.

Usage:
    python scripts/data/rebuild_final_output.py --recompute measurements  # da, pn
    python scripts/data/rebuild_final_output.py --recompute oad           # oad_*
    python scripts/data/rebuild_final_output.py --recompute all --check   # compare only
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def load_dataset_creation():
    """Import dataset-creation.py (hyphenated, so via importlib) without running main()."""
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("dataset_creation", ROOT / "dataset-creation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rebuild(module, existing: pd.DataFrame, recompute: str) -> pd.DataFrame:
    from forecasting.oad_features import OAD_FEATURES_ALL

    oad_cols = [c for c in OAD_FEATURES_ALL if c in existing.columns]
    redo_measurements = recompute in ("measurements", "all")
    redo_oad = recompute in ("oad", "all")
    drop = (["da", "pn"] if redo_measurements else []) + (oad_cols if redo_oad else [])
    data = existing.drop(columns=drop).rename(columns={"date": "Date", "site": "Site"})
    data["Date"] = pd.to_datetime(data["Date"], format="%m/%d/%Y")
    order = data[["Date", "Site"]].copy()

    if redo_measurements:
        da_parquet = module.convert_files_to_parquet(module.da_files)
        pn_parquet = module.convert_files_to_parquet(module.pn_files)
        try:
            da = module.process_da(da_parquet)
            pn = module.process_pn(pn_parquet)
        finally:
            for path in module.generated_parquet_files:
                if os.path.exists(path):
                    os.remove(path)
        data = module.convert_and_fill(module.compile_da_pn(data, da, pn))

    oad_parquet = Path("data/processed/oad_scores.parquet")
    cloud_parquet = Path("data/processed/oad_cloud_fractions.parquet")
    if redo_oad and oad_cols:
        missing = [str(p) for p in (oad_parquet, cloud_parquet) if not p.exists()]
        if missing:
            raise FileNotFoundError(f"OAD inputs missing: {missing}")
        data = module.add_oad_features(data, str(oad_parquet), cloud_parquet_path=str(cloud_parquet))

    data = data.rename(columns={"Date": "date", "Site": "site", "DA_Levels": "da", "PN_Levels": "pn"})
    # Same row order, column order, dtypes and date format as the existing file
    order = pd.MultiIndex.from_frame(order.rename(columns={"Date": "date", "Site": "site"}))
    data = data.set_index(["date", "site"]).reindex(order).reset_index()
    data["date"] = data["date"].dt.strftime("%m/%d/%Y")
    return data[existing.columns].astype(existing.dtypes.to_dict())


def describe_changes(old: pd.DataFrame, new: pd.DataFrame) -> list[str]:
    lines = []
    for col in old.columns:
        a, b = old[col], new[col]
        if a.dtype.kind in "fc":
            changed = ~((a == b) | (a.isna() & b.isna()))
            if changed.any():
                diff = (a - b).abs()
                lines.append(f"  {col:32s} {int(changed.sum()):6d} rows changed  max |diff| {np.nanmax(diff[changed]):.6g}")
        elif not a.equals(b):
            lines.append(f"  {col:32s} {int((a != b).sum()):6d} rows changed")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recompute", choices=["measurements", "oad", "all"], required=True,
                        help="Which derived columns to rebuild (everything else is kept as-is)")
    parser.add_argument("--check", action="store_true",
                        help="Compare the rebuild with the existing file; do not write")
    parser.add_argument("--output", default=None,
                        help="Write here instead of overwriting config.FINAL_OUTPUT_PATH")
    args = parser.parse_args()

    module = load_dataset_creation()
    path = Path(module.final_output_path)
    existing = pd.read_parquet(path)
    rebuilt = rebuild(module, existing, args.recompute)

    changes = describe_changes(existing, rebuilt)
    print(f"\n{path}: {len(existing)} rows x {len(existing.columns)} columns")
    print("\n".join(changes) if changes else "  rebuild is identical to the existing file")

    if args.check:
        return 1 if changes else 0
    out = Path(args.output) if args.output else path
    rebuilt.to_parquet(out, index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
