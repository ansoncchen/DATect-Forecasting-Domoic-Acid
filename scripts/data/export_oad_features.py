#!/usr/bin/env python3
"""Publish one OAD checkpoint's regional scores as DATect's OAD input.

The OAD subproject (oad/) writes per-checkpoint regional anomaly scores to
oad/outputs/scores/<checkpoint>.parquet (generated on Hyak, not tracked). DATect
reads a single tracked copy, data/processed/oad_scores.parquet, which
dataset-creation.py / rebuild_final_output.py turn into the oad_* features.
This script selects the checkpoint, validates the schema and writes that copy.

The companion cloud-fraction input, data/processed/oad_cloud_fractions.parquet,
comes from the cube itself (compute_region_valid_fractions in
oad/scripts/07_cloud_confound.py) and does not depend on the checkpoint.

Usage:
    python scripts/data/export_oad_features.py ae_3d_l32_c4_t4_s42_mae070
    python scripts/data/export_oad_features.py <checkpoint> --scores-dir /gscratch/.../outputs/scores
Then refresh the features:
    python scripts/data/rebuild_final_output.py --recompute oad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data" / "processed" / "oad_scores.parquet"
COLUMNS = ["date", "region", "method", "aggregation", "score", "checkpoint"]


def load_scores(scores_dir: Path, checkpoint: str) -> pd.DataFrame:
    sys.path.insert(0, str(ROOT))
    from forecasting.oad_features import SITE_TO_REGION

    df = pd.read_parquet(scores_dir / f"{checkpoint}.parquet")
    if list(df.columns) != COLUMNS:
        raise ValueError(f"unexpected columns {list(df.columns)}; expected {COLUMNS}")
    if set(df["aggregation"]) != {"mean"} or df["method"].nunique() != 1:
        raise ValueError("expected a single method with aggregation='mean'")
    missing = set(SITE_TO_REGION.values()) - set(df["region"])
    if missing:
        raise ValueError(f"regions used by DATect sites are missing: {sorted(missing)}")
    if df.duplicated(["date", "region"]).any():
        raise ValueError("duplicate (date, region) rows")
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", help="score file stem, e.g. ae_3d_l32_c4_t4_s42_mae070")
    parser.add_argument("--scores-dir", type=Path, default=ROOT / "oad" / "outputs" / "scores")
    args = parser.parse_args()

    df = load_scores(args.scores_dir, args.checkpoint)
    df.to_parquet(OUTPUT, index=False)
    print(f"wrote {OUTPUT} from {args.checkpoint}: {len(df)} rows, "
          f"{df['region'].nunique()} regions, {df['date'].min()} to {df['date'].max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
