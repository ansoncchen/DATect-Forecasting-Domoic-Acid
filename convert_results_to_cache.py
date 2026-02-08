#!/usr/bin/env python3
"""
Convert Raw Validation Results to API Cache
============================================

Converts the CSV output from ``validate_on_raw_data.py`` (run on Hyak)
into the JSON + Parquet cache format expected by ``cache_manager.py``.

Usage:
    # After SCP'ing results from Hyak:
    python convert_results_to_cache.py

    # Custom paths:
    python convert_results_to_cache.py \
        --input  raw_validation_plots/raw_data_validation_results.csv \
        --output cache/

Input:
    raw_validation_plots/raw_data_validation_results.csv
    (produced by validate_on_raw_data.py on Hyak)

Output:
    cache/retrospective/regression_ensemble.json + .parquet
    cache/retrospective/regression_xgboost.json + .parquet
    cache/retrospective/regression_rf.json + .parquet
    cache/retrospective/classification_ensemble.json + .parquet
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def clean_float(v):
    """Sanitise a value for JSON serialisation (no inf/nan)."""
    if v is None:
        return None
    if isinstance(v, float):
        if math.isinf(v) or math.isnan(v):
            return None
    return v


def da_to_category(da_value: float) -> int:
    """Map continuous DA to risk category using config bins."""
    if da_value is None or (isinstance(da_value, float) and math.isnan(da_value)):
        return None
    result = pd.cut(
        [da_value],
        bins=config.DA_CATEGORY_BINS,
        labels=config.DA_CATEGORY_LABELS,
    )
    return int(result[0])


def convert(input_csv: str, output_dir: str) -> None:
    """
    Read the raw validation CSV and produce API cache files.
    """
    print(f"Reading: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")

    # Standardise date columns
    for col in ("test_date", "anchor_date", "processed_test_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Ensure output directory exists
    retro_dir = Path(output_dir) / "retrospective"
    retro_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Regression — ensemble
    #    Uses ensemble_prediction as predicted_da
    # ------------------------------------------------------------------
    if "ensemble_prediction" in df.columns:
        _write_cache(
            df,
            pred_col="ensemble_prediction",
            task="regression",
            model="ensemble",
            output_dir=retro_dir,
        )
    else:
        print("  WARNING: 'ensemble_prediction' column not found, skipping regression_ensemble")

    # ------------------------------------------------------------------
    # 2. Regression — xgboost
    #    Uses predicted_da (XGB-only prediction)
    # ------------------------------------------------------------------
    if "predicted_da" in df.columns:
        _write_cache(
            df,
            pred_col="predicted_da",
            task="regression",
            model="xgboost",
            output_dir=retro_dir,
        )

    # ------------------------------------------------------------------
    # 2b. Regression — random forest
    #     Uses predicted_da_rf (RF-only prediction)
    # ------------------------------------------------------------------
    if "predicted_da_rf" in df.columns:
        _write_cache(
            df,
            pred_col="predicted_da_rf",
            task="regression",
            model="rf",
            output_dir=retro_dir,
        )
    else:
        print("  WARNING: 'predicted_da_rf' column not found, skipping regression_rf")

    # ------------------------------------------------------------------
    # 3. Classification — ensemble
    # ------------------------------------------------------------------
    if "ensemble_prediction" in df.columns:
        _write_cache(
            df,
            pred_col="ensemble_prediction",
            task="classification",
            model="ensemble",
            output_dir=retro_dir,
        )

    # ------------------------------------------------------------------
    # 4. Classification — xgboost
    # ------------------------------------------------------------------
    if "predicted_da" in df.columns:
        _write_cache(
            df,
            pred_col="predicted_da",
            task="classification",
            model="xgboost",
            output_dir=retro_dir,
        )

    print(f"\nDone! Cache files written to: {retro_dir}")


def _write_cache(
    df: pd.DataFrame,
    pred_col: str,
    task: str,
    model: str,
    output_dir: Path,
) -> None:
    """Write a single (task, model) cache pair (JSON + Parquet)."""
    records = []
    for _, row in df.iterrows():
        actual_da = clean_float(row.get("actual_da_raw"))
        predicted_da = clean_float(row.get(pred_col))

        record = {
            "date": row["test_date"].strftime("%Y-%m-%d") if pd.notnull(row.get("test_date")) else None,
            "site": row.get("site"),
            "actual_da": actual_da,
            "predicted_da": predicted_da,
            "actual_category": da_to_category(actual_da) if actual_da is not None else None,
            "predicted_category": da_to_category(predicted_da) if predicted_da is not None else None,
        }

        anchor = row.get("anchor_date")
        if anchor is not None and pd.notnull(anchor):
            record["anchor_date"] = pd.Timestamp(anchor).strftime("%Y-%m-%d")

        records.append(record)

    # Filter out rows where key columns are missing
    records = [r for r in records if r["date"] is not None and r["site"] is not None]

    base = output_dir / f"{task}_{model}"

    # JSON
    with open(f"{base}.json", "w") as f:
        json.dump(records, f, indent=2, default=str)

    # Parquet
    pdf = pd.DataFrame(records)
    pdf.to_parquet(f"{base}.parquet", index=False)

    # Quick metrics summary
    valid_reg = [(r["actual_da"], r["predicted_da"]) for r in records
                 if r["actual_da"] is not None and r["predicted_da"] is not None]
    if valid_reg and task == "regression":
        from sklearn.metrics import r2_score, mean_absolute_error, f1_score

        actuals = [r[0] for r in valid_reg]
        preds = [r[1] for r in valid_reg]
        r2 = r2_score(actuals, preds)
        mae = mean_absolute_error(actuals, preds)
        spike_actual = [1 if a > config.SPIKE_THRESHOLD else 0 for a in actuals]
        spike_pred = [1 if p > config.SPIKE_THRESHOLD else 0 for p in preds]
        f1 = f1_score(spike_actual, spike_pred, zero_division=0)
        print(f"  {task}_{model}: {len(records)} records  R²={r2:.4f}  MAE={mae:.2f}  F1={f1:.4f}")
    else:
        print(f"  {task}_{model}: {len(records)} records")


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw validation CSV to API cache format"
    )
    parser.add_argument(
        "--input", "-i",
        default="raw_validation_plots/raw_data_validation_results.csv",
        help="Path to the raw validation CSV (default: raw_validation_plots/raw_data_validation_results.csv)",
    )
    parser.add_argument(
        "--output", "-o",
        default="./cache",
        help="Path to the cache output directory (default: ./cache)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        print("Run validate_on_raw_data.py on Hyak first, then SCP results to local.")
        sys.exit(1)

    convert(args.input, args.output)


if __name__ == "__main__":
    main()
