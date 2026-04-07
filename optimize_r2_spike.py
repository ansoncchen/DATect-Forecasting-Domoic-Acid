#!/usr/bin/env python3
"""
Optimization runner for DATect R2 + spike skill.

Runs a small ablation matrix across dev and independent settings, then
summarizes the best variant by independent R2 with spike guardrails.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

import config as cfg
from eval_paper_metrics import compute_spike_tables
from forecasting.raw_forecast_engine import RawForecastEngine


@contextmanager
def patched_config(overrides: dict):
    old = {}
    for k, v in overrides.items():
        old[k] = getattr(cfg, k)
        setattr(cfg, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(cfg, k, v)


def evaluate_variant(
    name: str,
    overrides: dict,
    seed: int,
    sample_fraction: float,
    enable_parallel: bool = True,
) -> dict:
    runtime_overrides = dict(overrides)
    runtime_overrides["RANDOM_SEED"] = seed
    runtime_overrides["TEST_SAMPLE_FRACTION"] = sample_fraction
    runtime_overrides["ENABLE_PARALLEL"] = bool(enable_parallel)
    with patched_config(runtime_overrides):
        engine = RawForecastEngine(validate_on_init=False)
        engine.random_seed = seed
        df = engine.run_retrospective_evaluation(task="regression", model_type="xgb")
        if df is None or df.empty:
            return {
                "variant": name,
                "seed": seed,
                "sample_fraction": sample_fraction,
                "status": "failed",
            }
        pred = (
            df["ensemble_prediction"].values.astype(float)
            if "ensemble_prediction" in df.columns
            else df["predicted_da"].values.astype(float)
        )
        actual = df["actual_da"].values.astype(float)
        event_df, transition_df = compute_spike_tables(
            df,
            spike_threshold=getattr(cfg, "SPIKE_THRESHOLD", 20.0),
            reg_alert_threshold=getattr(cfg, "SPIKE_REGRESSION_ALERT_THRESHOLD", 12.0),
        )
        hyb_event = event_df[event_df["model"] == "hybrid"].iloc[0].to_dict()
        hyb_transition = transition_df[transition_df["model"] == "hybrid"].iloc[0].to_dict()
        return {
            "variant": name,
            "seed": seed,
            "sample_fraction": sample_fraction,
            "status": "ok",
            "n": int(len(df)),
            "r2": float(r2_score(actual, pred)),
            "mae": float(mean_absolute_error(actual, pred)),
            "hybrid_event_recall": float(hyb_event["recall"]),
            "hybrid_event_f1": float(hyb_event["f1"]),
            "hybrid_transition_recall": float(hyb_transition["recall"]),
            "hybrid_transition_f1": float(hyb_transition["f1"]),
        }


def main():
    parser = argparse.ArgumentParser(description="Run R2+spike optimization ablations.")
    parser.add_argument("--output-dir", default="eval_results/optimization")
    parser.add_argument("--quick", action="store_true", help="Use smaller sample fractions for fast local checks")
    parser.add_argument("--dev-fraction", type=float, default=None, help="Override dev sample fraction")
    parser.add_argument("--indep-fraction", type=float, default=None, help="Override independent sample fraction")
    parser.add_argument(
        "--variants",
        default="all",
        help="Comma-separated variant names (default: all)",
    )
    parser.add_argument(
        "--only-independent",
        action="store_true",
        help="Run only independent setting (seed=123).",
    )
    parser.add_argument(
        "--disable-parallel",
        action="store_true",
        help="Disable joblib parallelism inside retrospective runs.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dev_fraction = 0.005 if args.quick else 0.20
    indep_fraction = 0.010 if args.quick else 0.40
    if args.dev_fraction is not None:
        dev_fraction = args.dev_fraction
    if args.indep_fraction is not None:
        indep_fraction = args.indep_fraction

    variants = {
        "baseline": {
            "ENABLE_RESIDUAL_CORRECTION": False,
            "ENABLE_DYNAMIC_ENSEMBLE_GATING": False,
            "ENABLE_GAP_AWARE_SAMPLE_WEIGHTS": False,
            "SPIKE_CALIBRATION_METHOD": "none",
        },
        "residual_only": {
            "ENABLE_RESIDUAL_CORRECTION": True,
            "ENABLE_DYNAMIC_ENSEMBLE_GATING": False,
            "ENABLE_GAP_AWARE_SAMPLE_WEIGHTS": False,
            "SPIKE_CALIBRATION_METHOD": "none",
        },
        "gating_only": {
            "ENABLE_RESIDUAL_CORRECTION": False,
            "ENABLE_DYNAMIC_ENSEMBLE_GATING": True,
            "ENABLE_GAP_AWARE_SAMPLE_WEIGHTS": False,
            "SPIKE_CALIBRATION_METHOD": "none",
        },
        "data_weighting": {
            "ENABLE_RESIDUAL_CORRECTION": False,
            "ENABLE_DYNAMIC_ENSEMBLE_GATING": True,
            "ENABLE_GAP_AWARE_SAMPLE_WEIGHTS": True,
            "SPIKE_CALIBRATION_METHOD": "none",
        },
        "full_balanced": {
            "ENABLE_RESIDUAL_CORRECTION": True,
            "ENABLE_DYNAMIC_ENSEMBLE_GATING": True,
            "ENABLE_GAP_AWARE_SAMPLE_WEIGHTS": True,
            "SPIKE_CALIBRATION_METHOD": "platt",
        },
    }
    if args.variants.strip().lower() != "all":
        selected = [x.strip() for x in args.variants.split(",") if x.strip()]
        variants = {k: v for k, v in variants.items() if k in selected}
        if not variants:
            raise ValueError("No valid variants selected. Use names from baseline,residual_only,gating_only,data_weighting,full_balanced")

    rows = []
    settings = [(123, indep_fraction)] if args.only_independent else [(42, dev_fraction), (123, indep_fraction)]
    for variant_name, overrides in variants.items():
        for seed, frac in settings:
            print(f"Running {variant_name} seed={seed} frac={frac}")
            rows.append(
                evaluate_variant(
                    variant_name,
                    overrides,
                    seed,
                    frac,
                    enable_parallel=(not args.disable_parallel),
                )
            )

    result_df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, "optimization_matrix.csv")
    result_df.to_csv(csv_path, index=False)

    # Candidate selection: prioritize independent R2, then transition F1.
    indep = result_df[(result_df["seed"] == 123) & (result_df["status"] == "ok")].copy()
    if indep.empty:
        summary = {"best_variant": None, "reason": "No successful independent runs"}
    else:
        indep = indep.sort_values(
            ["r2", "hybrid_transition_f1", "hybrid_event_f1"],
            ascending=False,
        )
        best = indep.iloc[0].to_dict()
        summary = {
            "best_variant": best["variant"],
            "independent_metrics": best,
            "all_results_csv": csv_path,
        }

    json_path = os.path.join(args.output_dir, "optimization_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
