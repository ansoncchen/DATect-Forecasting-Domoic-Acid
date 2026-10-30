#!/usr/bin/env python3
"""
Gain importance of the pooled spike classifier, averaged over anchor dates.

The classifier is retrained at every anchor on all sites' real measurements up
to that date (RawForecastEngine._build_pooled_spike_data), so its importances
depend only on the anchor. For each multi-seed run (seeds 42-46) this draws
anchors from the anchors that run actually used, retrains the classifier exactly
as the engine does, and averages the normalized gain importances.

Usage (from repository root; run on Hyak):
    python3 scripts/eval/spike_classifier_importance.py [--anchors-per-seed 60]

Output: results/spike_classifier/importance_by_seed.csv
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd


def main():
    from _repo import ensure_repo_root

    ensure_repo_root()
    import config
    from forecasting.raw_forecast_engine import RawForecastEngine

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--anchors-per-seed", type=int, default=60)
    args = parser.parse_args()

    engine = RawForecastEngine(validate_on_init=False)
    feature_frame = engine._load_feature_frame()

    rows = []
    for path in sorted(glob.glob("results/multi_seed/baseline_seed*_predictions.parquet")):
        seed = int(re.search(r"seed(\d+)", path).group(1))
        anchors = np.sort(pd.to_datetime(pd.read_parquet(path)["anchor_date"]).unique())
        rng = np.random.default_rng(seed)
        picked = rng.choice(anchors, size=min(args.anchors_per_seed, len(anchors)), replace=False)
        fitted = 0
        for anchor in sorted(picked):
            X, y, _, _ = engine._build_pooled_spike_data(feature_frame, pd.Timestamp(anchor))
            if X is None:
                continue
            result = engine.classification_adapter.train_spike_binary_classifier(
                X, y, spike_threshold=config.SPIKE_THRESHOLD
            )
            if result is None:
                continue
            gain = pd.Series(result["model"].feature_importances_, index=result["columns"])
            for feature, value in (gain / gain.sum()).items():
                rows.append({"seed": seed, "anchor": pd.Timestamp(anchor), "feature": feature, "gain": value})
            fitted += 1
        print(f"seed {seed}: {fitted} classifiers from {len(picked)} anchors")

    long = pd.DataFrame(rows)
    by_seed = long.pivot_table(index="feature", columns="seed", values="gain", aggfunc="mean").fillna(0.0)
    by_seed["mean"] = by_seed.mean(axis=1)
    by_seed = by_seed.sort_values("mean", ascending=False)

    os.makedirs("results/spike_classifier", exist_ok=True)
    out = "results/spike_classifier/importance_by_seed.csv"
    by_seed.to_csv(out)
    print(by_seed.head(15).round(3).to_string())
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
