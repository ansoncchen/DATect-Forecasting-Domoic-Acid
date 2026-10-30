#!/usr/bin/env python3
"""
Per-site R² for each 5-seed bootstrap run (seeds 42-46, 20% of all real DA).

Reads results/multi_seed/baseline_seed*_predictions.parquet (written by
multi_seed_baseline.py) and tabulates per-site and pooled R² per seed, with the
mean and sample standard deviation across seeds. No models are trained.

Usage (from repository root):
    python3 scripts/eval/seed_stability_table.py

Output: results/multi_seed/per_site_r2_by_seed.csv
"""

import glob
import re

import pandas as pd
from sklearn.metrics import r2_score


def main():
    from _repo import ensure_repo_root

    ensure_repo_root()
    import config

    columns = {}
    for path in sorted(glob.glob("results/multi_seed/baseline_seed*_predictions.parquet")):
        seed = int(re.search(r"seed(\d+)", path).group(1))
        df = pd.read_parquet(path).dropna(subset=["actual_da", "predicted_da"])
        r2 = {site: r2_score(g["actual_da"], g["predicted_da"]) for site, g in df.groupby("site")}
        r2["Pooled"] = r2_score(df["actual_da"], df["predicted_da"])
        columns[f"s{seed}"] = r2

    table = pd.DataFrame(columns).reindex(list(config.SITES) + ["Pooled"])
    table["mean"] = table.mean(axis=1)
    table["std"] = table.drop(columns="mean").std(axis=1, ddof=1)

    out = "results/multi_seed/per_site_r2_by_seed.csv"
    table.to_csv(out, index_label="site")
    print(table.round(3).to_string())
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
