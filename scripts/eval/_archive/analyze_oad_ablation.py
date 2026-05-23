#!/usr/bin/env python3
"""Analyze paper_ablation_results.json for OAD A/B impact."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd

SW_WA = {"Twin Harbors", "Long Beach", "Clatsop Beach", "Cannon Beach"}
SMALL_N_SITES = {"Coos Bay", "Cannon Beach", "Gold Beach", "Newport"}


def main(path: str = "paper_ablation_results.json") -> int:
    data = json.loads(Path(path).read_text())

    if "baseline" not in data or "no_oad_features" not in data:
        print("ERROR: results must contain both 'baseline' and 'no_oad_features' keys")
        return 1

    baseline = data["baseline"]
    no_oad = data["no_oad_features"]

    print("=" * 78)
    print("OAD A/B  —  baseline (+OAD)  vs  no_oad_features")
    print("=" * 78)

    print()
    print("POOLED METRICS")
    print("-" * 78)
    b_ovr = baseline["overall"]
    n_ovr = no_oad["overall"]
    print(f"{'Metric':<10} {'baseline (+OAD)':>20} {'no_oad_features':>20} {'Δ (+OAD−no)':>15}")
    for m in ("r2", "mae", "rmse"):
        b = b_ovr[m]
        n = n_ovr[m]
        d = b - n
        print(f"{m.upper():<10} {b:>20.4f} {n:>20.4f} {d:>+15.4f}")
    print(f"{'N':<10} {b_ovr['n']:>20d} {n_ovr['n']:>20d}")

    print()
    print("PER-SITE R² (baseline − no_oad)  →  positive = OAD helps")
    print("-" * 78)
    rows = []
    for site, b_metrics in baseline["per_site"].items():
        n_metrics = no_oad["per_site"].get(site)
        if n_metrics is None:
            continue
        rows.append({
            "site": site,
            "in_SW_WA": site in SW_WA,
            "baseline_R2": b_metrics["r2"],
            "no_oad_R2": n_metrics["r2"],
            "delta_R2": b_metrics["r2"] - n_metrics["r2"],
            "baseline_MAE": b_metrics["mae"],
            "no_oad_MAE": n_metrics["mae"],
            "delta_MAE": b_metrics["mae"] - n_metrics["mae"],
            "N": b_metrics["n"],
        })
    df = pd.DataFrame(rows).sort_values("delta_R2", ascending=False)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(df.to_string(index=False))

    print()
    print("SW WASHINGTON SUBSET (where OAD's headline signal lives)")
    print("-" * 78)
    sw = df[df["in_SW_WA"]]
    if len(sw):
        sw_delta_r2 = sw["delta_R2"].mean()
        sw_delta_mae = sw["delta_MAE"].mean()
        print(f"  Mean Δ R²  in SW WA: {sw_delta_r2:+.4f} ({len(sw)} sites)")
        print(f"  Mean Δ MAE in SW WA: {sw_delta_mae:+.4f}")
    print()

    print("CONTEXT: other ablations for reference")
    print("-" * 78)
    base_r2 = baseline["overall"]["r2"]
    for name, result in data.items():
        if result is None or name == "baseline":
            continue
        r2 = result["overall"]["r2"]
        delta = r2 - base_r2
        print(f"  {name:<35} R²={r2:.4f}  Δ={delta:+.4f}")

    # Small-N follow-up: does adding OAD to Coos Bay / Cannon Beach / Gold Beach /
    # Newport help despite their handcrafted minimal feature subsets?
    small_n_result = data.get("with_oad_on_small_n")
    if small_n_result is not None:
        print()
        print("SMALL-N OAD EXPERIMENT")
        print("-" * 78)
        print("Per-site R² for the 4 small-N sites: baseline (no OAD)  vs  +OAD added")
        small_n_rows = []
        for site in sorted(SMALL_N_SITES):
            b = baseline["per_site"].get(site)
            w = small_n_result["per_site"].get(site)
            if not b or not w:
                continue
            small_n_rows.append({
                "site": site,
                "baseline_R2": b["r2"],
                "with_oad_R2": w["r2"],
                "delta_R2": w["r2"] - b["r2"],
                "baseline_MAE": b["mae"],
                "with_oad_MAE": w["mae"],
                "delta_MAE": w["mae"] - b["mae"],
                "N": b["n"],
            })
        if small_n_rows:
            sndf = pd.DataFrame(small_n_rows)
            print(sndf.to_string(index=False))
            print()
            print(f"  Mean Δ R²  across the 4 small-N sites: {sndf['delta_R2'].mean():+.4f}")
            print(f"  Mean Δ MAE across the 4 small-N sites: {sndf['delta_MAE'].mean():+.4f}")
            print()
            wins = (sndf["delta_R2"] > 0).sum()
            print(f"  Sites where OAD helped: {wins} of {len(sndf)}")
            if wins >= 3 and sndf["delta_R2"].mean() > 0.01:
                print("  → VERDICT: OAD as 'synthetic data' DOES seem to help small-N sites.")
                print("    Recommendation: promote OAD inclusion to all 10 sites in per_site_models.py.")
            elif wins <= 1 or sndf["delta_R2"].mean() < -0.01:
                print("  → VERDICT: OAD HURTS small-N sites (likely overfitting).")
                print("    Recommendation: keep current 5-site selective inclusion.")
            else:
                print("  → VERDICT: mixed / no clear signal. Keep current selective inclusion for v1.")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "paper_ablation_results.json"))
