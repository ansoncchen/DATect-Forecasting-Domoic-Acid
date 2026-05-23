#!/usr/bin/env python3
"""
Fill placeholders in docs/OAD_INTEGRATION_RESULTS.md from result JSONs.

Usage:
  python scripts/eval/fill_results_doc.py \
      --ablation paper_ablation_results.json \
      [--holdout-baseline holdout_validation/baseline_predictions.parquet] \
      [--holdout-tuned    holdout_validation/tuned_predictions.parquet] \
      [--out docs/OAD_INTEGRATION_RESULTS_FILLED.md]

Reads `docs/OAD_INTEGRATION_RESULTS.md` as the template; writes the filled
version to `--out` (default: same path with `_FILLED` suffix). Placeholders
that cannot be resolved are left as `{{NAME}}` so you can see what's missing.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

SW_WA = ["Twin Harbors", "Long Beach", "Clatsop Beach", "Cannon Beach"]
SMALL_N = ["Coos Bay", "Cannon Beach", "Gold Beach", "Newport"]


def fmt(x, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and (x != x)):
        return "—"
    return f"{x:.{digits}f}"


def build_subs_from_ablation(abl: dict) -> dict:
    subs = {"DATE": date.today().isoformat()}

    if "baseline" not in abl or "no_oad_features" not in abl:
        print("WARN: missing baseline or no_oad_features in ablation JSON")
        return subs

    base = abl["baseline"]
    noad = abl["no_oad_features"]

    # The "+ OAD" condition IS the baseline (which has OAD wired in).
    # The "baseline (no OAD)" is the no_oad_features ablation.
    bo = base["overall"]  # +OAD pooled
    no = noad["overall"]  # baseline-no-OAD pooled

    subs.update({
        "BASELINE_POOLED_R2":  fmt(no["r2"]),
        "BASELINE_POOLED_MAE": fmt(no["mae"], 2),
        "BASELINE_POOLED_RMSE": fmt(no.get("rmse"), 2),
        "BASELINE_POOLED_N":   str(no["n"]),
        "TUNED_POOLED_R2":     fmt(bo["r2"]),
        "TUNED_POOLED_MAE":    fmt(bo["mae"], 2),
        "TUNED_POOLED_RMSE":   fmt(bo.get("rmse"), 2),
        "TUNED_POOLED_N":      str(bo["n"]),
        "DELTA_R2_POOLED":     fmt(bo["r2"] - no["r2"]),
        "DELTA_MAE_POOLED":    fmt(bo["mae"] - no["mae"], 2),
        "DELTA_RMSE_POOLED":   fmt((bo.get("rmse", 0) or 0) - (no.get("rmse", 0) or 0), 2),
        "R2_WITH_OAD":         fmt(bo["r2"]),
        "R2_BASELINE":         fmt(no["r2"]),
        "N_HOLDOUT":           str(bo["n"]),  # NOTE: pooled N here, not holdout — overwrite below if holdout parquets given
    })

    # Ablation table
    abl_map = {
        "ABL_BASE":   abl.get("baseline"),
        "ABL_NI":     abl.get("no_interpolated_training"),
        "ABL_NP":     abl.get("no_per_site_customization"),
        "ABL_NL":     abl.get("no_observation_order_lags"),
        "ABL_ND":     abl.get("no_derived_features"),
        "ABL_NO":     abl.get("no_oad_features"),
    }
    base_r2 = base["overall"]["r2"]
    for k, v in abl_map.items():
        if v is None:
            continue
        ov = v["overall"]
        subs[f"{k}_R2"] = fmt(ov["r2"])
        subs[f"{k}_MAE"] = fmt(ov["mae"], 2)
        subs[f"{k}_N"] = str(ov["n"])
        subs[f"{k}_DELTA"] = "—" if k == "ABL_BASE" else f"{ov['r2'] - base_r2:+.4f}"

    # Per-site SW WA
    bsite = base["per_site"]
    nsite = noad["per_site"]
    sw_deltas = []
    site_key_map = {
        "Twin Harbors": "TH",
        "Long Beach": "LB",
        "Clatsop Beach": "CB",
        "Cannon Beach": "CnB",
    }
    for site, prefix in site_key_map.items():
        b = bsite.get(site, {})
        n = nsite.get(site, {})
        if b and n:
            subs[f"{prefix}_B"] = fmt(n["r2"])
            subs[f"{prefix}_W"] = fmt(b["r2"])
            d = b["r2"] - n["r2"]
            subs[f"{prefix}_D"] = f"{d:+.4f}"
            sw_deltas.append(d)
    if sw_deltas:
        subs["SW_WA_MEAN_DELTA"] = f"{sum(sw_deltas) / len(sw_deltas):+.4f}"

    # Per-site full table (replace the entire stub row)
    rows = []
    for site in sorted(bsite.keys()):
        b = bsite.get(site, {})
        n = nsite.get(site, {})
        if not b or not n:
            continue
        rows.append({
            "site": site,
            "B": fmt(n["r2"]),
            "W": fmt(b["r2"]),
            "D": f"{b['r2'] - n['r2']:+.4f}",
            "N": n["n"],
        })
    if rows:
        df = pd.DataFrame(rows).sort_values(
            "D", key=lambda c: c.str.replace("+", "").astype(float), ascending=False
        )
        table_md = "\n".join(
            f"| {r['site']} | — | {r['B']} | {r['W']} | {r['D']} | {r['N']} |"
            for _, r in df.iterrows()
        )
        subs["__PER_SITE_TABLE__"] = table_md

    # Small-N if present
    sn = abl.get("with_oad_on_small_n")
    if sn:
        sn_site = sn["per_site"]
        wins = 0
        deltas = []
        site_map = {
            "Coos Bay": "CN",
            "Cannon Beach": "CnB",
            "Gold Beach": "GB",
            "Newport": "NP",
        }
        for site, prefix in site_map.items():
            b = nsite.get(site, {})  # baseline-no-OAD per-site
            w = sn_site.get(site, {})
            if not b or not w:
                continue
            d = w["r2"] - b["r2"]
            subs[f"{prefix}_B_sn"] = fmt(b["r2"])
            subs[f"{prefix}_W_sn"] = fmt(w["r2"])
            subs[f"{prefix}_D_sn"] = f"{d:+.4f}"
            subs[f"{prefix}_V"] = "WIN" if d > 0.005 else ("LOSS" if d < -0.005 else "~")
            if d > 0:
                wins += 1
            deltas.append(d)
        if deltas:
            mean_d = sum(deltas) / len(deltas)
            if wins >= 3 and mean_d > 0.01:
                v = "PROMOTE — extend OAD to all 10 sites"
            elif wins <= 1 or mean_d < -0.01:
                v = "KEEP SELECTIVE — current 5-site inclusion is correct"
            else:
                v = "MIXED — keep selective for v1, investigate per-site"
            subs["SMALL_N_VERDICT"] = v

    return subs


def build_subs_from_holdout(baseline_p: str, tuned_p: str) -> dict:
    from sklearn.metrics import r2_score, mean_absolute_error
    subs = {}
    VAL_START = pd.Timestamp("2019-01-01")
    VAL_END = pd.Timestamp("2022-01-01")

    def metrics_for(df, window):
        if window == "holdout":
            mask = df["date"] >= VAL_END
        elif window == "val":
            mask = (df["date"] >= VAL_START) & (df["date"] < VAL_END)
        elif window == "pre":
            mask = df["date"] < VAL_START
        sub = df[mask]
        if len(sub) < 5:
            return None
        return {
            "r2": r2_score(sub["actual_da"], sub["predicted_da"]),
            "mae": mean_absolute_error(sub["actual_da"], sub["predicted_da"]),
            "n": len(sub),
        }

    for label, path in [("baseline", baseline_p), ("tuned", tuned_p)]:
        if not path or not Path(path).exists():
            continue
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        for w_key, w_name in [("holdout", "HOLDOUT"), ("val", "HT_VAL"), ("pre", "HT_PRE")]:
            m = metrics_for(df, w_key)
            if m is None:
                continue
            prefix = "HOLDOUT" if w_key == "holdout" else w_name
            l = "BASELINE" if label == "baseline" else "TUNED"
            subs[f"{prefix}_{l}_R2"] = fmt(m["r2"])
            subs[f"{prefix}_{l}_MAE"] = fmt(m["mae"], 2)
            if w_key == "holdout":
                subs["HOLDOUT_N"] = str(m["n"])

    # Compute holdout delta if both labels present
    if "HOLDOUT_BASELINE_R2" in subs and "HOLDOUT_TUNED_R2" in subs:
        b = float(subs["HOLDOUT_BASELINE_R2"])
        t = float(subs["HOLDOUT_TUNED_R2"])
        subs["HOLDOUT_DELTA_R2"] = f"{t - b:+.4f}"
        bm = float(subs["HOLDOUT_BASELINE_MAE"])
        tm = float(subs["HOLDOUT_TUNED_MAE"])
        subs["HOLDOUT_DELTA_MAE"] = f"{tm - bm:+.2f}"

    return subs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ablation", type=str, default="paper_ablation_results.json")
    p.add_argument("--holdout-baseline", type=str, default=None)
    p.add_argument("--holdout-tuned", type=str, default=None)
    p.add_argument("--template", type=str, default="docs/OAD_INTEGRATION_RESULTS.md")
    p.add_argument("--out", type=str, default="docs/OAD_INTEGRATION_RESULTS_FILLED.md")
    args = p.parse_args()

    tpl = Path(args.template).read_text()
    subs = {}

    if Path(args.ablation).exists():
        abl = json.loads(Path(args.ablation).read_text())
        subs.update(build_subs_from_ablation(abl))
    else:
        print(f"WARN: ablation file not found: {args.ablation}")

    if args.holdout_baseline or args.holdout_tuned:
        subs.update(build_subs_from_holdout(args.holdout_baseline, args.holdout_tuned))

    # Replace placeholders
    out = tpl
    for k, v in subs.items():
        out = out.replace(f"{{{{{k}}}}}", str(v))

    # Replace per-site stub row if we have data
    if "__PER_SITE_TABLE__" in subs:
        out = re.sub(
            r"\| \{\{Site1\}\} \|.*?\| \.\.\. \| \.\.\. \|",
            subs["__PER_SITE_TABLE__"],
            out, count=1, flags=re.DOTALL,
        )

    unfilled = re.findall(r"\{\{[A-Z_]+\}\}", out)
    Path(args.out).write_text(out)
    print(f"Wrote {args.out}")
    print(f"Filled {len(subs)} placeholders, {len(set(unfilled))} unique placeholders remain unfilled")
    if unfilled:
        print("Remaining placeholders (need other JSONs or manual fill):")
        for u in sorted(set(unfilled)):
            print(f"  {u}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
