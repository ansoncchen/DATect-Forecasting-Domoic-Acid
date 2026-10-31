#!/usr/bin/env python3
"""Experiment tables for docs/FORECAST_EXPERIMENTS.md from summary_{val,dev}.csv.

Verdicts are mechanical. Level: "better" if the validation ΔR² CI excludes 0, dev
agrees in sign and |Δ| >= 0.01 (the model-seed noise floor); "worse" if a CI is below 0,
the other window agrees and |Δ| >= 0.01 in that window; "lean ±" if
|ΔR²| >= 0.02 in both windows with the same sign; otherwise "tie". Onset: the same rule
on Δ onset AUC with a 0.02 lean threshold.

    python scripts/experiments/doc_tables.py > /tmp/tables.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "experiments"

SECTIONS = {
    "Post-processing (rescoring of the baseline run; no retraining)": [
        ("clip=none", "Clip off"),
        ("clip=site_real", "Clip quantile over real rows only"),
        ("clip=q99", "Clip at q = 0.99 everywhere"),
        ("weights=half (diagnostic)", "Weights 50/50 at every site (diagnostic)"),
        ("weights=rf (diagnostic)", "RF only at every site (diagnostic)"),
        ("naive_blend=0.1", "10% persistence blend"),
        ("naive_blend=0.3", "30% persistence blend"),
        ("scale x1.2", "Scale forecasts ×1.2"),
        ("ens + 10*spike_prob", "Add 10 × P(spike)"),
        ("ens + 20*spike_prob", "Add 20 × P(spike)"),
        ("online bias (add, last 5)", "Online per-site bias (additive, last 5 verified)"),
        ("online bias (mult, last 10)", "Online per-site bias (multiplicative, last 10)"),
        ("clip=site_real + online bias (add, last 5)", "Clip over real rows + online bias"),
        ("clip=none + online bias (add, last 5)", "Clip off + online bias"),
        ("clip=none + naive_blend=0.2", "Clip off + 20% persistence"),
        ("alert threshold 10", "Regression alert at 10 (F2 only)"),
        ("alert threshold 14", "Regression alert at 14 (F2 only)"),
    ],
    "Training rows and features (retrained)": [
        ("refresh", "Forecast-row lags recomputed to the anchor (A1, test side only)"),
        ("asof", "Training rows built like forecast rows (A1), real rows only"),
        ("asof_gapfill", "asof + gap-filled rows"),
        ("real_only", "Real-measurement rows only (engine rows)"),
        ("obs_age", "Add age of the last measurement"),
        ("no_lags", "Drop observation-order lags 1–4"),
        ("regional", "Add cross-site DA features"),
        ("pn_asof", "Add raw-PN lead features"),
        ("asof+regional", "asof + cross-site DA"),
        ("asof+pn_asof", "asof + raw-PN"),
        ("asof+regional+pn_asof", "asof + cross-site DA + raw-PN"),
        ("no_pn", "Drop PN"),
        ("no_oad", "Drop the 16 OAD features"),
        ("no_oni", "Drop ONI"),
        ("no_discharge", "Drop discharge"),
        ("no_sat", "Drop satellite SST, SST anomaly, FLH"),
        ("no_env", "Drop every environmental feature"),
        ("all_features", "All features at every site"),
        ("no_monotone", "No monotone constraints"),
        ("no_per_site", "No per-site configuration"),
    ],
    "Targets and losses (retrained)": [
        ("log1p", "log1p target"),
        ("log1p_smear", "log1p target + smearing"),
        ("sqrt", "sqrt target"),
        ("tweedie", "Tweedie loss (RF Poisson criterion)"),
        ("poisson", "Poisson loss"),
        ("huber10", "Pseudo-Huber loss (slope 10)"),
        ("delta", "Change target (actual − last known)"),
        ("asof+delta", "asof + change target"),
        ("spike_weight3", "Spike rows weighted ×3"),
    ],
    "Training window, pooling, hyperparameters, noise floor (retrained)": [
        ("recency_hl5", "Recency weights, 5-year half-life"),
        ("recency_hl10", "Recency weights, 10-year half-life"),
        ("window_8y", "Last 8 years only"),
        ("pooled", "One XGBoost across sites (engine rows, site one-hot)"),
        ("xgb_shallow", "XGBoost depth 3"),
        ("xgb_more_trees", "XGBoost ×2 trees"),
        ("rf_leaf1", "RF min leaf 1"),
        ("seed43", "Model seed 43 (noise floor)"),
        ("seed44", "Model seed 44 (noise floor)"),
    ],
}


def ci(d, lo, hi) -> str:
    return "—" if pd.isna(d) else f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def verdict(dv, lov, hiv, dd, lod, hid, lean=0.02, floor=0.01) -> str:
    """``floor``: changes smaller than the model-seed noise floor (|Δ| < 0.01 for both R²
    and onset AUC, from the seed43/seed44 runs) are ties even when the row-bootstrap CI,
    which ignores model-seed variance, excludes zero."""
    if any(pd.isna(x) for x in (dv, dd)):
        return "—"
    if lov > 1e-9 and dd > 0 and dv >= floor:
        return "**better**"
    if ((hiv < -1e-9 and dd < 0 and dv <= -floor) or (hid < -1e-9 and dv < 0 and dd <= -floor)):
        return "worse"
    if dv >= lean and dd >= lean:
        return "lean better"
    if dv <= -lean and dd <= -lean:
        return "lean worse"
    return "tie"


def main() -> None:
    v = pd.read_csv(RES / "summary_val.csv").set_index("experiment")
    d = pd.read_csv(RES / "summary_dev.csv").set_index("experiment")
    for title, items in SECTIONS.items():
        print(f"\n**{title}**\n")
        hdr = ["Experiment", "Val ΔR² [95% CI]", "Val ΔR², both unclipped", "Dev ΔR² [95% CI]", "Val MAE (Δ)",
               "Val F2", "Val trans. recall", "Val Δ onset AUC [95% CI]", "Dev Δ onset AUC", "R² verdict",
               "Onset verdict"]
        print("| " + " | ".join(hdr) + " |")
        print("|" + "|".join("---" for _ in hdr) + "|")
        for key, label in items:
            if key not in v.index:
                continue
            a = v.loc[key]
            b = d.loc[key] if key in d.index else pd.Series(dtype=float)
            nc = f"{key} (both clip=none)"
            nc_txt = f"{v.loc[nc, 'd_r2']:+.3f}" if nc in v.index else "same"
            row = [
                label,
                ci(a.d_r2, a.d_r2_lo, a.d_r2_hi), nc_txt,
                ci(b.get("d_r2"), b.get("d_r2_lo"), b.get("d_r2_hi")),
                f"{a.mae:.2f} ({a.d_mae:+.2f})", f"{a.f2:.3f}",
                "—" if pd.isna(a.trans_recall) else f"{a.trans_recall:.2f}",
                ci(a.d_onset_auc, a.d_onset_auc_lo, a.d_onset_auc_hi),
                "—" if pd.isna(b.get("d_onset_auc", np.nan)) else f"{b['d_onset_auc']:+.3f}",
                verdict(a.d_r2, a.d_r2_lo, a.d_r2_hi, b.get("d_r2", np.nan), b.get("d_r2_lo", np.nan), b.get("d_r2_hi", np.nan)),
                verdict(a.d_onset_auc, a.d_onset_auc_lo, a.d_onset_auc_hi, b.get("d_onset_auc", np.nan),
                        b.get("d_onset_auc_lo", np.nan), b.get("d_onset_auc_hi", np.nan)),
            ]
            print("| " + " | ".join(row) + " |")
    # Classifier / hybrid
    print("\n**Spike classifier and hybrid alert (classifier ≥ 0.10 OR regression ≥ 12)**\n")
    hv = pd.read_csv(RES / "summary_val_hybrid.csv").set_index("experiment")
    hd = pd.read_csv(RES / "summary_dev_hybrid.csv").set_index("experiment")
    hdr = ["Classifier", "Val onset AUC", "Val ΔAUC [95% CI]", "Dev ΔAUC [95% CI]", "Val onset recall @20% FA",
           "Val hybrid: onsets caught / quiet-week alarms", "Dev hybrid: onsets / alarms", "Val F2", "Val trans. recall"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join("---" for _ in hdr) + "|")
    names = {"committed classifier": "Engine (committed)", "cls_safe_fixed": "Safe-baseline filter applied (A2)",
             "cls_refresh": "Forecast-row lags refreshed (A1)", "cls_asof": "asof rows",
             "cls_asof_safe_fixed": "asof rows + safe-baseline filter", "cls_onset": "Onset-only (asof, last < 20)",
             "cls_onset_regional": "Onset-only + cross-site DA", "cls_onset_pn": "Onset-only + raw PN",
             "cls_onset_regional_pn": "Onset-only + cross-site DA + raw PN", "cls_regional": "Engine + cross-site DA",
             "cls_asof_regional": "asof rows + cross-site DA"}
    for key, label in names.items():
        k = f"hybrid: {key} + baseline regression"
        if k not in hv.index:
            continue
        a = hv.loc[k]
        b = hd.loc[k] if k in hd.index else None
        n_on = 34
        row = [label, f"{a.cls_onset_auc:.3f}", ci(a.cls_d_onset_auc, a.cls_d_onset_auc_lo, a.cls_d_onset_auc_hi),
               ci(b.cls_d_onset_auc, b.cls_d_onset_auc_lo, b.cls_d_onset_auc_hi) if b is not None else "—",
               f"{a.cls_onset_recall_far20:.2f}",
               f"{round(a.hyb_alert_onset_recall * n_on)}/{n_on} at {a.hyb_alert_quiet_far:.1%}",
               f"{round(b.hyb_alert_onset_recall * 74)}/74 at {b.hyb_alert_quiet_far:.1%}" if b is not None else "—",
               f"{a.f2:.3f}", f"{a.trans_recall:.2f}"]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
