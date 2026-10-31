#!/usr/bin/env python3
"""One-time holdout check of the shortlist S1-S5 (docs/FORECAST_EXPERIMENTS.md §4).

Computes only the shortlisted changes, each against the committed engine on the 409
holdout rows; nothing else is scored on the holdout.

    python scripts/experiments/holdout_check.py --raw-dir results/experiments/raw
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from rescore import ensemble, load, onset_cols, row  # noqa: E402
from scoring import ALERT, SPIKE, transition_mask  # noqa: E402
from forecasting.metrics import detection_scores  # noqa: E402

W = "holdout"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="results/experiments/raw")
    args = ap.parse_args()
    raw = Path(args.raw_dir)
    base = load(raw, W, "baseline")
    diff = np.abs(ensemble(base) - base["committed_ensemble"].to_numpy(float))
    print(f"baseline reproduces the committed holdout ensemble: max |diff| = {diff.max():.2e}")
    y = base["actual_da"].to_numpy(float)
    naive = base["naive_prediction"].to_numpy(float)
    tr = transition_mask(base)
    ens = ensemble(base)
    p = base["spike_probability"].to_numpy(float)
    rows = [
        row("S0 committed engine", "reference", y, ens, ens, tr, naive=naive),
        row("S1 clip quantile over real rows", "", y, ens, ensemble(base, "site_real"), tr, naive=naive),
        row("S2 no clip", "", y, ens, ensemble(base, "none"), tr, naive=naive),
        row("S3 level + 10 x P(spike)", "", y, ens, ens + 10 * p, tr, naive=naive),
    ]
    s5 = load(raw, W, "asof+regional+pn_asof")
    m = base[["site", "date"]].merge(s5, on=["site", "date"])
    rows.append(row("S5 asof + cross-site DA + raw PN", "", y, ens, ensemble(m), tr, naive=naive))
    rows.append(row("S5 (both unclipped)", "", y, ensemble(base, "none"), ensemble(m, "none"), tr, naive=naive))
    reg = pd.DataFrame(rows)

    # S4: classifier with refreshed forecast-row lags, inside the hybrid alert
    hyb = []
    for name, probs in (("S0 committed classifier", p),
                        ("S4 classifier, refreshed lags", base[["site", "date"]].merge(
                            load(raw, W, "cls_refresh"), on=["site", "date"])["spike_prob"].to_numpy(float))):
        alert = (probs >= 0.10) | (ens >= ALERT)
        det = detection_scores(y >= SPIKE, alert)
        oc = onset_cols(naive, y, p, probs)
        hc = onset_cols(naive, y, np.maximum(p / 0.10, ens / ALERT), np.maximum(probs / 0.10, ens / ALERT), alert=alert)
        hyb.append({"classifier": name, "onset_auc": oc["onset_auc"], "d_auc": oc["d_onset_auc"],
                    "d_lo": oc["d_onset_auc_lo"], "d_hi": oc["d_onset_auc_hi"],
                    "recall_far20": oc["onset_recall_far20"], "hyb_onsets": hc["alert_onset_recall"],
                    "hyb_quiet_far": hc["alert_quiet_far"], "hyb_vs_lastvalue": hc["alert_vs_lastvalue"],
                    "recall": det["recall"], "precision": det["precision"], "f2": det["f2"],
                    "trans_recall": float(alert[tr].mean()), "n_onset": oc["n_onset"]})
    hyb = pd.DataFrame(hyb)
    out = ROOT / "results" / "experiments"
    reg.to_csv(out / "holdout_shortlist.csv", index=False, float_format="%.4f")
    hyb.to_csv(out / "holdout_shortlist_classifier.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    print(reg[["experiment", "r2", "d_r2", "d_r2_lo", "d_r2_hi", "mae", "d_mae", "f2", "trans_recall",
               "onset_auc", "d_onset_auc", "d_onset_auc_lo", "d_onset_auc_hi", "onset_recall_far20"]].round(3).to_string(index=False))
    print(hyb.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
