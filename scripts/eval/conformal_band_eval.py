#!/usr/bin/env python3
"""
Error band on the regression forecast, and the declared one-model test.

Input: one chronological retrospective over every real DA measurement, so each
forecast can be calibrated by all earlier ones:

    python scripts/eval/chronological_eval.py --window 2003-01-01:2024-01-01

The protocol is fixed in docs/ERROR_BAND.md before the holdout is scored:

1. Validation [2019, 2022) chooses the band (window x Mondrian) and the alert
   threshold on its exceedance probability. Selection sees validation rows only.
2. Holdout [2022, 2024) is scored once: coverage of the 80% band, and whether
   the band's P(DA >= 20) catches onsets as well as the pooled spike classifier.

Usage:
    python scripts/eval/conformal_band_eval.py
    python scripts/eval/conformal_band_eval.py --predictions <parquet> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from _repo import ensure_repo_root
    ensure_repo_root()
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config  # noqa: E402
from forecasting.conformal import interval_score, past_residual_bands  # noqa: E402
from forecasting.metrics import onset_scores  # noqa: E402
from forecasting.tuned_config import get_eval_windows  # noqa: E402

COVERAGE = 0.8
CANDIDATES = [{"window": w, "mondrian": m} for w, m in itertools.product((250, 500, None), (False, True))]
COVERAGE_TOLERANCE = 0.03
AUC_MARGIN = 0.05          # band may trail the classifier's onset AUC by at most this (CI lower bound)
ONSET_SLACK = 1            # ...and catch at most this many fewer holdout onsets
ALARM_SLACK = 0.02         # ...at a quiet-week alarm rate at most this much higher
N_BOOT = 10_000
COLUMNS = ["date", "anchor_date", "site", "actual_da", "predicted_da", "naive_prediction",
           "spike_probability", "spike_alert"]


def _in(frame: pd.DataFrame, start: str, end: str) -> pd.Series:
    return (frame["date"] >= start) & (frame["date"] < end)


def _coverage(sub: pd.DataFrame, lo: str = "band_lo", hi: str = "band_hi") -> float:
    y = sub["actual_da"]
    return float(((y >= sub[lo]) & (y <= sub[hi])).mean())


def _band_summary(sub: pd.DataFrame) -> dict:
    thr = config.SPIKE_THRESHOLD
    last = sub["naive_prediction"].to_numpy(dtype=float)
    onset = (last < thr) & (sub["actual_da"].to_numpy() >= thr)
    cat = pd.cut(sub["predicted_da"], [0, 5, 20, np.inf], right=False,
                 labels=["[0,5)", "[5,20)", "[20,inf)"])
    return {
        "n": int(len(sub)),
        "coverage": _coverage(sub),
        "interval_score_log1p": float(interval_score(sub["actual_da"], sub["band_lo"],
                                                     sub["band_hi"], COVERAGE).mean()),
        "median_width": float((sub["band_hi"] - sub["band_lo"]).median()),
        "coverage_by_predicted_category": {
            str(c): {"n": int((cat == c).sum()), "coverage": _coverage(sub[cat == c])}
            for c in cat.cat.categories if (cat == c).sum()},
        "coverage_by_site": {s: _coverage(g) for s, g in sub.groupby("site")},
        "coverage_at_onsets": {"n": int(onset.sum()),
                               "coverage": _coverage(sub[onset]) if onset.any() else None},
    }


def choose_band(val: pd.DataFrame, bands: dict) -> tuple[int, list]:
    """Declared rule: coverage within the tolerance, then lowest interval score;
    if no candidate is within it, the closest coverage."""
    rows = []
    for i, cand in enumerate(CANDIDATES):
        sub = val.join(bands[i]).dropna(subset=["band_lo"])
        rows.append({**cand, "n": len(sub), "coverage": _coverage(sub),
                     "interval_score_log1p": float(interval_score(
                         sub["actual_da"], sub["band_lo"], sub["band_hi"], COVERAGE).mean())})
    ok = [i for i, r in enumerate(rows) if abs(r["coverage"] - COVERAGE) <= COVERAGE_TOLERANCE]
    best = (min(ok, key=lambda i: rows[i]["interval_score_log1p"]) if ok
            else min(range(len(rows)), key=lambda i: abs(rows[i]["coverage"] - COVERAGE)))
    return best, rows


def choose_alert_threshold(val: pd.DataFrame) -> float:
    """Smallest P(DA >= 20) threshold whose validation quiet-week alarm rate does
    not exceed the hybrid alert's, so the two alerts are compared at equal cost."""
    thr = config.SPIKE_THRESHOLD
    last = val["naive_prediction"].to_numpy(dtype=float)
    target = onset_scores(last, val["actual_da"], val["spike_alert"], threshold=thr)["quiet_alarm_rate"]
    for tau in np.unique(val["p_exceed"].to_numpy()):
        rate = onset_scores(last, val["actual_da"], val["p_exceed"] >= tau, threshold=thr)["quiet_alarm_rate"]
        if rate <= target:
            return float(tau)
    return float("inf")


def _onset_block(sub: pd.DataFrame, tau: float) -> dict:
    thr = config.SPIKE_THRESHOLD
    last = sub["naive_prediction"].to_numpy(dtype=float)
    return {
        "band": onset_scores(last, sub["actual_da"], sub["p_exceed"] >= tau, sub["p_exceed"], thr),
        "hybrid_alert": onset_scores(last, sub["actual_da"], sub["spike_alert"],
                                     sub["spike_probability"], thr),
        "regression_line_auc": onset_scores(last, sub["actual_da"], score=sub["predicted_da"],
                                            threshold=thr).get("onset_auc"),
    }


def paired_auc_difference(sub: pd.DataFrame, n_boot: int = N_BOOT, seed: int = 42) -> dict:
    """Band minus classifier onset AUC over the below-threshold rows, with a
    paired row bootstrap (replicates without both classes are skipped)."""
    from sklearn.metrics import roc_auc_score
    thr = config.SPIKE_THRESHOLD
    below = sub["naive_prediction"].to_numpy(dtype=float) < thr
    y = (sub["actual_da"].to_numpy() >= thr)[below]
    a, b = sub["p_exceed"].to_numpy()[below], sub["spike_probability"].to_numpy()[below]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].all() or not y[idx].any():
            continue
        diffs.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    point = roc_auc_score(y, a) - roc_auc_score(y, b)
    return {"difference": float(point), "ci95": [float(np.percentile(diffs, 2.5)),
                                                 float(np.percentile(diffs, 97.5))]}


def parity_with_committed(frame: pd.DataFrame, w: dict) -> dict:
    """The all-years run must reproduce the committed headline forecasts exactly."""
    out = {}
    for start, end in ((w["validation_start"], w["validation_end"]), (w["holdout_start"], w["holdout_end"])):
        path = Path(f"results/chronological/chronological_regression_ensemble_"
                    f"{start.replace('-', '')}_{end.replace('-', '')}.parquet")
        if not path.exists():
            continue
        ref = pd.read_parquet(path, columns=["date", "site", "predicted_da", "spike_probability"])
        ref["date"] = pd.to_datetime(ref["date"])
        m = ref.merge(frame, on=["date", "site"], suffixes=("_ref", ""), validate="one_to_one")
        out[path.name] = {
            "n_ref": len(ref), "n_matched": len(m),
            "max_abs_diff_predicted": float((m["predicted_da"] - m["predicted_da_ref"]).abs().max()),
            "max_abs_diff_probability": float((m["spike_probability"] - m["spike_probability_ref"]).abs().max()),
        }
    return out


def main() -> int:
    w = get_eval_windows()
    p = argparse.ArgumentParser()
    p.add_argument("--predictions",
                   default="results/chronological/chronological_regression_ensemble_20030101_20240101.parquet")
    p.add_argument("--out-dir", default="results/conformal_band")
    args = p.parse_args()

    frame = pd.read_parquet(args.predictions, columns=COLUMNS).reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["anchor_date"] = pd.to_datetime(frame["anchor_date"])
    val_mask = _in(frame, w["validation_start"], w["validation_end"])
    hold_mask = _in(frame, w["holdout_start"], w["holdout_end"])

    # Bands for every row; calibration uses only earlier measurements, so
    # computing them on the whole frame cannot leak.
    bands = {i: past_residual_bands(frame, COVERAGE, **cand) for i, cand in enumerate(CANDIDATES)}
    best, candidate_rows = choose_band(frame[val_mask], bands)
    chosen = CANDIDATES[best]
    frame = frame.join(bands[best])
    for level in (0.5, 0.9):  # calibration check at other levels, same choice
        extra = past_residual_bands(frame[COLUMNS], level, **chosen)[["band_lo", "band_hi"]]
        frame[[f"band_lo_{level}", f"band_hi_{level}"]] = extra.to_numpy()

    val, hold = frame[val_mask].dropna(subset=["band_lo"]), frame[hold_mask].dropna(subset=["band_lo"])
    tau = choose_alert_threshold(val)

    def levels(sub):
        return {str(lv): _coverage(sub, f"band_lo_{lv}", f"band_hi_{lv}") for lv in (0.5, 0.9)}

    hold_onsets = _onset_block(hold, tau)
    auc_diff = paired_auc_difference(hold)
    band_o, hyb_o = hold_onsets["band"], hold_onsets["hybrid_alert"]
    one_model = (auc_diff["ci95"][0] > -AUC_MARGIN
                 and band_o["onsets_caught"] >= hyb_o["onsets_caught"] - ONSET_SLACK
                 and band_o["quiet_alarm_rate"] <= hyb_o["quiet_alarm_rate"] + ALARM_SLACK)

    out = {
        "protocol": "docs/ERROR_BAND.md",
        "parity_with_committed": parity_with_committed(frame, w),
        "coverage_target": COVERAGE,
        "n_rows_with_band": int(frame["band_lo"].notna().sum()),
        "selection": {"candidates": candidate_rows, "chosen": chosen,
                      "alert_threshold_p_exceed": tau},
        "validation": {**_band_summary(val), "coverage_other_levels": levels(val),
                       "onsets": _onset_block(val, tau)},
        "holdout": {**_band_summary(hold), "coverage_other_levels": levels(hold),
                    "onsets": hold_onsets, "auc_band_minus_classifier": auc_diff},
        "decision": {
            "rule": (f"band replaces the classifier if the AUC difference CI lower bound > -{AUC_MARGIN}, "
                     f"it catches >= hybrid onsets - {ONSET_SLACK}, and its quiet alarm rate <= "
                     f"hybrid + {ALARM_SLACK}"),
            "one_model": bool(one_model),
        },
    }

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "band_eval.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    frame.to_parquet(outdir / "band_predictions.parquet", index=False)

    print(f"chosen band: {chosen}; alert at P(DA>=20) >= {tau:.3f}")
    for name, block in (("validation", out["validation"]), ("holdout", out["holdout"])):
        o = block["onsets"]
        print(f"{name}: 80% coverage {block['coverage']:.3f} (50%: {block['coverage_other_levels']['0.5']:.3f}, "
              f"90%: {block['coverage_other_levels']['0.9']:.3f}), median width {block['median_width']:.1f} ug/g")
        for k in ("band", "hybrid_alert"):
            print(f"  {k}: {o[k]['onsets_caught']}/{o[k]['n_onsets']} onsets, "
                  f"{o[k]['quiet_alarm_rate']:.1%} quiet alarms, AUC {o[k].get('onset_auc', float('nan')):.3f}")
    print(f"holdout AUC band - classifier: {auc_diff['difference']:+.3f} "
          f"[{auc_diff['ci95'][0]:+.3f}, {auc_diff['ci95'][1]:+.3f}] -> one model: {one_model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
