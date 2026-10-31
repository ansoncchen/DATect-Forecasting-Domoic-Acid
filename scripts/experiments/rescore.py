#!/usr/bin/env python3
"""Score harness outputs: post-processing variants and paired comparisons.

Reads ``<raw_dir>/<window>__<variant>.parquet`` written by harness.py and writes
``results/experiments/summary_<window>.csv`` with one row per experiment:
paired ΔR² [95% CI] vs the baseline on identical rows, MAE (and paired ΔMAE),
spike recall / precision / F2 (actual >= 20, prediction >= 12) and transition
recall.

    python scripts/experiments/rescore.py --window val --raw-dir results/experiments/raw
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from scoring import (  # noqa: E402
    ALERT, SPIKE, onset_metrics, paired_delta_auc, paired_delta_mae, paired_delta_r2, r2,
    summarize, transition_mask,
)


# ---------------------------------------------------------------------------
# Post-processing (clip rules) on stored raw predictions
# ---------------------------------------------------------------------------

def _bound(res: pd.DataFrame, rule: str) -> np.ndarray:
    """Per-row upper bound under a clip rule; inf = no clip."""
    if rule == "none":
        return np.full(len(res), np.inf)
    if rule == "site":  # committed: the site's quantile of all (real + gap-filled) training targets
        return np.array([res[f"yq{int(round(q * 100))}"].iat[i] for i, q in enumerate(res["clip_q_site"])])
    if rule == "site_real":  # same quantile, real measurements only
        return np.array([res[f"yq{int(round(q * 100))}_real"].iat[i] for i, q in enumerate(res["clip_q_site"])])
    col = {"q99": "yq99", "max": "yq100", "real_q99": "yq99_real", "real_max": "yq100_real"}[rule]
    return res[col].to_numpy(float)


def components(res: pd.DataFrame, rule: str = "site") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b = _bound(res, rule)
    cm = res["clip_max_site"].to_numpy(float) if "clip_max_site" in res else np.full(len(res), np.nan)

    def clip(col):
        v = np.minimum(np.maximum(res[col].to_numpy(float), 0.0), b)
        return np.where(np.isfinite(cm), np.minimum(v, cm), v)
    xgb = clip("xgb_raw")
    rf = clip("rf_raw") if res["rf_raw"].notna().all() else xgb
    lin = clip("lin_raw") if res["lin_raw"].notna().all() else xgb
    return xgb, rf, lin


def ensemble(res: pd.DataFrame, rule: str = "site", weights: str = "site") -> np.ndarray:
    xgb, rf, _ = components(res, rule)
    if weights == "site":
        return res["w_xgb"].to_numpy() * xgb + res["w_rf"].to_numpy() * rf
    if weights == "half":
        return 0.5 * xgb + 0.5 * rf
    if weights == "xgb":
        return xgb
    if weights == "rf":
        return rf
    raise ValueError(weights)


def online_bias(res: pd.DataFrame, pred: np.ndarray, k: int = 10, shrink: float = 10.0,
                mult: bool = False) -> np.ndarray:
    """Causal per-site correction from earlier forecasts whose measurement is known by the anchor.

    Additive: subtract the shrunk mean residual of the last ``k`` such forecasts at the site
    (shrinkage n / (n + shrink)). Multiplicative: scale by the shrunk ratio of sums.
    Only uses rows inside the scored window (the first forecasts at a site are uncorrected).
    """
    out = pred.copy()
    df = res[["site", "date", "anchor_date", "actual_da"]].copy()
    df["pred"] = pred
    for site, g in df.groupby("site"):
        idx = g.index.to_numpy()
        dates = g["date"].to_numpy()
        anchors = g["anchor_date"].to_numpy()
        act = g["actual_da"].to_numpy(float)
        pr = g["pred"].to_numpy(float)
        for j, i in enumerate(idx):
            known = np.flatnonzero(dates <= anchors[j])[-k:]
            if len(known) == 0:
                continue
            n = len(known)
            lam = n / (n + shrink)
            if mult:
                ratio = (act[known].sum() + 1.0) / (pr[known].sum() + 1.0)
                out[i] = pred[i] * (1 + lam * (ratio - 1))
            else:
                out[i] = max(0.0, pred[i] + lam * np.mean(act[known] - pr[known]))
    return out


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def onset_cols(naive, y, base_score, new_score, alert=None) -> dict:
    """Onset columns for a score (higher = more alarming) vs the baseline score."""
    m = onset_metrics(naive, y, new_score, alert=alert)
    da, dalo, dahi = paired_delta_auc(naive, y, base_score, new_score)
    out = {"onset_auc": m["onset_auc"], "d_onset_auc": da, "d_onset_auc_lo": dalo, "d_onset_auc_hi": dahi,
           "onset_recall_far10": m["onset_recall_far10"], "onset_recall_far20": m["onset_recall_far20"],
           "onset_vs_lastvalue_far20": m["onset_vs_lastvalue_far20"], "n_onset": m["n_onset"]}
    if alert is not None:
        out.update({k: m[k] for k in ("alert_onset_recall", "alert_quiet_far", "alert_vs_lastvalue")})
    return out


def row(name: str, hypothesis: str, y, base, new, trans, alert: float = ALERT, base_alert: float = ALERT,
        naive=None) -> dict:
    d, lo, hi = paired_delta_r2(y, base, new)
    dm, dmlo, dmhi = paired_delta_mae(y, base, new)
    s = summarize(y, new, trans, alert)
    sb = summarize(y, base, trans, base_alert)
    verdict = "tie"
    if lo > 1e-9:
        verdict = "better"
    elif hi < -1e-9:
        verdict = "worse"
    elif abs(d) >= 0.02:
        verdict = "lean better" if d > 0 else "lean worse"
    return {
        "experiment": name, "hypothesis": hypothesis, "n": s["n"], "r2": s["r2"],
        "d_r2": d, "d_r2_lo": lo, "d_r2_hi": hi, "mae": s["mae"], "d_mae": dm,
        "d_mae_lo": dmlo, "d_mae_hi": dmhi, "bias": s["bias"],
        "recall": s["recall"], "precision": s["precision"], "f2": s["f2"], "d_f2": s["f2"] - sb["f2"],
        "trans_recall": s.get("trans_recall", np.nan), "verdict": verdict,
        **(onset_cols(naive, y, base, new, alert=np.asarray(new, float) >= alert) if naive is not None else {}),
    }


def load(raw_dir: Path, window: str, variant: str) -> pd.DataFrame | None:
    p = raw_dir / f"{window}__{variant}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    df["anchor_date"] = pd.to_datetime(df["anchor_date"])
    return df.sort_values(["date", "site"]).reset_index(drop=True)


HYPOTHESES = json.loads((ROOT / "scripts" / "experiments" / "hypotheses.json").read_text())


def postprocessing_rows(base: pd.DataFrame) -> list[dict]:
    y = base["actual_da"].to_numpy(float)
    tr = transition_mask(base)
    ens = ensemble(base)
    naive = base["naive_prediction"].to_numpy(float)
    rows = [row("baseline (committed engine)", "reference", y, ens, ens, tr, naive=naive)]
    for rule in ("none", "site_real", "q99", "max", "real_max"):
        rows.append(row(f"clip={rule}", HYPOTHESES["clip"], y, ens, ensemble(base, rule), tr, naive=naive))
    for w in ("half", "xgb", "rf"):
        rows.append(row(f"weights={w} (diagnostic)", HYPOTHESES["weights"], y, ens, ensemble(base, "site", w), tr, naive=naive))
    for w in (0.1, 0.2, 0.3, 0.5):
        rows.append(row(f"naive_blend={w}", HYPOTHESES["naive_blend"], y, ens, (1 - w) * ens + w * naive, tr, naive=naive))
    nc = ensemble(base, "none")
    for w in (0.2, 0.3):
        rows.append(row(f"clip=none + naive_blend={w}", HYPOTHESES["naive_blend"], y, ens, (1 - w) * nc + w * naive, tr, naive=naive))
    for s in (1.1, 1.2, 1.3):
        rows.append(row(f"scale x{s}", HYPOTHESES["scale"], y, ens, s * ens, tr, naive=naive))
    if base["spike_probability"].notna().all():
        p = base["spike_probability"].to_numpy(float)
        for g in (10, 20, 40):
            rows.append(row(f"ens + {g}*spike_prob", HYPOTHESES["spike_prob"], y, ens, ens + g * p, tr, naive=naive))
            rows.append(row(f"clip=none + {g}*spike_prob", HYPOTHESES["spike_prob"], y, ens, nc + g * p, tr, naive=naive))
    for k, mult in ((10, False), (5, False), (10, True)):
        tag = "mult" if mult else "add"
        rows.append(row(f"online bias ({tag}, last {k})", HYPOTHESES["bias"], y, ens,
                        online_bias(base, ens, k=k, mult=mult), tr, naive=naive))
    for t in (8.0, 10.0, 14.0, 16.0, 20.0):
        rows.append(row(f"alert threshold {t:g}", HYPOTHESES["threshold"], y, ens, ens, tr, alert=t, naive=naive))
    return rows


def variant_rows(raw_dir: Path, window: str, base: pd.DataFrame, names: list[str]) -> list[dict]:
    rows = []
    for name in names:
        v = load(raw_dir, window, name)
        if v is None:
            continue
        m = base[["site", "date", "actual_da"]].merge(v, on=["site", "date", "actual_da"], how="inner")
        b = base.merge(m[["site", "date"]], on=["site", "date"], how="inner")
        y = b["actual_da"].to_numpy(float)
        tr = transition_mask(b)
        hyp = HYPOTHESES.get(name, HYPOTHESES.get(name.split("+")[-1], ""))
        tag = "" if len(m) == len(base) else f" [{len(m)}/{len(base)} rows]"
        rows.append(row(name + tag, hyp, y, ensemble(b), ensemble(m), tr, naive=b["naive_prediction"].to_numpy(float)))
        rows.append(row(name + " (both clip=none)" + tag, hyp, y, ensemble(b, "none"), ensemble(m, "none"), tr,
                        naive=b["naive_prediction"].to_numpy(float)))
        if "+" in name:  # also against the combination without its last part
            ref_name = name.rsplit("+", 1)[0]
            ref = load(raw_dir, window, ref_name)
            if ref is not None:
                mm = ref[["site", "date"]].merge(m, on=["site", "date"])
                rr = ref.merge(mm[["site", "date"]], on=["site", "date"])
                rows.append(row(f"{name} vs {ref_name}", hyp, rr["actual_da"].to_numpy(float),
                                ensemble(rr), ensemble(mm), transition_mask(rr),
                                naive=rr["naive_prediction"].to_numpy(float)))
    return rows


def hybrid_rows(raw_dir: Path, window: str, base: pd.DataFrame, prob_threshold: float = 0.10) -> list[dict]:
    """Hybrid alert (classifier >= 0.10 OR ensemble >= 12) for each classifier variant."""
    from sklearn.metrics import roc_auc_score
    from forecasting.metrics import detection_scores
    rows = []
    regs = {"baseline": base}
    for name in ("asof", "refresh"):
        r = load(raw_dir, window, name)
        if r is not None:
            regs[name] = r
    cls_names = sorted(p.name.split("__", 1)[1][:-8] for p in raw_dir.glob(f"{window}__cls_*.parquet"))
    committed = base[["site", "date", "spike_probability"]].rename(columns={"spike_probability": "spike_prob"})
    sources = [("committed classifier", committed)] + [(n, load(raw_dir, window, n)) for n in cls_names]
    for reg_name, reg in regs.items():
        ens = pd.DataFrame({"site": reg["site"], "date": reg["date"], "actual_da": reg["actual_da"],
                            "ens": ensemble(reg)})
        for cls_name, cls in sources:
            if cls is None or cls["spike_prob"].isna().all():
                continue
            m = base[["site", "date"]].merge(ens, on=["site", "date"]).merge(
                cls[["site", "date", "spike_prob"]], on=["site", "date"]).sort_values(["date", "site"])
            y = m["actual_da"].to_numpy(float)
            spike = y >= SPIKE
            tr = transition_mask(m)
            alert = (m["spike_prob"].to_numpy() >= prob_threshold) | (m["ens"].to_numpy() >= ALERT)
            det = detection_scores(spike, alert)
            m = m.merge(base[["site", "date", "naive_prediction", "spike_probability"]], on=["site", "date"])
            naive = m["naive_prediction"].to_numpy(float)
            p = m["spike_prob"].to_numpy(float)
            hyb_score = np.maximum(p / prob_threshold, m["ens"].to_numpy() / ALERT)
            committed_score = np.maximum(m["spike_probability"].to_numpy(float) / prob_threshold,
                                         ensemble(base.merge(m[["site", "date"]], on=["site", "date"])) / ALERT)
            cls_on = onset_cols(naive, y, m["spike_probability"].to_numpy(float), p)
            hyb_on = onset_cols(naive, y, committed_score, hyb_score, alert=alert)
            rows.append({
                "experiment": f"hybrid: {cls_name} + {reg_name} regression", "n": len(m),
                **{f"cls_{k}": v for k, v in cls_on.items() if k != "n_onset"},
                **{f"hyb_{k}": v for k, v in hyb_on.items() if k != "n_onset"},
                "recall": det["recall"], "precision": det["precision"], "f2": det["f2"],
                "trans_recall": float(alert[tr].mean()) if tr.any() else np.nan,
                "cls_only_recall": float((m["spike_prob"].to_numpy() >= prob_threshold)[spike].mean()),
                "cls_only_trans_recall": float((m["spike_prob"].to_numpy() >= prob_threshold)[tr].mean()) if tr.any() else np.nan,
                "auc": roc_auc_score(spike, m["spike_prob"]) if 0 < spike.sum() < len(spike) else np.nan,
                "n_alerts": int(alert.sum()),
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="val")
    ap.add_argument("--raw-dir", default="results/experiments/raw")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    raw_dir = Path(args.raw_dir)
    base = load(raw_dir, args.window, "baseline")
    names = sorted(p.name.split("__", 1)[1][:-8] for p in raw_dir.glob(f"{args.window}__*.parquet"))
    names = [n for n in names if n != "baseline" and not n.startswith("cls_")]
    rows = postprocessing_rows(base) + variant_rows(raw_dir, args.window, base, names)
    out = pd.DataFrame(rows)
    path = Path(args.out) if args.out else ROOT / "results" / "experiments" / f"summary_{args.window}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, float_format="%.4f")
    pd.set_option("display.width", 220)
    cols = ["experiment", "n", "r2", "d_r2", "d_r2_lo", "d_r2_hi", "mae", "d_mae", "f2", "trans_recall", "verdict"]
    print(out[cols].round(3).to_string(index=False))
    ref_cls = load(raw_dir, args.window, "cls_baseline")
    if base["spike_probability"].isna().all() and ref_cls is not None:
        # No committed classifier for this window: the re-implemented one is the reference.
        base = base.drop(columns="spike_probability").merge(
            ref_cls[["site", "date", "spike_prob"]].rename(columns={"spike_prob": "spike_probability"}),
            on=["site", "date"], how="inner")
    if base["spike_probability"].notna().all():
        hyb = pd.DataFrame(hybrid_rows(raw_dir, args.window, base))
        hyb.to_csv(path.with_name(path.stem + "_hybrid.csv"), index=False, float_format="%.4f")
        print(hyb.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
