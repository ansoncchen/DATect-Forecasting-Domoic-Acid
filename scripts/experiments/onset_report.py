#!/usr/bin/env python3
"""Score onset round 2 (docs/FORECAST_EXPERIMENTS.md §6) with the declared rule.

Classifier variants (A-D) come from ``onset_study.py`` parquets; the alert-design models
(E) are rolling-origin logistic regressions on the committed out-of-sample predictions.
Onset rows follow ``forecasting.metrics.onset_scores``: last value known at the anchor
< 20 and a measurement >= 20; quiet rows are the other rows with last known < 20.
CIs resample calendar months (every row of a year-month together).

    python scripts/experiments/onset_report.py --raw-dir <onset_study out-dir>
    python scripts/experiments/onset_report.py --alert-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from onset_study import COMMITTED, REGIONS, SITE_REGION  # noqa: E402

BAND = "results/conformal_band/band_predictions.parquet"
OUT = ROOT / "results" / "experiments"
SPIKE, ALERT_REG, ALERT_P = 20.0, 12.0, 0.10
N_BOOT = 2000
BUDGET = 0.10  # P1: quiet-week alarm budget (amendment §6.1)
VALIDATION = ("2019", "2022")
CLASSIFIER_WINDOWS = {"2015–2021": ("2015", "2022"), "2015–2018": ("2015", "2019"), "2019–2021": ("2019", "2022")}
ALERT_WINDOWS = {"2016–2021": ("2016", "2022"), "2016–2018": ("2016", "2019"), "2019–2021": ("2019", "2022")}
HOLDOUT = {"2022–2023": ("2022", "2024")}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def candidates(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    last = df["naive_prediction"].to_numpy(float)
    cand = ~np.isnan(last) & (last < SPIKE)
    onset = cand & (df["actual_da"].to_numpy(float) >= SPIKE)
    return cand, onset


def block_resamples(dates: pd.Series, n_boot: int = N_BOOT, seed: int = 42) -> list[np.ndarray]:
    """Row indices of ``n_boot`` resamples that draw calendar months with replacement."""
    ym = pd.to_datetime(dates).dt.to_period("M").to_numpy()
    months, inv = np.unique(ym, return_inverse=True)
    rows = [np.flatnonzero(inv == i) for i in range(len(months))]
    rng = np.random.default_rng(seed)
    return [np.concatenate([rows[j] for j in rng.integers(0, len(months), len(months))])
            for _ in range(n_boot)]


def auc(onset: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(onset, score)) if 0 < onset.sum() < len(onset) else np.nan


def _lowest_admissible(values: np.ndarray, limit: float) -> float:
    """Lowest t among ``values`` with #(values >= t) <= limit (inf if none)."""
    srt = np.sort(values)
    cand = np.unique(values)[::-1]
    count = len(srt) - np.searchsorted(srt, cand, side="left")
    ok = count <= limit
    return float(cand[ok][-1]) if ok.any() else np.inf


def recall_at_far(onset: np.ndarray, score: np.ndarray, far: float) -> float:
    """Onset recall when the alarm is score >= t, with t the lowest value that keeps the
    quiet-row alarm rate <= far (ties included); with no such value, only scores above
    every quiet score alarm."""
    q = score[~onset]
    t = _lowest_admissible(q, far * len(q))
    return float((score[onset] >= t).mean()) if np.isfinite(t) else float((score[onset] > q.max()).mean())


def caught_at_budget(onset: np.ndarray, score: np.ndarray, far: float = BUDGET) -> int:
    """P1: onsets caught when the threshold is the lowest score keeping <= ``far`` of the
    quiet rows alarmed (as ``recall_at_far``, as a count)."""
    return int(round(recall_at_far(onset, score, far) * onset.sum()))


def threshold_for(onset: np.ndarray, score: np.ndarray, far: float) -> float:
    """The threshold P1 uses, for transfer to other windows (inf: above every quiet score)."""
    q = score[~onset]
    t = _lowest_admissible(q, far * len(q))
    return t if np.isfinite(t) else float(np.nextafter(q.max(), np.inf))


def transfer(df: pd.DataFrame, score: np.ndarray, t: float) -> dict:
    """P2: a threshold fixed elsewhere, applied to ``df``'s onset-candidate rows."""
    cand, onset = candidates(df)
    alarm = cand & (score >= t)
    fa = alarm & ~onset
    meas = df["actual_da"].to_numpy(float)
    caught = int((alarm & onset).sum())
    return {"caught": caught, "n_onset": int(onset.sum()), "quiet_rate": float(fa.sum() / (cand & ~onset).sum()),
            "false_alarms": int(fa.sum()), "fa_per_caught": float(fa.sum() / caught) if caught else np.inf,
            "fa_lt5": int((fa & (meas < 5)).sum()), "fa_5_10": int((fa & (meas >= 5) & (meas < 10)).sum()),
            "fa_10_20": int((fa & (meas >= 10)).sum())}


def matched_threshold(score: np.ndarray, quiet: np.ndarray, fixed: np.ndarray, budget: int) -> float:
    """Lowest t such that (score >= t) | fixed raises at most ``budget`` quiet alarms."""
    free = quiet & ~fixed
    used = int((quiet & fixed).sum())
    return _lowest_admissible(score[free], budget - used)


def paired(onset_c: np.ndarray, s0: np.ndarray, s1: np.ndarray, boots: list[np.ndarray], fn) -> tuple:
    d = []
    for idx in boots:
        o = onset_c[idx]
        if 0 < o.sum() < len(o):
            d.append(fn(o, s1[idx]) - fn(o, s0[idx]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(fn(onset_c, s1) - fn(onset_c, s0)), float(lo), float(hi)


def compare(df: pd.DataFrame, s0: np.ndarray, s1: np.ndarray, alert0: np.ndarray,
            fixed1: np.ndarray, deployed1: np.ndarray | None = None) -> dict:
    """Variant score s1 vs baseline score s0 on the onset-candidate rows of ``df``.

    alert0: the baseline alert (bool per row). fixed1: rows the variant's alert raises
    regardless of s1 (the regression arm for a classifier; nothing for a single score).
    The variant's threshold is matched to alert0's quiet-alarm count."""
    cand, onset = candidates(df)
    c = np.flatnonzero(cand)
    oc, q = onset[c], cand[c] & ~onset[c]
    boots = block_resamples(df["date"].iloc[c].reset_index(drop=True))
    budget = int((alert0[c] & q).sum())
    t = matched_threshold(s1[c], q, fixed1[c], budget)
    alert1 = (s1[c] >= t) | fixed1[c]
    hit0, hit1 = alert0[c][oc], alert1[oc]
    d_hits, d_p1 = [], []
    for idx in boots:
        o = oc[idx]
        if o.any():
            d_hits.append(int((alert1[idx] & o).sum()) - int((alert0[c][idx] & o).sum()))
        if 0 < o.sum() < len(o):
            d_p1.append(caught_at_budget(o, s1[c][idx]) - caught_at_budget(o, s0[c][idx]))
    out = {"n_cand": int(len(c)), "n_onset": int(oc.sum()), "onset_auc": auc(oc, s1[c]),
           "onset_auc_base": auc(oc, s0[c]),
           "caught10": caught_at_budget(oc, s1[c]), "caught10_base": caught_at_budget(oc, s0[c])}
    out["d_caught10"] = out["caught10"] - out["caught10_base"]
    out["d_caught10_lo"], out["d_caught10_hi"] = (float(x) for x in np.percentile(d_p1, [2.5, 97.5]))
    out["d_auc"], out["d_auc_lo"], out["d_auc_hi"] = paired(oc, s0[c], s1[c], boots, auc)
    for far in (0.10, 0.20):
        out[f"recall_far{int(far * 100)}"] = recall_at_far(oc, s1[c], far)
        out[f"recall_far{int(far * 100)}_base"] = recall_at_far(oc, s0[c], far)
    out.update({"quiet_alarms_matched": int((alert1 & q).sum()), "quiet_alarm_budget": budget,
                "caught_matched": int(hit1.sum()), "caught_base": int(hit0.sum()),
                "d_caught": int(hit1.sum() - hit0.sum()),
                "d_caught_lo": float(np.percentile(d_hits, 2.5)), "d_caught_hi": float(np.percentile(d_hits, 97.5))})
    if deployed1 is not None:
        out["caught_deployed"] = int(deployed1[c][oc].sum())
        out["quiet_rate_deployed"] = float(deployed1[c][q].mean())
    return out


# ---------------------------------------------------------------------------
# Classifier variants (A-D)
# ---------------------------------------------------------------------------

def classifier_rows(raw_dir: Path, tag: str, windows: dict) -> pd.DataFrame:
    base = pd.read_parquet(raw_dir / f"{tag}__baseline.parquet")
    print(f"baseline vs committed spike_probability: max |diff| = "
          f"{np.abs(base['spike_prob'] - base['spike_probability']).max():.2e} over {len(base)} rows")
    rows = []
    for path in sorted(raw_dir.glob(f"{tag}__*.parquet")):
        name = path.stem.split("__", 1)[1]
        if name == "baseline":
            continue
        v = pd.read_parquet(path)[["site", "date", "spike_prob"]]
        m = base.merge(v, on=["site", "date"], suffixes=("_base", ""))
        for wname, (a, b) in windows.items():
            w = m[(m["date"] >= a) & (m["date"] < b)].reset_index(drop=True)
            if w.empty:
                continue
            ens = w["ensemble_prediction"].to_numpy(float)
            p0, p1 = w["spike_prob_base"].to_numpy(float), w["spike_prob"].to_numpy(float)
            alert0 = (p0 >= ALERT_P) | (ens >= ALERT_REG)
            r = compare(w, p0, p1, alert0, ens >= ALERT_REG, (p1 >= ALERT_P) | (ens >= ALERT_REG))
            rows.append({"variant": name, "window": wname, "n_rows": len(w), "n_rows_base": int(len(base[(base.date >= a) & (base.date < b)])), **r})
    return pd.DataFrame(rows)


def decide(tab: pd.DataFrame, pooled: str, subs: tuple[str, str]) -> pd.DataFrame:
    """The amended rule (§6.1): pooled Δ onsets caught at the 10% budget above the seed
    floor (>= 2) with CI > 0, Δ >= 0 in both sub-windows, pooled Δ onset AUC >= 0.
    ``passes_auc_rule`` records the original rule (§6): pooled Δ AUC >= max(0.01, seed
    floor) with CI > 0, Δ AUC >= 0 in both sub-windows, matched-alarm hybrid not worse."""
    p = tab[tab["window"] == pooled].set_index("variant")
    seeds = [v for v in p.index if v.startswith("seed")]
    floor = max([2] + [abs(p.loc[s, "d_caught10"]) for s in seeds])
    floor_auc = max([0.01] + [abs(p.loc[s, "d_auc"]) for s in seeds])
    out = []
    for v, r in p.iterrows():
        sub = {w: tab[(tab["variant"] == v) & (tab["window"] == w)].iloc[0] for w in subs}
        r1 = r["d_caught10"] > floor and r["d_caught10_lo"] > 0
        r2 = all(s["d_caught10"] >= 0 for s in sub.values())
        r3 = r["d_auc"] >= 0
        auc_rule = (r["d_auc"] >= floor_auc and r["d_auc_lo"] > 0
                    and all(s["d_auc"] >= 0 for s in sub.values()) and r["d_caught"] >= 0)
        seed = v.startswith("seed")
        out.append({"variant": v, "floor_caught10": floor, "floor_auc": floor_auc, "rule1": r1, "rule2": r2,
                    "rule3": r3, "passes": bool(r1 and r2 and r3 and not seed),
                    "passes_auc_rule": bool(auc_rule and not seed)})
    return pd.DataFrame(out)


def transfer_rows(frames: dict[str, pd.DataFrame], score_col: str, windows: dict,
                  fars=(0.05, BUDGET, 0.20)) -> pd.DataFrame:
    """P2 and tiers: thresholds set on validation for each budget, applied to every window.
    ``frames`` maps a name to a frame with the score in ``score_col``."""
    rows = []
    for name, df in frames.items():
        v = df[(df["date"] >= VALIDATION[0]) & (df["date"] < VALIDATION[1])]
        cand, onset = candidates(v)
        s = v[score_col].to_numpy(float)
        for far in fars:
            t = threshold_for(onset[cand], s[cand], far)
            for wname, (a, b) in windows.items():
                w = df[(df["date"] >= a) & (df["date"] < b)]
                rows.append({"variant": name, "budget": far, "threshold": t, "window": wname,
                             **transfer(w, w[score_col].to_numpy(float), t)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Alert design (E): rolling-origin logistic regressions on committed predictions
# ---------------------------------------------------------------------------

def _logit(p, eps=1e-4):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def alert_frame() -> pd.DataFrame:
    d = pd.read_parquet(ROOT / COMMITTED)
    d["date"] = pd.to_datetime(d["date"])
    b = pd.read_parquet(ROOT / BAND)[["site", "date", "p_exceed"]]
    b["date"] = pd.to_datetime(b["date"])
    d = d.merge(b, on=["site", "date"], how="left")
    d["region"] = d["site"].map(SITE_REGION)
    d["woy"] = d["date"].dt.isocalendar().week.astype(int).clip(upper=52)
    d["year"] = d["date"].dt.year
    return d.sort_values(["date", "site"]).reset_index(drop=True)


def seasonal_prior(train: pd.DataFrame, rows: pd.DataFrame, exclude_own_year: bool) -> np.ndarray:
    """Log-odds of the onset rate for the row's region and week of year (±2 weeks) among
    ``train``'s candidate rows, with add-one smoothing; optionally leaving out the row's
    own year (for training rows)."""
    t = train[["region", "woy", "year", "onset"]]
    out = np.empty(len(rows))
    for i, (reg, wk, yr) in enumerate(zip(rows["region"], rows["woy"], rows["year"])):
        dist = np.abs(t["woy"].to_numpy() - wk)
        near = (t["region"].to_numpy() == reg) & (np.minimum(dist, 52 - dist) <= 2)
        if exclude_own_year:
            near &= t["year"].to_numpy() != yr
        k, n = t["onset"].to_numpy()[near].sum(), near.sum()
        out[i] = np.log((k + 1) / (n - k + 1))
    return out


def design(df: pd.DataFrame, kind: str, prior: np.ndarray | None = None) -> np.ndarray:
    lp = _logit(df["spike_probability"])
    cols = [lp]
    if kind in ("stack", "stack+band"):
        cols += [np.log1p(df["ensemble_prediction"].clip(lower=0)), np.log1p(df["naive_prediction"].clip(lower=0))]
    if kind == "stack+band":
        cols.append(_logit(df["p_exceed"].fillna(0.0)))
    if kind == "region_calibrated":
        for r in list(REGIONS)[1:]:
            ind = (df["region"] == r).to_numpy(float)
            cols += [ind, ind * lp]
    if kind == "seasonal_prior":
        cols.append(prior)
    return np.column_stack(cols)


def rolling_scores(d: pd.DataFrame, kind: str, years: range) -> pd.Series:
    """Score of each candidate row in ``years`` from a model refit every 1 January on the
    candidate rows dated before it."""
    cand = d[(d["naive_prediction"] < SPIKE) & d["naive_prediction"].notna()].copy()
    cand["onset"] = (cand["actual_da"] >= SPIKE).astype(int)
    out = pd.Series(np.nan, index=d.index)
    for y in years:
        tr = cand[cand["date"] < f"{y}-01-01"]
        te = cand[cand["year"] == y]
        if te.empty:
            continue
        pr_tr = pr_te = None
        if kind == "seasonal_prior":
            pr_tr = seasonal_prior(tr, tr, exclude_own_year=True)
            pr_te = seasonal_prior(tr, te, exclude_own_year=False)
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
        model.fit(design(tr, kind, pr_tr), tr["onset"])
        out.loc[te.index] = model.predict_proba(design(te, kind, pr_te))[:, 1]
    return out


def alert_rows(windows: dict, years: range) -> pd.DataFrame:
    d = alert_frame()
    hybrid_score = np.maximum(d["spike_probability"] / ALERT_P, d["ensemble_prediction"] / ALERT_REG).to_numpy()
    alert0 = d["spike_alert"].astype(bool).to_numpy()
    rows = []
    for kind in ("stack", "stack+band", "region_calibrated", "seasonal_prior"):
        s = rolling_scores(d, kind, years).to_numpy()
        for wname, (a, b) in windows.items():
            m = ((d["date"] >= a) & (d["date"] < b)).to_numpy()
            w = d[m].reset_index(drop=True)
            s1 = np.where(np.isnan(s[m]), -np.inf, s[m])
            r = compare(w, hybrid_score[m], s1, alert0[m], np.zeros(m.sum(), bool))
            rows.append({"variant": kind, "window": wname, "n_rows": len(w), **r})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Recalibration (R): rolling maps fitted on past forecasts measured by the anchor
# ---------------------------------------------------------------------------

RECAL = ("raw", "platt", "isotonic", "quantile_2y", "platt_2y")


def recalibrated(d: pd.DataFrame, kind: str, start: str) -> pd.Series:
    """Score of every candidate row dated >= ``start``. Per anchor, the map is fitted on
    the candidate rows measured by it (date <= anchor; ``_2y``: the last 730 days). R3
    returns P minus the 90th percentile of P among those rows' quiet weeks, so it alarms
    at score > 0 without a threshold to choose."""
    from sklearn.isotonic import IsotonicRegression
    cand = d[(d["naive_prediction"] < SPIKE) & d["naive_prediction"].notna()]
    onset = (cand["actual_da"] >= SPIKE).to_numpy()
    p = cand["spike_probability"].to_numpy(float)
    dates = cand["date"].to_numpy("datetime64[ns]")
    out = pd.Series(np.nan, index=d.index)
    for A, g in cand[cand["date"] >= start].groupby("anchor_date"):
        x = g["spike_probability"].to_numpy(float)
        if kind == "raw":
            out.loc[g.index] = x
            continue
        past = dates <= np.datetime64(pd.Timestamp(A), "ns")
        if kind.endswith("_2y"):
            past &= dates > np.datetime64(pd.Timestamp(A) - pd.Timedelta(days=730), "ns")
        pp, oo = p[past], onset[past]
        if kind == "quantile_2y":
            s = x - np.quantile(pp[~oo], 1 - BUDGET)
        elif oo.sum() < 2 or oo.all():
            s = np.full(len(x), oo.mean() if len(oo) else 0.0)
        elif kind in ("platt", "platt_2y"):
            m = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(pp)[:, None], oo)
            s = m.predict_proba(_logit(x)[:, None])[:, 1]
        else:
            s = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(pp, oo).predict(x)
        out.loc[g.index] = s
    return out


def recalibration_rows(years: range) -> tuple[pd.DataFrame, pd.DataFrame]:
    """R1-R4 vs the raw probability: validation threshold at 10% (R3: alarm at > 0),
    realized quiet-week alarm rate per year, and the transfer to 2015-2018."""
    d = alert_frame()
    d["anchor_date"] = pd.to_datetime(d["anchor_date"])
    rows, yearly = [], []
    for kind in RECAL:
        s = recalibrated(d, kind, f"{years.start}-01-01").to_numpy()
        f = d.assign(score=np.where(np.isnan(s), -np.inf, s))
        v = f[(f["date"] >= VALIDATION[0]) & (f["date"] < VALIDATION[1])]
        cand, onset = candidates(v)
        t = float(np.nextafter(0.0, 1.0)) if kind == "quantile_2y" else threshold_for(
            onset[cand], v["score"].to_numpy()[cand], BUDGET)
        for y in years:
            w = f[f["date"].dt.year == y]
            yearly.append({"variant": kind, "year": y, **transfer(w, w["score"].to_numpy(), t)})
        for wname, (a, b) in {"2015–2018": ("2015", "2019"), "2019–2021": VALIDATION}.items():
            w = f[(f["date"] >= a) & (f["date"] < b)]
            r = transfer(w, w["score"].to_numpy(), t)
            c, o = candidates(w)
            r["caught10_in_window"] = caught_at_budget(o[c], w["score"].to_numpy()[c])
            rows.append({"variant": kind, "window": wname, "threshold": t, **r})
    yearly = pd.DataFrame(yearly)
    tab = pd.DataFrame(rows)
    mad = yearly.groupby("variant")["quiet_rate"].apply(lambda r: float(np.mean(np.abs(r - BUDGET))))
    tab["yearly_mad"] = tab["variant"].map(mad)
    raw = tab[(tab["variant"] == "raw") & (tab["window"] == "2015–2018")].iloc[0]
    dev = tab[tab["window"] == "2015–2018"].set_index("variant")
    dec = {k: bool(abs(r["quiet_rate"] - BUDGET) <= 0.03 and r["yearly_mad"] < raw["yearly_mad"]
                   and r["caught"] / max(r["false_alarms"], 1) >= raw["caught"] / max(raw["false_alarms"], 1))
           for k, r in dev.iterrows() if k != "raw"}
    tab["passes"] = tab["variant"].map(dec)
    return tab, yearly


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def ci(d, lo, hi, nd=3) -> str:
    return f"{d:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def markdown(tab: pd.DataFrame, pooled: str, subs: tuple[str, str], dec: pd.DataFrame | None) -> str:
    hdr = ["Change", f"Caught at 10% budget, {pooled} (Δ) [95% CI]", f"Δ caught {subs[0]} / {subs[1]}",
           f"Onset AUC {pooled}", f"Δ AUC [95% CI]", f"Δ AUC {subs[0]} / {subs[1]}",
           "Matched-alarm hybrid: caught (Δ)", "Deployed P ≥ 0.10: caught / quiet rate", "Passes (amended)", "Passes (original AUC rule)"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "|".join("---" for _ in hdr) + "|"]
    p = tab[tab["window"] == pooled].set_index("variant")
    dd = dec.set_index("variant") if dec is not None else None
    for v, r in p.iterrows():
        sub = [tab[(tab["variant"] == v) & (tab["window"] == w)].iloc[0] for w in subs]
        dep = (f"{int(r['caught_deployed'])}/{int(r['n_onset'])} at {r['quiet_rate_deployed']:.1%}"
               if "caught_deployed" in r and pd.notna(r.get("caught_deployed")) else "—")
        yes = lambda k: "—" if dd is None or v.startswith("seed") else ("**yes**" if dd.loc[v, k] else "no")
        lines.append("| " + " | ".join([
            v,
            f"{int(r['caught10'])}/{int(r['n_onset'])} ({int(r['d_caught10']):+d}) [{r['d_caught10_lo']:+.0f}, {r['d_caught10_hi']:+.0f}]",
            f"{int(sub[0]['d_caught10']):+d} / {int(sub[1]['d_caught10']):+d}",
            f"{r['onset_auc']:.3f}", ci(r["d_auc"], r["d_auc_lo"], r["d_auc_hi"]),
            f"{sub[0]['d_auc']:+.3f} / {sub[1]['d_auc']:+.3f}",
            f"{int(r['caught_matched'])} ({int(r['d_caught']):+d})", dep,
            yes("passes"), yes("passes_auc_rule")]) + " |")
    return "\n".join(l.replace("-0.000", "+0.000") for l in lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="results/experiments/onset_raw")
    ap.add_argument("--tag", default="2015_2022")
    ap.add_argument("--alert-only", action="store_true")
    ap.add_argument("--holdout", action="store_true", help="score the holdout (run once, shortlist only)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.holdout:
        return 0
    pd.set_option("display.width", 250)
    al = alert_rows(ALERT_WINDOWS, range(2016, 2022))
    adec = decide(al, "2016–2021", ("2016–2018", "2019–2021"))
    al.to_csv(OUT / "onset_round2_alert.csv", index=False, float_format="%.4f")
    print("\nE. Alert design (rolling-origin; vs the deployed hybrid's score max(P/0.10, ensemble/12))\n")
    print(markdown(al, "2016–2021", ("2016–2018", "2019–2021"), adec))
    rc, yearly = recalibration_rows(range(2015, 2022))
    rc.to_csv(OUT / "onset_round2_recalibration.csv", index=False, float_format="%.4f")
    yearly.to_csv(OUT / "onset_round2_recalibration_yearly.csv", index=False, float_format="%.4f")
    print("\nR. Recalibration (threshold from 2019-2021 at 10%; quantile_2y alarms above its rolling 90th percentile)\n")
    print(rc.round(3).to_string(index=False))
    print(yearly.pivot(index="year", columns="variant", values="quiet_rate").round(3).to_string())
    if args.alert_only:
        return 0
    raw_dir = Path(args.raw_dir)
    tab = classifier_rows(raw_dir, args.tag, CLASSIFIER_WINDOWS)
    dec = decide(tab, "2015–2021", ("2015–2018", "2019–2021"))
    tab.to_csv(OUT / "onset_round2_classifier.csv", index=False, float_format="%.4f")
    dec.to_csv(OUT / "onset_round2_decision.csv", index=False)
    print(f"\nA-D. Classifier variants (floors: {int(dec['floor_caught10'].iloc[0])} onsets at 10%, "
          f"{dec['floor_auc'].iloc[0]:.3f} AUC)\n")
    print(markdown(tab, "2015–2021", ("2015–2018", "2019–2021"), dec))
    frames = {p.stem.split("__", 1)[1]: pd.read_parquet(p) for p in sorted(raw_dir.glob(f"{args.tag}__*.parquet"))}
    tr = transfer_rows(frames, "spike_prob", {"2015–2018": ("2015", "2019"), "2019–2021": VALIDATION})
    tr.to_csv(OUT / "onset_round2_transfer.csv", index=False, float_format="%.4f")
    print("\nP2. Thresholds set on 2019-2021, applied to 2015-2018\n")
    print(tr[tr["window"] == "2015–2018"].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
