"""Build data.json for the throwaway dashboard mockup from committed results (offline).

Run from the repo root: .venv/bin/python docs/superpowers/specs/dashboard-mockup/build_data.py \
    docs/superpowers/specs/dashboard-mockup/data.json
"""
import json, sys
import pandas as pd
sys.path.insert(0, ".")
from forecasting.raw_data_forecaster import load_raw_da_measurements

b = pd.read_parquet("results/conformal_band/band_predictions.parquet")
chrono = json.load(open("results/chronological/chronological_regression_ensemble_20220101_20240101.json"))
band = json.load(open("results/conformal_band/band_eval.json"))
raw = load_raw_da_measurements()
raw["date"] = pd.to_datetime(raw["date"])

out = {"overall": {
    "holdout": {k: chrono["overall"][k] for k in ("n", "n_spikes", "r2", "mae", "spike_recall", "spike_precision", "spike_f2")},
    "onset_hybrid": chrono["overall"]["onset"]["hybrid_alert"],
    "band_holdout_coverage": band["holdout"]["coverage"],
    "band_median_width": band["holdout"]["median_width"],
    "band_onset_coverage": band["holdout"]["coverage_at_onsets"],
}, "sites": {}}

# Alarm quality on below-20 weeks (onset weeks plus quiet weeks), and by placeholder tier.
# WARN_AT is a placeholder for the alert experiments' tiers, not a tuned cutoff.
WARN_AT = 0.40
for name, (lo, hi) in {"holdout": ("2022-01-01", "2024-01-01"), "validation": ("2019-01-01", "2022-01-01")}.items():
    w = b[(b["date"] >= lo) & (b["date"] < hi) & (b["naive_prediction"] < 20)]
    spike, alarm = w["actual_da"] >= 20, w["spike_alert"]
    false = w[alarm & ~spike]
    tier = pd.Series("quiet", index=w.index).mask(alarm, "watch").mask(w["spike_probability"] >= WARN_AT, "warning")
    out["overall"][f"alarms_{name}"] = {
        "onsets": int(spike.sum()), "caught": int((alarm & spike).sum()), "alarms": int(alarm.sum()),
        "false": int(len(false)), "false_10_20": int((false["actual_da"] >= 10).sum()),
        "false_below_5": int((false["actual_da"] < 5).sum()),
        "tiers": {t: {"weeks": int((tier == t).sum()), "followed": int(((tier == t) & spike).sum())}
                  for t in ("quiet", "watch", "warning")},
    }
out["overall"]["warn_at"] = WARN_AT

for site, g in b.sort_values("date").groupby("site"):
    cur = g.iloc[-1]
    hist = g[g["date"] >= "2019-01-01"].iloc[:-1]
    r = raw[(raw["site"] == site)].sort_values("date")
    known = r[r["date"] <= cur["anchor_date"]]
    h = g[(g["date"] >= "2022-01-01") & (g["date"] < "2024-01-01")]
    onset = h[(h["naive_prediction"] < 20) & (h["actual_da"] >= 20)]
    quiet = h[(h["naive_prediction"] < 20) & (h["actual_da"] < 20)]
    ps = chrono["per_site"].get(site)
    out["sites"][site] = {
        "current": {
            "date": cur["date"].strftime("%Y-%m-%d"), "anchor": cur["anchor_date"].strftime("%Y-%m-%d"),
            "pred": cur["predicted_da"], "lo": cur["band_lo"], "hi": cur["band_hi"],
            "p_class": cur["spike_probability"], "alert": bool(cur["spike_alert"]),
            "p_band": cur["p_exceed"], "naive": cur["naive_prediction"],
            "lo5": cur["band_lo_0.5"], "hi5": cur["band_hi_0.5"], "lo9": cur["band_lo_0.9"], "hi9": cur["band_hi_0.9"],
        },
        "recent": [{"date": d.strftime("%Y-%m-%d"), "da": float(v)} for d, v in zip(known["date"].tail(6), known["da_raw"].tail(6))],
        "history": [{"date": x.date.strftime("%Y-%m-%d"), "actual": x.actual_da, "pred": x.predicted_da, "lo": x.band_lo, "hi": x.band_hi,
                     "p": x.spike_probability, "alert": bool(x.spike_alert)} for x in hist.itertuples()],
        "holdout": None if ps is None else {k: ps[k] for k in ("n", "n_spikes", "r2", "mae", "spike_recall", "spike_precision", "spike_f2")},
        "n_holdout_rows": int(len(h)),
        "band_cov_holdout": band["holdout"]["coverage_by_site"].get(site),
        "band_cov_validation": band["validation"]["coverage_by_site"].get(site),
        "onsets": {"n": int(len(onset)), "caught": int(onset["spike_alert"].sum()),
                   "quiet": int(len(quiet)), "quiet_alarmed": int(quiet["spike_alert"].sum())},
    }

def rounded(x):
    if isinstance(x, float):
        return round(x, 3)
    if isinstance(x, dict):
        return {k: rounded(v) for k, v in x.items()}
    if isinstance(x, list):
        return [rounded(v) for v in x]
    return x


json.dump(rounded(json.loads(json.dumps(out, default=float))), open(sys.argv[1], "w"), separators=(",", ":"))
for s, v in out["sites"].items():
    print(f"{s:14s} cur={v['current']['pred']:.1f} [{v['current']['lo']:.1f},{v['current']['hi']:.1f}] p={v['current']['p_class']:.2f} alert={v['current']['alert']} last={v['recent'][-1]} cov={v['band_cov_holdout']} onsets={v['onsets']} nhist={len(v['history'])}")
