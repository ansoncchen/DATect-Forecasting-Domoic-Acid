"""Score the declared one-algorithm comparison (docs/FORECAST_EXPERIMENTS.md §3c).

    scripts/eval/one_algorithm_compare.py results/one_algorithm/shared_config_20030101_20240101.parquet validation   # 2015-18 + 2019-21, applies the rule
    scripts/eval/one_algorithm_compare.py results/one_algorithm/shared_config_20030101_20240101.parquet holdout      # 2022-23, read once after the pick
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

import config
from forecasting.metrics import exceeds

NOPER, STAGE = sys.argv[1], sys.argv[2]   # NOPER: results/one_algorithm/shared_config_20030101_20240101.parquet
COMMITTED = "results/chronological/chronological_regression_ensemble_20030101_20240101.parquet"
WINDOWS = {"validation": {"2015-2018": ("2015-01-01", "2019-01-01"), "2019-2021": ("2019-01-01", "2022-01-01")},
           "holdout": {"2022-2023": ("2022-01-01", "2024-01-01")}}[STAGE]

n = pd.read_parquet(NOPER)
c = pd.read_parquet(COMMITTED)
for d in (n, c):
    d["date"] = pd.to_datetime(d["date"])
key = ["site", "date"]
n = n.sort_values(key).reset_index(drop=True)
c = c.sort_values(key).reset_index(drop=True)
assert (n[key].values == c[key].values).all() and np.allclose(n.actual_da, c.actual_da), "row sets differ"

avg, rf = n["ensemble_prediction"].to_numpy(float), n["predicted_da_rf"].to_numpy(float)
xgb = 2 * avg - rf
assert xgb.min() > -1e-9, xgb.min()
cands = {"X: XGBoost alone": np.clip(xgb, 0, None), "R: random forest alone": rf,
         "50/50 average (ref)": avg, "per-beach picks, committed (ref)": c["ensemble_prediction"].to_numpy(float)}
probs = {"X: XGBoost alone": n, "R: random forest alone": n, "50/50 average (ref)": n,
         "per-beach picks, committed (ref)": c}

y = n["actual_da"].to_numpy(float)
last = n["naive_prediction"].to_numpy(float)
thr = config.SPIKE_THRESHOLD
below = ~exceeds(last, thr)
onset, quiet = below & exceeds(y, thr), below & ~exceeds(y, thr)
rng = np.random.default_rng(42)

for label, (a, b) in WINDOWS.items():
    m = ((n.date >= a) & (n.date < b)).to_numpy()
    print(f"\n{label}  N={m.sum()}  onsets={int((onset & m).sum())}  quiet={int((quiet & m).sum())}")
    for name, p in cands.items():
        sp = probs[name]["spike_probability"].to_numpy(float)
        alert = (np.nan_to_num(sp) >= config.SPIKE_ALERT_PROB_THRESHOLD) | (p >= config.SPIKE_REGRESSION_ALERT_THRESHOLD)
        print(f"  {name:34s} R² {r2_score(y[m], p[m]):.3f}  MAE {mean_absolute_error(y[m], p[m]):.2f}  "
              f"alert: caught {int((alert & onset & m).sum())}, quiet alarmed {int((alert & quiet & m).sum())}")
    idx = rng.integers(0, m.sum(), size=(2000, m.sum()))
    yx, px, pr = y[m], cands["X: XGBoost alone"][m], cands["R: random forest alone"][m]
    d = np.array([r2_score(yx[i], px[i]) - r2_score(yx[i], pr[i]) for i in idx])
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"  ΔR² X − R = {r2_score(yx, px) - r2_score(yx, pr):+.3f} [{lo:+.3f}, {hi:+.3f}]")
    if label == "2019-2021":
        if lo > 0 or hi < 0:
            pick = "X" if lo > 0 else "R"
            why = "interval excludes zero"
        else:
            pick = "X" if mean_absolute_error(yx, px) < mean_absolute_error(yx, pr) else "R"
            why = "tie on R², lower validation MAE"
        print(f"  PICK: {pick} ({why})")
        sites = n.loc[m, "site"].to_numpy()
        print("  per-site R² (X / R):", {s: (round(r2_score(yx[sites == s], px[sites == s]), 2),
                                           round(r2_score(yx[sites == s], pr[sites == s]), 2))
                                       for s in pd.unique(sites) if (sites == s).sum() >= 10})
