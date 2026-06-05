# Transport-Aware Toxin Forecasting — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a transport-aware predictor that advects the offshore OAD ocean-anomaly signal to each beach, recovering the predictive signal that naive OAD features waste (prior ΔR²≈0), starting with a Week-1 go/no-go gate.

**Architecture:** A new `forecasting/transport_features.py` (sibling to the existing `oad_features.py`) computes a leak-safe, BEUTI-conditioned advection-lag feature per (site, anchor). It is wired into the feature pipeline behind an env flag (`DATECT_USE_TRANSPORT_FEATURES`), mirroring the existing `DATECT_*` ablation pattern. A multi-seed ablation harness compares no-OAD / naive-OAD / L1-transport on the 2019–2022 validation window and emits a GO/NO-GO verdict.

**Tech Stack:** Python 3, pandas/numpy, pytest, the existing `RawForecastEngine`; Hyak Slurm for the multi-seed run. (Later phases add `torch`/`torchdiffeq` for L4 — out of this plan.)

---

## Scope note

This plan covers **Phase 1 only** (L1 hand-built advection feature + the go/no-go gate + the
evaluation harness everything downstream depends on). Phases 2–6 (L2/L3/L4 operators, conformal
& generative heads, causal mediation, benchmark) are **gated on Phase 1's signal being alive**
and each becomes its own plan — see the Roadmap at the end. Writing exact code for them now
would be speculative. Spec: `docs/superpowers/specs/2026-06-02-transport-aware-toxin-forecasting-design.md`.

---

## File structure (Phase 1)

| File | Responsibility |
|---|---|
| `forecasting/transport_geometry.py` | **new** — static alongshore geometry: site→region, ordered region chain, alongshore distances, upstream-region lookup |
| `forecasting/transport_features.py` | **new** — leak-safe BEUTI-conditioned advection-lag feature builder (`add_transport_features`, `TRANSPORT_FEATURES_ALL`) |
| `forecasting/raw_data_forecaster.py` | **modify** — call `add_transport_features` behind `DATECT_USE_TRANSPORT_FEATURES`, at the same point OAD features are joined |
| `per_site_models.py` | **modify** — append `TRANSPORT_FEATURES_ALL` to feature lists when the flag is on |
| `scripts/eval/transport_ablation.py` | **new** — per-seed subprocess runner: no-OAD / naive-OAD / L1, val window |
| `scripts/eval/transport_ablation.sbatch` | **new** — Hyak array job (seeds 42–46) |
| `scripts/eval/transport_ablation_aggregate.py` | **new** — aggregate per-seed parquets → ΔR²/Δspike-recall mean±std + GO/NO-GO verdict |
| `tests/test_transport_features.py` | **new** — leak-safety, advection-lag math, NaN handling |
| `tests/test_transport_geometry.py` | **new** — chain order, upstream lookup |

**Leak-safety contract (non-negotiable, from `oad_features.py`):** every transport feature for a
row with anchor date `R` may only read OAD/BEUTI data at dates `≤ R − LEAK_SHIFT_DAYS` (=`R−5`),
because MODIS 8-day composites are centered. Reuse `LEAK_SHIFT_DAYS = 5`.

---

## Task 1: Alongshore geometry module

**Files:**
- Create: `forecasting/transport_geometry.py`
- Test: `tests/test_transport_geometry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transport_geometry.py
from forecasting.transport_geometry import (
    REGION_CHAIN, upstream_region, region_centroid_lat,
)

def test_region_chain_is_north_to_south():
    # Olympic (~47.6N) is northmost, Southern OR/N CA (~42.4N) is southmost
    lats = [region_centroid_lat(r) for r in REGION_CHAIN]
    assert lats == sorted(lats, reverse=True), "chain must be ordered N->S"

def test_upstream_during_upwelling_is_north_neighbor():
    # Equatorward (southward) summer upwelling jet => upstream = the region to the NORTH
    assert upstream_region("Central Oregon", beuti=5.0) == "SW Washington / Long Beach"

def test_upstream_during_relaxation_is_self():
    # Relaxation / downwelling (beuti<=0) => no equatorward transport, local source
    assert upstream_region("Central Oregon", beuti=-3.0) == "Central Oregon"

def test_northmost_region_upstream_is_self():
    assert upstream_region("Olympic Coast (WA)", beuti=5.0) == "Olympic Coast (WA)"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_transport_geometry.py -v`
Expected: FAIL with `ModuleNotFoundError: forecasting.transport_geometry`

- [ ] **Step 3: Write minimal implementation**

```python
# forecasting/transport_geometry.py
"""Static alongshore geometry for the transport operator.

PNW shelf transport during the bloom season is dominated by the equatorward
(southward) upwelling jet: when upwelling is active (BEUTI > 0), signal is carried
from a region toward the region immediately to its SOUTH. We therefore define each
region's *upstream* (source) neighbor as the region to its NORTH during upwelling,
and self (local) during relaxation (BEUTI <= 0).
"""
from __future__ import annotations

from forecasting.oad_features import (
    REGION_OLYMPIC, REGION_SW_WA, REGION_CENTRAL_OR, REGION_SOUTHERN,
)

# North -> South
REGION_CHAIN = [REGION_OLYMPIC, REGION_SW_WA, REGION_CENTRAL_OR, REGION_SOUTHERN]

# Approx region centroid latitudes (deg N), for ordering/distance only.
_REGION_LAT = {
    REGION_OLYMPIC: 47.6,
    REGION_SW_WA: 46.4,
    REGION_CENTRAL_OR: 44.0,
    REGION_SOUTHERN: 42.4,
}


def region_centroid_lat(region: str) -> float:
    return _REGION_LAT[region]


def upstream_region(region: str, beuti: float) -> str:
    """Source region whose offshore signal is advected into `region`."""
    if beuti is None or beuti <= 0:
        return region  # relaxation: local source only
    idx = REGION_CHAIN.index(region)
    if idx == 0:
        return region  # northmost: nothing further upstream
    return REGION_CHAIN[idx - 1]  # region to the north
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_transport_geometry.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add forecasting/transport_geometry.py tests/test_transport_geometry.py
git commit -m "feat(transport): alongshore geometry + upstream-region lookup"
```

---

## Task 2: Advection-lag math (pure, leak-safe)

**Files:**
- Create: `forecasting/transport_features.py`
- Test: `tests/test_transport_features.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transport_features.py
import numpy as np
import pandas as pd
from forecasting.transport_features import advection_lag_days, LEAK_SHIFT_DAYS

def test_leak_shift_matches_oad():
    from forecasting.oad_features import LEAK_SHIFT_DAYS as OAD_SHIFT
    assert LEAK_SHIFT_DAYS == OAD_SHIFT == 5

def test_lag_decreases_with_stronger_upwelling():
    # Faster current (higher BEUTI) => shorter transit time
    slow = advection_lag_days(distance_km=150.0, beuti=2.0)
    fast = advection_lag_days(distance_km=150.0, beuti=10.0)
    assert fast < slow

def test_lag_is_clamped_to_bounds():
    # Tiny BEUTI must not produce an unbounded lag
    lag = advection_lag_days(distance_km=150.0, beuti=0.01)
    assert 1 <= lag <= 30

def test_lag_zero_distance_is_min():
    assert advection_lag_days(distance_km=0.0, beuti=5.0) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_transport_features.py -v`
Expected: FAIL with `ModuleNotFoundError: forecasting.transport_features`

- [ ] **Step 3: Write minimal implementation**

```python
# forecasting/transport_features.py  (Part 1 of 2 — math primitives)
"""Leak-safe, BEUTI-conditioned advection-lag transport feature (L1).

For each (site, anchor R), we sample the *upstream* region's OAD score at a lag
equal to the physical transit time from the upstream region to the site, where
transit time = alongshore_distance / advection_speed(BEUTI). All reads are at
dates <= R - LEAK_SHIFT_DAYS to respect the centered-composite leakage rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LEAK_SHIFT_DAYS = 5            # matches oad_features.LEAK_SHIFT_DAYS (centered composites)
_MIN_LAG_DAYS = 1
_MAX_LAG_DAYS = 30
# Map BEUTI (mmol N m^-1 s^-1, O(1-10)) to an effective alongshore speed (km/day).
# Calibrated so a typical summer BEUTI~5 with ~150 km hop gives ~7-10 day transit.
_SPEED_PER_BEUTI_KM_PER_DAY = 4.0
_BASE_SPEED_KM_PER_DAY = 2.0


def advection_lag_days(distance_km: float, beuti: float) -> int:
    """Physical transit time (days) for a parcel over `distance_km` given BEUTI."""
    if distance_km <= 0:
        return _MIN_LAG_DAYS
    speed = _BASE_SPEED_KM_PER_DAY + _SPEED_PER_BEUTI_KM_PER_DAY * max(beuti, 0.0)
    speed = max(speed, 0.1)
    lag = distance_km / speed
    return int(np.clip(round(lag), _MIN_LAG_DAYS, _MAX_LAG_DAYS))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_transport_features.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add forecasting/transport_features.py tests/test_transport_features.py
git commit -m "feat(transport): BEUTI-conditioned advection-lag primitive"
```

---

## Task 3: Per-row transport feature with explicit leak-safety test

**Files:**
- Modify: `forecasting/transport_features.py`
- Test: `tests/test_transport_features.py`

- [ ] **Step 1: Write the failing test** (append to the test file)

```python
def _toy_region_scores():
    # Two regions, daily scores; upstream (north) region has a spike on 2020-06-01
    dates = pd.date_range("2020-05-01", "2020-07-01", freq="D")
    rows = []
    for d in dates:
        rows.append({"date": d, "region": "SW Washington / Long Beach",
                     "score": 100.0 if d == pd.Timestamp("2020-06-01") else 1.0})
        rows.append({"date": d, "region": "Central Oregon", "score": 1.0})
    return pd.DataFrame(rows)

def test_transport_feature_is_leak_safe():
    from forecasting.transport_features import compute_transport_feature
    scores = _toy_region_scores()
    R = pd.Timestamp("2020-06-15")          # anchor
    # Any data strictly after R - 5 must NOT influence the value:
    poisoned = scores.copy()
    poisoned.loc[poisoned["date"] > R - pd.Timedelta(days=LEAK_SHIFT_DAYS), "score"] = 1e9
    v_clean = compute_transport_feature("Newport", R, scores, beuti=5.0, distance_km=150.0)
    v_pois  = compute_transport_feature("Newport", R, poisoned, beuti=5.0, distance_km=150.0)
    assert v_clean == v_pois, "feature must not read data after R - LEAK_SHIFT_DAYS"

def test_transport_feature_picks_up_upstream_spike():
    from forecasting.transport_features import compute_transport_feature
    scores = _toy_region_scores()
    # Newport(Central OR) upstream during upwelling = SW WA; spike on 06-01.
    # With distance 150km, beuti 5 -> lag ~7d; anchor chosen so the window hits 06-01.
    R = pd.Timestamp("2020-06-13")  # d_anchor=06-08, lag~7 -> ~06-01
    v = compute_transport_feature("Newport", R, scores, beuti=5.0, distance_km=150.0)
    assert v > 10.0, "should reflect the upstream spike after advection lag"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_transport_features.py -k transport_feature -v`
Expected: FAIL with `ImportError: cannot import name 'compute_transport_feature'`

- [ ] **Step 3: Implement** (append to `forecasting/transport_features.py`)

```python
# forecasting/transport_features.py  (Part 2 of 2 — per-row feature)
from forecasting.transport_geometry import upstream_region


def compute_transport_feature(
    site: str,
    R: pd.Timestamp,
    region_scores: pd.DataFrame,   # columns: date, region, score
    beuti: float,
    distance_km: float,
    window_days: int = 7,
) -> float:
    """Leak-safe advected upstream-region OAD score for one (site, anchor R)."""
    from forecasting.oad_features import SITE_TO_REGION
    region = SITE_TO_REGION[site]
    src = upstream_region(region, beuti)
    d_anchor = R - pd.Timedelta(days=LEAK_SHIFT_DAYS)        # latest legal read
    lag = advection_lag_days(distance_km, beuti)
    center = d_anchor - pd.Timedelta(days=lag)
    s = region_scores[region_scores["region"] == src].set_index("date")["score"].sort_index()
    if s.empty:
        return np.nan
    lo = center - pd.Timedelta(days=window_days // 2)
    hi = min(center + pd.Timedelta(days=window_days // 2), d_anchor)   # never past d_anchor
    w = s.loc[(s.index >= lo) & (s.index <= hi)]
    return float(w.mean()) if len(w) > 0 else np.nan
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_transport_features.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add forecasting/transport_features.py tests/test_transport_features.py
git commit -m "feat(transport): leak-safe per-row advected upstream feature"
```

---

## Task 4: Frame-level integrator + feature registry + env flag

**Files:**
- Modify: `forecasting/transport_features.py`
- Modify: `forecasting/raw_data_forecaster.py` (join site, at the OAD-join point)
- Modify: `per_site_models.py` (append features under the flag)
- Test: `tests/test_transport_features.py`

- [ ] **Step 1: Write the failing test** (append)

```python
def test_add_transport_features_adds_registered_columns():
    from forecasting.transport_features import add_transport_features, TRANSPORT_FEATURES_ALL
    df = pd.DataFrame({
        "site": ["Newport", "Newport"],
        "date": pd.to_datetime(["2020-06-13", "2020-06-20"]),
        "beuti": [5.0, 5.0],
        "lat": [44.6, 44.6], "lon": [-124.05, -124.05],
    })
    scores = _toy_region_scores()
    out = add_transport_features(df, region_scores=scores)
    for col in TRANSPORT_FEATURES_ALL:
        assert col in out.columns
    assert len(out) == len(df)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_transport_features.py -k add_transport -v`
Expected: FAIL with `ImportError: cannot import name 'add_transport_features'`

- [ ] **Step 3: Implement** (append to `forecasting/transport_features.py`)

```python
TRANSPORT_FEATURES_ALL = ["transport_upstream_score"]

# Alongshore hop distances (km) from each site's region to its upstream region.
# Derived once from region centroid latitudes (~111 km/deg); local hop = 0.
def _hop_distance_km(site: str, beuti: float) -> float:
    from forecasting.oad_features import SITE_TO_REGION
    from forecasting.transport_geometry import region_centroid_lat, upstream_region
    region = SITE_TO_REGION[site]
    src = upstream_region(region, beuti)
    if src == region:
        return 0.0
    return abs(region_centroid_lat(region) - region_centroid_lat(src)) * 111.0


def add_transport_features(df: pd.DataFrame, region_scores: pd.DataFrame) -> pd.DataFrame:
    """Add TRANSPORT_FEATURES_ALL columns to a (site, date, beuti) frame."""
    out = df.copy()
    vals = []
    for _, row in out.iterrows():
        beuti = row.get("beuti", np.nan)
        beuti = 0.0 if pd.isna(beuti) else float(beuti)
        R = pd.Timestamp(row["date"])
        dist = _hop_distance_km(row["site"], beuti)
        vals.append(compute_transport_feature(row["site"], R, region_scores, beuti, dist))
    out["transport_upstream_score"] = vals
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_transport_features.py -v`
Expected: PASS (all)

- [ ] **Step 5: Wire into the pipeline behind the flag**

In `forecasting/raw_data_forecaster.py`, locate where OAD features are joined (search `add_oad_features`). Immediately after that block add:

```python
import os
if os.environ.get("DATECT_USE_TRANSPORT_FEATURES", "").lower() in ("1", "true", "yes"):
    from forecasting.transport_features import add_transport_features
    from forecasting.oad_features import _load_region_scores  # reuse existing loader
    _region_scores = _load_region_scores()   # date, region, score (mae070)
    final_data = add_transport_features(final_data, region_scores=_region_scores)
```

> If `oad_features.py` has no public region-scores loader, add a thin `_load_region_scores()`
> there that reads `data/processed/oad_scores.parquet` and returns `[date, region, score]`,
> and import it. Keep the loader in ONE place (DRY).

In `per_site_models.py`, near the existing `OAD_FEATURES_ALL` usage:

```python
import os as _os2
if _os2.environ.get("DATECT_USE_TRANSPORT_FEATURES", "").lower() in ("1", "true", "yes"):
    from forecasting.transport_features import TRANSPORT_FEATURES_ALL
    # append to whatever global/default feature list this module exposes
    # (mirror exactly how OAD_FEATURES_ALL is appended above this line)
```

- [ ] **Step 6: Leak-audit smoke test (engine path)**

Run (local, tiny):
```bash
DATECT_USE_TRANSPORT_FEATURES=true DATECT_MIN_TRAINING_FOR_TUNING=99999 \
.venv/bin/python -c "
import warnings; warnings.filterwarnings('ignore')
import config
from forecasting.raw_forecast_engine import RawForecastEngine
e = RawForecastEngine(validate_on_init=True)
df = e.run_retrospective_evaluation(task='regression', model_type='ensemble',
        n_anchors=20, min_test_date='2021-01-01')
print('rows:', len(df), 'cols ok:', 'predicted_da' in df.columns)
"
```
Expected: runs without `verify_no_data_leakage` errors; prints a row count > 0.

- [ ] **Step 7: Commit**

```bash
git add forecasting/transport_features.py forecasting/raw_data_forecaster.py \
        forecasting/oad_features.py per_site_models.py tests/test_transport_features.py
git commit -m "feat(transport): frame integrator + DATECT_USE_TRANSPORT_FEATURES flag"
```

---

## Task 5: Multi-seed ablation harness (Hyak)

**Files:**
- Create: `scripts/eval/transport_ablation.py`
- Create: `scripts/eval/transport_ablation.sbatch`

- [ ] **Step 1: Write the per-seed runner**

Model it on `scripts/eval/multi_seed_baseline.py` (subprocess + env-var pattern). One seed runs
three arms on the **validation window** and writes a per-seed parquet.

```python
# scripts/eval/transport_ablation.py
"""Per-seed transport ablation on the 2019-2022 validation window.

Arms (env overrides per subprocess):
  no_oad   : DATECT_USE_OAD_FEATURES=false, DATECT_USE_TRANSPORT_FEATURES=false
  naive_oad: OAD on,  transport off   (the prior ~0 result)
  l1       : OAD on,  transport on    (this plan)

Always: DATECT_MIN_TRAINING_FOR_TUNING=99999 (disable inner per-anchor tuning).
Usage: python scripts/eval/transport_ablation.py --seed 42
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

VAL_START, VAL_END = "2019-01-01", "2022-01-01"
ARMS = {
    "no_oad":    {"DATECT_USE_OAD_FEATURES": "false", "DATECT_USE_TRANSPORT_FEATURES": "false"},
    "naive_oad": {"DATECT_USE_OAD_FEATURES": "true",  "DATECT_USE_TRANSPORT_FEATURES": "false"},
    "l1":        {"DATECT_USE_OAD_FEATURES": "true",  "DATECT_USE_TRANSPORT_FEATURES": "true"},
}
RUNNER = '''
import warnings; warnings.filterwarnings("ignore")
import os, pandas as pd, config
config.RANDOM_SEED = int(os.environ["SEED"])
from forecasting.raw_forecast_engine import RawForecastEngine
e = RawForecastEngine(validate_on_init=False)
df = e.run_retrospective_evaluation(task="regression", model_type="ensemble",
        n_anchors=int(os.environ.get("N_ANCHORS","500")), min_test_date="2008-01-01")
df = df.dropna(subset=["predicted_da","actual_da"])
df["date"] = pd.to_datetime(df["date"])
df = df[(df["date"] >= os.environ["VAL_START"]) & (df["date"] < os.environ["VAL_END"])]
df.to_parquet(os.environ["OUT"], index=False)
print("wrote", os.environ["OUT"], "rows", len(df))
'''

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-anchors", type=int, default=500); a = ap.parse_args()
    outdir = Path("eval_outputs/transport_ablation"); outdir.mkdir(parents=True, exist_ok=True)
    for arm, flags in ARMS.items():
        out = outdir / f"{arm}_seed{a.seed}.parquet"
        env = {**os.environ, **flags, "SEED": str(a.seed), "N_ANCHORS": str(a.n_anchors),
               "VAL_START": VAL_START, "VAL_END": VAL_END, "OUT": str(out),
               "DATECT_MIN_TRAINING_FOR_TUNING": "99999"}
        print(f"[seed {a.seed}] arm={arm} -> {out}")
        subprocess.run([sys.executable, "-c", RUNNER], env=env, check=True)

if __name__ == "__main__":
    main()
```

> **Note:** if `DATECT_USE_OAD_FEATURES` does not yet exist, add it in `config.py` mirroring the
> existing `DATECT_USE_PER_SITE_MODELS` flag (default the prior behavior), and gate the
> `add_oad_features` call on it. One-line change; keeps the ablation honest.

- [ ] **Step 2: Write the sbatch (ckpt partition, array over seeds)**

```bash
# scripts/eval/transport_ablation.sbatch
#!/bin/bash
#SBATCH --job-name=transport_abl
#SBATCH --partition=ckpt --account=stf-ckpt --requeue
#SBATCH --array=0-4
#SBATCH --cpus-per-task=8 --mem=32G --time=8:00:00
#SBATCH --output=logs/transport_abl_%A_%a.out --error=logs/transport_abl_%A_%a.err
SEEDS=(42 43 44 45 46)
PY=/gscratch/stf/ac283/envs/datect_scratch/bin/python
$PY scripts/eval/transport_ablation.py --seed ${SEEDS[$SLURM_ARRAY_TASK_ID]}
```

- [ ] **Step 3: Local smoke (tiny, do NOT run the full thing locally)**

Run:
```bash
.venv/bin/python scripts/eval/transport_ablation.py --seed 42 --n-anchors 15
```
Expected: writes `eval_outputs/transport_ablation/{no_oad,naive_oad,l1}_seed42.parquet`.

- [ ] **Step 4: Commit**

```bash
git add scripts/eval/transport_ablation.py scripts/eval/transport_ablation.sbatch config.py
git commit -m "feat(eval): multi-seed transport ablation harness (val window)"
```

- [ ] **Step 5: Launch on Hyak** (manual; record the job id)

```bash
ssh klone-login 'cd <repo> && sbatch scripts/eval/transport_ablation.sbatch'
```

---

## Task 6: Aggregate + GO/NO-GO verdict

**Files:**
- Create: `scripts/eval/transport_ablation_aggregate.py`

- [ ] **Step 1: Write the aggregator**

```python
# scripts/eval/transport_ablation_aggregate.py
"""Aggregate per-seed transport-ablation parquets -> ΔR²/Δspike-recall + verdict.

GO if L1 beats naive_oad by more than the seed noise floor on EITHER R² or spike
recall (mean delta > 1 std of the per-seed deltas, and mean delta > 0). Otherwise
NO-GO -> pivot to the C-led (calibrated spike) paper per the spec gate rule.
"""
from __future__ import annotations
import glob
import numpy as np, pandas as pd
from sklearn.metrics import r2_score
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.eval.spike_detection_eval import compute_spike_metrics  # reuse existing

SPIKE_THRESHOLD = 20.0
ARMS = ["no_oad", "naive_oad", "l1"]
D = Path("eval_outputs/transport_ablation")

def _metrics(df):
    r2 = r2_score(df["actual_da"], df["predicted_da"])
    sm = compute_spike_metrics(df["actual_da"].values, df["predicted_da"].values, SPIKE_THRESHOLD)
    return r2, sm["recall"], sm["f2"]

def main():
    seeds = sorted({int(p.split("seed")[-1].split(".")[0]) for p in glob.glob(str(D / "*_seed*.parquet"))})
    rows = []
    for s in seeds:
        rec = {"seed": s}
        for arm in ARMS:
            df = pd.read_parquet(D / f"{arm}_seed{s}.parquet")
            rec[f"{arm}_r2"], rec[f"{arm}_recall"], rec[f"{arm}_f2"] = _metrics(df)
        rows.append(rec)
    t = pd.DataFrame(rows)
    dr2 = t["l1_r2"] - t["naive_oad_r2"]
    drec = t["l1_recall"] - t["naive_oad_recall"]
    print(t.round(3).to_string(index=False))
    print(f"\nΔR²  (L1 - naive): mean={dr2.mean():+.3f} std={dr2.std():.3f}")
    print(f"Δrec (L1 - naive): mean={drec.mean():+.3f} std={drec.std():.3f}")
    go = ((dr2.mean() > dr2.std() and dr2.mean() > 0) or
          (drec.mean() > drec.std() and drec.mean() > 0))
    print("\nVERDICT:", "GO -> proceed to L2 (Phase 2)" if go else
          "NO-GO -> pivot to C-led calibrated-spike paper (spec gate rule)")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run after Hyak job completes**

Run: `.venv/bin/python scripts/eval/transport_ablation_aggregate.py`
Expected: a per-seed table + ΔR²/Δrecall mean±std + a GO/NO-GO line.

- [ ] **Step 3: Commit**

```bash
git add scripts/eval/transport_ablation_aggregate.py
git commit -m "feat(eval): transport ablation aggregator + GO/NO-GO verdict"
```

---

## Phase 1 self-review checklist (run before handing off)

- [ ] All tests pass: `.venv/bin/python -m pytest tests/test_transport_geometry.py tests/test_transport_features.py -v`
- [ ] Leak-safety test (`test_transport_feature_is_leak_safe`) is GREEN — this is the gate's integrity.
- [ ] Engine smoke (Task 4 Step 6) runs `verify_no_data_leakage` clean with the flag on.
- [ ] Ablation harness produces all three arms × 5 seeds on Hyak.
- [ ] Verdict printed. **Record it in the spec** before starting Phase 2.

---

## Roadmap: Phases 2–6 (each its own plan, gated)

> Expand each into its own `docs/superpowers/plans/` doc **only when its gate clears**.

- **Phase 2 — L2 learned 1-D operator** *(gate: Phase 1 = GO).* Replace the hand-built lag with a
  learned, BEUTI/season-conditioned propagation kernel (attention over region × time-lag).
  Same flag, same harness; add to ablation table. Deliverable: `forecasting/transport_operator.py`.
- **Phase 3 — C conformal head** *(independent of B; can run in parallel).* Split/adaptive
  conformal intervals → calibrated spike thresholds; coverage + spike-recall vs. quantile clipping.
  Deliverable: `forecasting/conformal_head.py`. **This is the C-led pivot target if Phase 1 = NO-GO.**
- **Phase 4 — D benchmark** *(after L2).* Freeze splits, document the leak-safe protocol, publish
  per-site + pooled metrics. Deliverable: `docs/benchmark/`.
- **Phase 5 — Causal mediation (spine)** *(after L2).* Decompose offshore→DA total effect into
  direct vs. transport-mediated; report mediated effect size + CI. Deliverable:
  `scripts/eval/causal_mediation.py`.
- **Phase 6 — L4 neural-ODE velocity field (crown jewel)** *(gate: L3 working).* Learned ocean
  velocity field + differentiable Lagrangian advection (`torchdiffeq`), trained end-to-end vs DA;
  validate the learned field against PNW circulation. Deliverable: `forecasting/transport_neural_ode.py`.
  Stretch within Phase 6: generative diffusion/flow nowcast head (`forecasting/generative_head.py`).

**Not in scope (roadmap only):** external surface currents (OSCAR/HF-radar), 2-D FNO operator,
joint Pn+DA multi-task, transferable backbone.
