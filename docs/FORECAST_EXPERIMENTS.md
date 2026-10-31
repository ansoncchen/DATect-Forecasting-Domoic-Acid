# Forecast audit and experiments (exploratory)

Branch `claude/forecast-experiments-b1c907`, starting from `def401da` (holdout R² 0.497,
validation 0.324). **Nothing here is adopted.** This page records problems found in the
one-week-ahead DA forecast and many small one-change-at-a-time experiments, so the
winners can be ported to the main line. `docs/CORRECTED_NUMBERS.md` stays the source of
reported numbers.

<!-- SUMMARY -->

## 1. Audit: problems found

Each issue was confirmed in code and with data. Severity is about the forecast and the
claims made for it.

### A1. High: forecast rows see older persistence information than training rows (train/forecast skew)

Training row for target week *T* (Monday): every lag and rolling feature uses
measurements strictly before *T*, so information ends 0–6 days before the target
measurement. Forecast row for a measurement on day *D* (anchor *A* = *D* − 7):
`get_site_anchor_row` takes the last weekly row *W* whose week ends by *A*
(*W* + 6 ≤ *A*), and `recompute_test_row_persistence_features` refreshes only
`last_observed_da_raw` and `weeks_since_last_spike`. The other 14 persistence
columns (`da_raw_prev_obs_1..4`, `_2/_3_weeks_ago`, `_diff_1_2`, all
`raw_obs_roll_*`) are row *W*'s, built from measurements strictly before *W*, i.e.
13–19 days before *D*, and the "weeks ago" values are measured from *W*, not from *D*.

Evidence on the 635 validation points (`results/experiments/` scratch check, reproducible
with `harness.py`):

| Check | Value |
|---|---:|
| Weeks between the target week's Monday and *W* | 2 for 581 points, 1 for 54 |
| Points with ≥ 1 real measurement in [*W*, *A*] that the lag/rolling features ignore | 73.5% |
| Points where `da_raw_prev_obs_1` ≠ the latest measurement at the anchor | 51.5% (mean gap 5.2 µg/g) |
| Training rows where `da_raw_prev_obs_1` = `last_observed_da_raw` | 100% (4,447 of 4,447) |

So the model learns `prev_obs_1 == last_observed` and short lags, and is then fed
two-week-old lags that disagree with `last_observed` half the time. This is the same
class of bug as `e14db2eb`, one step further along. The pooled spike classifier is hit
hardest: its top features are `raw_obs_roll_max_4/8/12` (57% of gain).
The fix tested here (`asof`) builds every training row exactly like a forecast row:
one row per real measurement on day *d*, with features as of *d* − 7 (same feature
definitions; checked equal to the feature frame on 1,500 rows when the cutoff is
moved back to the row's Monday, and leak-tested by corrupting every post-anchor value).

### A2. High (paper and docs), measured below: the spike classifier's safe-baseline filter does nothing

`ClassificationAdapter.train_spike_binary_classifier` keeps rows with
`da_raw_prev_obs_1 < 20` ("it must learn onsets, not persistence";
`docs/PIPELINE_DEEP_DIVE.md`, paper §"Transition-Focused Binary Classifier"). It
receives the pooled matrix *after* `MinMaxScaler` (`_build_pooled_spike_data` returns
`transformer.fit_transform(...)`), where every value is in [0, 1]. At an anchor of
2020-06-01 the filter keeps 3,724 of 3,724 rows; applied to unscaled values it would keep
86.4% and only 149 of the 506 spikes (the rest are sustained highs). `da_raw_prev_obs_1`,
equal to the last measurement in training rows, is never dropped either. So the
classifier that runs is a general spike classifier with persistence features, and the
paper's statement that environmental features matter little "once persistence features
are excluded by the safe-baseline filter" describes a model that is not the one evaluated.

### A3. Medium: the clip bound comes from targets that are mostly gap-filled

`_postprocess` caps XGBoost and RF at `np.quantile(train_data["da_raw"], q)` with
per-site *q* = 0.95–0.97, but `train_data["da_raw"]` has been filled with the
decay-extrapolated `da` (59% of training rows at a 2019 anchor; 17–61% of gap-filled rows
are ~0). The cap is therefore far below the real-measurement quantile:

| Site (anchor 2019-01-01) | gap-filled share | q95 all targets | q95 real only | max real |
|---|---:|---:|---:|---:|
| Coos Bay | 79% | 81 | 145 | 240 |
| Kalaloch | 59% | 38 | 55 | 126 |
| Quinault | 62% | 25 | 31 | 110 |
| Clatsop Beach | 46% | 40 | 43 | 140 |

Validation consequence: on "sustained" rows (last measurement ≥ 20 and target ≥ 20) the
ensemble averages 30.4 against 44.5 observed (persistence 46.2). Clatsop Beach alone is
39% of validation squared error, with bias −10.4 µg/g and a largest forecast of 48.6
against 12 measurements ≥ 40.

### A4. Medium: strong shrinkage of high values (under-dispersion)

The forecast SD is 0.54 (validation) / 0.62 (holdout) of the observed SD; the slope of
observed on forecast is 1.08 / 1.16 (1.05–1.29 over the five multi-seed runs). Rows with
actual ≥ 40 are under-predicted by 47 / 44 µg/g on average and carry 80% / 74% of the
squared error. Onsets average 10.3 predicted vs 42.9 observed (validation).

### A5. Medium (evaluation): the pooled R² rests on a handful of rows and on one site

| | Validation | Holdout |
|---|---:|---:|
| R² | 0.324 | 0.497 |
| Share of squared error in the top 10 rows | 58% | 61% |
| R² without the 5 worst rows | 0.426 | 0.560 |
| R² without Coos Bay | 0.325 | **0.292** |
| R² without Newport | 0.341 | 0.614 |
| R² without Gold Beach | 0.410 | 0.544 |
| R² on log1p scale (naive) | 0.326 (0.317) | 0.380 (0.262) |

The holdout headline depends on Coos Bay's 35 rows (mean 34 µg/g, R² 0.67): without them
it is 0.29. The 0.17 gap between validation and holdout is mostly which site had extreme
events in which window (Clatsop and Gold Beach in 2019–21, Coos Bay and Newport in
2022–23). Per-site R² and a log-scale R² should accompany the pooled number. Checked and
fine: causal site and site-month climatologies score R² ≈ 0 (−0.08/−0.10 validation,
0.06/0.08 holdout), so the pooled R² is not between-site mean variance; the history filter
(which counts each site's total, future measurements included) excludes no window point.

### A6. Low–medium: environmental features are concurrent with the target in training rows

Training row *T* carries the PN maximum of week *T* itself (a PN sample exists in the
same week for 62.5% of DA weeks, the same day for 23.9%), BEUTI and discharge of
Monday *T*, OAD scores to *T* − 5 and satellite to *T* − 7. The forecast row carries the
same columns from *W*, 1–2 weeks older. A training-only nowcast relationship, not
test-time leakage. Pooled Spearman of PN with DA is ~0.05 at lags 0–3 weeks, so the
damage is probably small (`no_pn`, `asof` measure it).

### A7. Low: smaller data-availability issues

| Issue | Evidence | Effect on the forecast |
|---|---|---|
| ONI publication lag | ONI is merged as month − 2. The stored values are consistent with the centre-month convention (month 2016-05 = 0.49, 2016-06 = 0.00, 2016-07 = −0.31, as the AMJ/MJJ/JJA 2016 ONI), so month *m* − 2 needs SST through *m* − 1, published around the 10th of month *m*: rows in roughly the first third of each month use a value not yet public. | `no_oni` |
| One Columbia River gauge (USGS 14246900) for all sites | In the feature subsets of Copalis, Kalaloch, Twin Harbors, Quinault, Long Beach and Clatsop (all features); absent at the southern Oregon sites. | `no_discharge` |
| OAD autoencoder trained on all years | Score windows end at *R* − 5 (no time leak) and the DOY z-score uses prior years only, but the encoder saw 2022–23 imagery. | `no_oad` |
| Satellite NaNs | `modis-flr` 41%, `modis-sst` 12% NaN, median-imputed with no missingness flag; the as-of merge has no tolerance. FLH is used only at Clatsop (all features). | `no_sat` |
| BEUTI | Merged as the single Monday value (not a weekly mean). Kalaloch (47.59°N) and Quinault (47.28°N) lie north of the 47°N bin, so IDW is dominated by 47°N. Kalaloch's subset excludes BEUTI, so its extrapolation has no effect. | negligible |
| No "age of last measurement" feature | `weeks_ago` exists only for lags 2–3; the model cannot tell a 1-week-old from a 6-month-old `last_observed_da_raw` (age at the anchor: median 7 days, max 924), although gap-filled targets are a deterministic decay of exactly that age. | `obs_age` |
| Weekly-max training target vs daily test target | 11.5% of real weeks have more than one sampling day. | `asof` uses daily targets |

### A8. Low: housekeeping

- `chronological_eval.py` runs every eligible point 2003–2023 (~5,000) and then keeps the
  window (635 or 409): ~8× the needed compute. `harness.py` runs only the window.
- The saved parquet overwrites `predicted_da` (XGBoost) with the ensemble, so the XGBoost
  component is lost at sites whose XGBoost weight is 0.

## 2. Diagnostics of the committed predictions

`scripts/experiments/diagnostics.py` on `results/chronological/*.parquet`; CSVs in
`results/experiments/diagnostics/`. "Regime" uses the last measurement at the anchor
(the naive forecast) and the target: quiet (< 20 → < 20), onset (< 20 → ≥ 20),
sustained (≥ 20 → ≥ 20), decline (≥ 20 → < 20).

**By regime** (mean actual / ensemble / persistence, share of squared error):

| Regime | Validation N | actual | ensemble | naive | SSE share | Holdout N | actual | ensemble | naive | SSE share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| quiet | 486 | 4.0 | 4.7 | 4.2 | 6% | 292 | 5.9 | 6.1 | 5.9 | 7% |
| onset | 34 | 42.9 | 10.3 | 8.8 | 48% | 27 | 36.6 | 12.5 | 8.1 | 30% |
| sustained | 80 | 44.5 | 30.4 | 46.2 | 39% | 57 | 47.4 | 34.4 | 46.1 | 51% |
| decline | 35 | 11.0 | 25.2 | 37.5 | 7% | 33 | 10.7 | 24.1 | 41.1 | 13% |

The model earns its skill over persistence on declines (and quiet weeks at some sites);
it loses to persistence on sustained highs, which the clip and shrinkage pull down; and
neither forecasts onsets.

**By magnitude of the actual value:** `[0,5)` is over-predicted (+2.7 validation, +4.6
holdout, which is why 4-category accuracy fell after the leakage fixes); `[40,∞)` is
under-predicted by 47 / 44 µg/g and holds 80% / 74% of the squared error.

**By site (validation):** Clatsop Beach (bias −10.4, R² 0.24 vs persistence 0.51) and
Gold Beach (R² 0.09) carry 39% and 37% of the squared error. Persistence beats the
ensemble at Clatsop, Coos Bay, Kalaloch, Long Beach and Quinault in validation, and the
ensemble beats persistence at Copalis, Twin Harbors, Gold Beach and Newport.

**By season:** winter (DJF) holds 52% of validation squared error (bias −4.9); autumn
(SON) holds 67% of holdout squared error (bias −5.4).

**Calibration:** binned by forecast, the observed mean exceeds the forecast in every bin
(e.g. forecast 5–10 → observed 10.5 in validation), consistent with A4.

**Components on identical rows** (ΔR² vs the ensemble, paired 95% CI):

| Model | Validation R² | ΔR² | Holdout R² | ΔR² |
|---|---:|---|---:|---|
| Ensemble | 0.324 | — | 0.497 | — |
| Random Forest | 0.342 | +0.018 [−0.047, +0.079] | 0.473 | −0.024 [−0.071, +0.013] |
| Ridge | 0.349 | +0.025 [−0.021, +0.089] | 0.415 | −0.082 [−0.145, −0.025] |
| Persistence | 0.213 | −0.111 [−0.501, +0.114] | 0.225 | −0.271 [−0.679, −0.024] |

Hybrid alert (committed): validation recall 0.912, precision 0.495, transition recall
0.677 (21/31); holdout 0.893 / 0.381 / 0.690 (20/29).

## 3. Experiments

<!-- EXPERIMENTS -->

## 4. Shortlist and one-time holdout check

<!-- HOLDOUT -->

## 5. Reproducing

<!-- REPRO -->
