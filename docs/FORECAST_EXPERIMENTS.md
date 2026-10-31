# Forecast audit and experiments (exploratory)

Branch `claude/forecast-experiments-b1c907`, starting from `def401da` (holdout R² 0.497,
validation 0.324). **Nothing here is adopted.** This page records problems found in the
one-week-ahead DA forecast and many small one-change-at-a-time experiments, so the
winners can be ported to the main line. `docs/CORRECTED_NUMBERS.md` stays the source of
reported numbers.

## Summary

**Objective.** The user made onset detection primary: forecasting the week DA first
crosses 20, scored on rows whose last known value at the anchor is < 20. The level
forecast (R², MAE) is secondary, and the user wants one model rather than per-site
picking. Selection used validation (2019–2021) only, with 2015–2018 as a consistency check;
the 2022–2023 holdout was run once for a shortlist fixed in advance and for the
pre-declared pooled comparison.

**Top problems** (§1):

| | Problem | Severity | Measured effect |
|---|---|---|---|
| A3 | The clip bound is a quantile of training targets that are ~59% gap-filled, so it caps forecasts far below real highs (Coos Bay q95 81 vs 145) | **High for the level forecast** | No clip: ΔR² +0.102 / +0.090 / **+0.053** (val / dev / holdout); quantile over real rows: +0.017 / +0.057 / **+0.046** |
| A1 | Forecast rows reuse the last complete week's lags (13–19 days old); training rows see measurements up to their own week; `da_raw_prev_obs_1` = last value in 100% of training rows, 49% of forecast rows | High as a correctness issue, small in skill | Refreshing the forecast-row lags: R² tie; onset AUC +0.010 / +0.018 / +0.015 for the classifier |
| A2 | The classifier's documented safe-baseline filter runs on MinMax-scaled values and keeps every row | High for the paper (the described model is not the evaluated one) | Applying it as documented *hurts*: ΔAUC −0.023 [−0.046, −0.002] / −0.004; fix the text, not the code |
| A5 | Holdout R² 0.497 is 0.292 without Coos Bay; the top 10 rows hold ~60% of squared error | Medium (reporting) | — |
| A4 | Forecast SD is 0.54–0.62 of observed; ≥ 40 µg/g under-predicted by ~45; onsets forecast at ~10 vs ~40 observed | Medium | Mostly the clip plus MSE shrinkage; losses and transforms do not fix it |
| A6–A7 | Concurrent PN/env in training rows, ONI publication lag, one discharge gauge, all-years OAD encoder, satellite NaNs, BEUTI | Low | Dropping each group is a tie (ΔR² within ±0.02, Δ onset AUC within ±0.01) |

**Worth porting** (numbers are ΔR² or Δ onset AUC, validation / 2015–2018 / holdout):

1. **Drop the quantile clip (S2), or compute it over real rows wherever a quantile stays (S1).**
   S2: +0.102 / +0.090 / +0.053 R², MAE −0.64 / −0.86 / −0.22. S1: +0.017 [+0.004, +0.030]
   / +0.057 / +0.046. The parent's Spark re-tune reaches the same place (7 of 10 sites chose
   no quantile clip).
2. **Recompute the forecast row's lag and rolling features from every measurement up to
   the anchor (A1, test side only), at least for the spike classifier (S4).** Onset AUC
   +0.010 / +0.018 / +0.015. The hybrid alert catches 27 vs 24 (val) and 19 vs 18 (holdout)
   onsets at the same or a lower false-alarm rate; holdout F2 0.704 → 0.717 and
   transition recall 0.69 → 0.72. Low risk: a test-row change, no new features.
3. **For "one model": keep the existing pooled spike classifier as the single onset
   model.** It already is one XGBoost across all sites with one hyperparameter set, and it
   is best or tied in every window. A pre-declared comparison of pooled linear / RF / XGBoost
   (§3b) found no significant family difference; the rule's pick, pooled logistic
   regression, ties it on 2015–2018 (0.736 vs 0.736) and the holdout (0.744 vs 0.758).
4. **Correct the paper and `PIPELINE_DEEP_DIVE.md` description of the classifier (A2)** and
   report onset metrics plus per-site and log-scale R² next to the pooled R² (A5).

**Not worth porting:** log1p / sqrt / Tweedie / Poisson / Huber / change targets (worse,
up to −0.19 onset AUC); online bias correction, persistence blending and scaling (they
compensate for the clip and stop helping once it is gone); horizon-consistent training
rows (`asof`) with or without cross-site and raw-PN features (S5: no holdout gain, +0.39
MAE); recency weights, shorter windows, monotone constraints, per-site feature subsets,
hyperparameter nudges and feature-group drops (ties). The safe-baseline filter fix (hurts).

**One model for the level forecast is not settled.** Pooled RF without a clip ties the
unclipped per-site ensemble on validation (+0.011 to +0.028) and trails it on the holdout
(−0.079 [−0.263, +0.055]). The parent's re-tune shows that per-site *weights* do not
matter; whether per-site *models* matter cannot be resolved with these windows.

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
**Measured:** refreshing the forecast row's lags (`refresh`, `cls_refresh`) is an R² tie
and a small, consistent onset gain (classifier ΔAUC +0.010 / +0.018 / +0.015 on validation
/ 2015–2018 / holdout); fully horizon-consistent training rows (`asof`) are an R² tie with
inconsistent onset effects (§3). The skew is real but the models lean on
`last_observed_da_raw`, which is already fresh.
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
**Measured:** applying the filter as documented lowers onset AUC (−0.023 [−0.046, −0.002]
validation, −0.004 on 2015–2018), so the fix is to describe what runs, not to change it.

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

**Protocol.** `scripts/experiments/harness.py` re-implements the engine's single-point
forecast with one switch per experiment; on Hyak its baseline reproduces the committed
validation predictions exactly (max |difference| 0.0 over 635 rows), and its classifier
reproduces the committed `spike_probability` (ΔAUC 0.000). Every experiment changes one
thing against the committed engine and is scored on identical rows with a paired
bootstrap (2,000 resamples): validation `[2019-01-01, 2022-01-01)` (635 points, 114
spikes, 34 onsets) for selection, and 2015–2018 (942 points, 74 onsets) as a consistency
check. The per-site configs were curated on pre-2019 data, so 2015–2018 is not independent
of them. Retrained experiments are also compared with both runs unclipped, because the clip
(A3) interacts with almost everything. Onsets follow the definition the user asked for:
rows whose last known value at the anchor is < 20 are candidates, and onsets are the
candidates measured ≥ 20.

**Noise floor.** Re-running the baseline with model seeds 43 and 44 moves ΔR² by up to
0.007 and onset AUC by up to 0.008, and on 2015–2018 one seed's row-bootstrap CI excludes
zero. The row bootstrap ignores model-seed variance, so the mechanical verdicts below
require |Δ| ≥ 0.01 as well as a CI that excludes zero. A verdict of "better" also needs
the validation CI to exclude zero with 2015–2018 agreeing in sign; "lean" needs |Δ| ≥ 0.02
in both windows. Raw outputs: `/gscratch/scrubbed/ac283/experiments/raw/` (Hyak); summaries:
`results/experiments/summary_{val,dev}.csv` and `summary_{val,dev}_hybrid.csv`.

**Post-processing (rescoring of the baseline run; no retraining)**

| Experiment | Val ΔR² [95% CI] | Val ΔR², both unclipped | Dev ΔR² [95% CI] | Val MAE (Δ) | Val F2 | Val trans. recall | Val Δ onset AUC [95% CI] | Dev Δ onset AUC | R² verdict | Onset verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| Clip off | +0.102 [-0.029, +0.207] | same | +0.090 [+0.030, +0.149] | 6.55 (-0.64) | 0.734 | 0.39 | +0.000 [+0.000, +0.000] | +0.006 | lean better | tie |
| Clip quantile over real rows only | +0.017 [+0.004, +0.030] | same | +0.057 [+0.007, +0.102] | 7.14 (-0.05) | 0.734 | 0.39 | +0.000 [+0.000, +0.000] | +0.002 | **better** | tie |
| Clip at q = 0.99 everywhere | +0.092 [-0.026, +0.188] | same | +0.082 [+0.027, +0.133] | 6.68 (-0.51) | 0.734 | 0.39 | +0.000 [+0.000, +0.000] | +0.006 | lean better | tie |
| Weights 50/50 at every site (diagnostic) | +0.019 [-0.013, +0.053] | same | +0.001 [-0.014, +0.017] | 7.04 (-0.15) | 0.737 | 0.42 | +0.010 [-0.043, +0.080] | +0.009 | tie | tie |
| RF only at every site (diagnostic) | +0.018 [-0.047, +0.079] | same | +0.016 [-0.004, +0.036] | 7.02 (-0.17) | 0.751 | 0.48 | -0.006 [-0.069, +0.069] | +0.014 | tie | tie |
| 10% persistence blend | +0.024 [+0.001, +0.045] | same | +0.026 [+0.006, +0.045] | 7.05 (-0.14) | 0.733 | 0.39 | +0.001 [-0.011, +0.014] | -0.003 | **better** | tie |
| 30% persistence blend | +0.049 [-0.030, +0.113] | same | +0.037 [-0.034, +0.095] | 6.83 (-0.36) | 0.761 | 0.45 | -0.013 [-0.046, +0.018] | -0.013 | lean better | tie |
| Scale forecasts ×1.2 | +0.009 [-0.029, +0.035] | same | +0.029 [+0.007, +0.046] | 7.63 (+0.44) | 0.740 | 0.48 | +0.000 [+0.000, +0.000] | +0.000 | tie | tie |
| Add 10 × P(spike) | +0.020 [-0.033, +0.051] | same | +0.041 [+0.021, +0.059] | 7.67 (+0.47) | 0.760 | 0.52 | +0.057 [+0.019, +0.113] | +0.026 | lean better | **better** |
| Add 20 × P(spike) | -0.047 [-0.190, +0.031] | same | +0.030 [-0.018, +0.070] | 8.73 (+1.53) | 0.776 | 0.58 | +0.076 [+0.028, +0.139] | +0.037 | tie | **better** |
| Online per-site bias (additive, last 5 verified) | +0.054 [-0.014, +0.113] | same | +0.070 [+0.032, +0.101] | 7.02 (-0.17) | 0.741 | 0.42 | -0.001 [-0.069, +0.074] | -0.003 | lean better | tie |
| Online per-site bias (multiplicative, last 10) | -0.103 [-0.453, +0.091] | same | +0.058 [-0.034, +0.129] | 7.50 (+0.31) | 0.752 | 0.42 | -0.009 [-0.036, +0.011] | -0.002 | tie | tie |
| Clip over real rows + online bias | +0.059 [-0.017, +0.123] | same | +0.088 [+0.008, +0.149] | 7.00 (-0.19) | 0.741 | 0.42 | -0.001 [-0.069, +0.074] | -0.008 | lean better | tie |
| Clip off + online bias | +0.078 [-0.064, +0.189] | same | +0.094 [+0.003, +0.165] | 6.63 (-0.56) | 0.734 | 0.39 | -0.029 [-0.077, +0.005] | -0.010 | lean better | tie |
| Clip off + 20% persistence | +0.093 [-0.046, +0.201] | same | +0.074 [-0.010, +0.148] | 6.46 (-0.73) | 0.740 | 0.39 | -0.006 [-0.029, +0.017] | -0.003 | lean better | tie |
| Regression alert at 10 (F2 only) | +0.000 [+0.000, +0.000] | same | +0.000 [+0.000, +0.000] | 7.19 (+0.00) | 0.740 | 0.48 | +0.000 [+0.000, +0.000] | +0.000 | tie | tie |
| Regression alert at 14 (F2 only) | +0.000 [+0.000, +0.000] | same | +0.000 [+0.000, +0.000] | 7.19 (+0.00) | 0.718 | 0.32 | +0.000 [+0.000, +0.000] | +0.000 | tie | tie |

**Training rows and features (retrained)**

| Experiment | Val ΔR² [95% CI] | Val ΔR², both unclipped | Dev ΔR² [95% CI] | Val MAE (Δ) | Val F2 | Val trans. recall | Val Δ onset AUC [95% CI] | Dev Δ onset AUC | R² verdict | Onset verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| Forecast-row lags recomputed to the anchor (A1, test side only) | -0.001 [-0.024, +0.019] | -0.003 | -0.010 [-0.024, +0.004] | 7.15 (-0.04) | 0.745 | 0.45 | +0.014 [+0.001, +0.033] | +0.018 | tie | **better** |
| Training rows built like forecast rows (A1), real rows only | +0.037 [-0.034, +0.094] | -0.005 | -0.006 [-0.044, +0.033] | 7.74 (+0.55) | 0.750 | 0.52 | +0.048 [-0.006, +0.116] | -0.003 | tie | tie |
| asof + gap-filled rows | +0.024 [-0.015, +0.061] | +0.012 | -0.019 [-0.040, +0.001] | 7.08 (-0.11) | 0.752 | 0.45 | +0.058 [+0.011, +0.123] | -0.010 | tie | tie |
| Real-measurement rows only (engine rows) | +0.016 [-0.040, +0.056] | -0.013 | +0.029 [-0.023, +0.074] | 7.79 (+0.59) | 0.751 | 0.52 | +0.023 [-0.011, +0.064] | +0.002 | tie | tie |
| Add age of the last measurement | -0.003 [-0.018, +0.008] | -0.004 | -0.005 [-0.010, +0.001] | 7.20 (+0.01) | 0.729 | 0.39 | +0.011 [-0.002, +0.028] | +0.001 | tie | tie |
| Drop observation-order lags 1–4 | -0.003 [-0.025, +0.016] | -0.005 | -0.011 [-0.024, +0.001] | 7.27 (+0.08) | 0.728 | 0.39 | +0.016 [+0.003, +0.034] | -0.001 | tie | tie |
| Add cross-site DA features | +0.007 [-0.002, +0.018] | +0.008 | -0.004 [-0.014, +0.006] | 7.07 (-0.12) | 0.741 | 0.39 | +0.025 [-0.001, +0.055] | -0.018 | tie | tie |
| Add raw-PN lead features | +0.005 [-0.000, +0.010] | +0.004 | -0.005 [-0.015, +0.005] | 7.16 (-0.03) | 0.721 | 0.35 | -0.002 [-0.014, +0.011] | +0.006 | tie | tie |
| asof + cross-site DA | +0.031 [-0.047, +0.095] | -0.001 | +0.002 [-0.041, +0.042] | 7.79 (+0.60) | 0.783 | 0.65 | +0.064 [+0.005, +0.142] | +0.002 | tie | **better** |
| asof + raw-PN | +0.034 [-0.032, +0.088] | -0.005 | +0.004 [-0.046, +0.046] | 7.75 (+0.56) | 0.753 | 0.52 | +0.041 [-0.016, +0.112] | +0.007 | tie | tie |
| asof + cross-site DA + raw-PN | +0.036 [-0.037, +0.098] | +0.004 | +0.016 [-0.038, +0.061] | 7.74 (+0.55) | 0.773 | 0.61 | +0.077 [+0.015, +0.154] | +0.017 | tie | **better** |
| Drop PN | -0.000 [-0.001, +0.001] | +0.000 | -0.001 [-0.002, -0.000] | 7.21 (+0.02) | 0.734 | 0.39 | +0.001 [-0.003, +0.005] | -0.001 | tie | tie |
| Drop the 16 OAD features | +0.009 [-0.011, +0.027] | +0.007 | -0.016 [-0.033, -0.001] | 7.07 (-0.12) | 0.719 | 0.35 | +0.003 [-0.012, +0.019] | -0.008 | tie | tie |
| Drop ONI | +0.002 [-0.000, +0.004] | +0.002 | -0.000 [-0.002, +0.001] | 7.17 (-0.02) | 0.734 | 0.39 | +0.005 [-0.001, +0.011] | +0.002 | tie | tie |
| Drop discharge | +0.001 [-0.001, +0.003] | +0.000 | +0.000 [-0.001, +0.001] | 7.17 (-0.03) | 0.735 | 0.39 | +0.004 [-0.001, +0.010] | -0.000 | tie | tie |
| Drop satellite SST, SST anomaly, FLH | -0.000 [-0.010, +0.008] | -0.002 | -0.003 [-0.007, +0.002] | 7.22 (+0.02) | 0.728 | 0.35 | +0.006 [-0.007, +0.020] | +0.002 | tie | tie |
| Drop every environmental feature | +0.005 [-0.024, +0.035] | +0.004 | -0.018 [-0.036, -0.000] | 7.07 (-0.12) | 0.728 | 0.39 | +0.000 [-0.016, +0.016] | +0.005 | tie | tie |
| All features at every site | +0.009 [-0.019, +0.040] | +0.009 | -0.002 [-0.018, +0.013] | 7.18 (-0.01) | 0.730 | 0.42 | +0.011 [-0.013, +0.040] | +0.012 | tie | tie |
| No monotone constraints | -0.001 [-0.003, +0.002] | -0.000 | +0.001 [+0.000, +0.003] | 7.22 (+0.03) | 0.750 | 0.45 | +0.005 [-0.002, +0.012] | +0.011 | tie | tie |
| No per-site configuration | +0.024 [-0.188, +0.162] | -0.084 | +0.024 [-0.079, +0.104] | 6.73 (-0.47) | 0.719 | 0.45 | +0.003 [-0.083, +0.090] | -0.022 | lean better | tie |

**Targets and losses (retrained)**

| Experiment | Val ΔR² [95% CI] | Val ΔR², both unclipped | Dev ΔR² [95% CI] | Val MAE (Δ) | Val F2 | Val trans. recall | Val Δ onset AUC [95% CI] | Dev Δ onset AUC | R² verdict | Onset verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| log1p target | -0.040 [-0.105, +0.030] | -0.030 | -0.112 [-0.160, -0.062] | 6.75 (-0.44) | 0.659 | 0.23 | -0.091 [-0.164, -0.029] | -0.128 | worse | worse |
| log1p target + smearing | -0.019 [-0.061, +0.029] | -0.010 | -0.087 [-0.126, -0.043] | 7.03 (-0.16) | 0.691 | 0.26 | -0.061 [-0.099, -0.027] | -0.091 | worse | worse |
| sqrt target | -0.016 [-0.056, +0.033] | -0.013 | -0.050 [-0.083, -0.016] | 6.77 (-0.42) | 0.729 | 0.35 | -0.074 [-0.139, -0.018] | -0.081 | worse | worse |
| Tweedie loss (RF Poisson criterion) | -0.002 [-0.027, +0.031] | +0.001 | -0.029 [-0.050, -0.007] | 7.06 (-0.14) | 0.714 | 0.35 | -0.023 [-0.050, +0.001] | -0.049 | worse | worse |
| Poisson loss | +0.000 [-0.035, +0.041] | +0.002 | -0.039 [-0.070, -0.008] | 7.24 (+0.05) | 0.731 | 0.42 | +0.021 [-0.005, +0.049] | -0.033 | tie | tie |
| Pseudo-Huber loss (slope 10) | -0.011 [-0.056, +0.042] | -0.013 | -0.048 [-0.084, -0.011] | 6.79 (-0.40) | 0.730 | 0.32 | -0.051 [-0.107, -0.004] | -0.045 | worse | worse |
| Change target (actual − last known) | -0.071 [-0.140, -0.024] | -0.181 | -0.079 [-0.128, -0.032] | 7.40 (+0.21) | 0.720 | 0.32 | -0.131 [-0.252, -0.026] | -0.189 | worse | worse |
| asof + change target | -0.004 [-0.070, +0.043] | -0.078 | +0.002 [-0.070, +0.067] | 7.58 (+0.38) | 0.749 | 0.45 | -0.040 [-0.108, +0.023] | -0.040 | tie | lean worse |
| Spike rows weighted ×3 | -0.026 [-0.100, +0.022] | -0.028 | +0.016 [-0.009, +0.037] | 7.96 (+0.77) | 0.732 | 0.45 | +0.019 [-0.016, +0.057] | +0.016 | tie | tie |

**Training window, pooling, hyperparameters, noise floor (retrained)**

| Experiment | Val ΔR² [95% CI] | Val ΔR², both unclipped | Dev ΔR² [95% CI] | Val MAE (Δ) | Val F2 | Val trans. recall | Val Δ onset AUC [95% CI] | Dev Δ onset AUC | R² verdict | Onset verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| Recency weights, 5-year half-life | +0.006 [-0.013, +0.020] | +0.004 | -0.005 [-0.013, +0.003] | 7.18 (-0.01) | 0.726 | 0.32 | +0.013 [-0.012, +0.040] | -0.021 | tie | tie |
| Recency weights, 10-year half-life | +0.004 [-0.005, +0.011] | +0.002 | -0.002 [-0.006, +0.003] | 7.17 (-0.02) | 0.727 | 0.35 | +0.013 [-0.003, +0.031] | -0.011 | tie | tie |
| Last 8 years only | +0.025 [-0.013, +0.053] | -0.003 | -0.098 [-0.158, -0.051] | 7.19 (-0.01) | 0.720 | 0.35 | +0.013 [-0.014, +0.045] | -0.032 | tie | tie |
| One XGBoost across sites (engine rows, site one-hot) | -0.036 [-0.119, +0.026] | -0.085 | -0.051 [-0.089, -0.011] | 6.91 (-0.29) | 0.737 | 0.39 | -0.039 [-0.134, +0.042] | -0.111 | worse | worse |
| XGBoost depth 3 | +0.002 [-0.017, +0.017] | -0.000 | -0.004 [-0.011, +0.003] | 7.06 (-0.14) | 0.740 | 0.42 | +0.002 [-0.012, +0.014] | -0.008 | tie | tie |
| XGBoost ×2 trees | -0.019 [-0.047, +0.002] | -0.020 | +0.005 [-0.010, +0.017] | 7.20 (+0.00) | 0.733 | 0.39 | -0.036 [-0.071, -0.007] | -0.013 | tie | worse |
| RF min leaf 1 | -0.001 [-0.006, +0.002] | -0.002 | -0.003 [-0.010, +0.003] | 7.22 (+0.03) | 0.720 | 0.35 | -0.003 [-0.012, +0.005] | -0.021 | tie | worse |
| Model seed 43 (noise floor) | -0.002 [-0.017, +0.010] | -0.001 | -0.003 [-0.007, +0.001] | 7.20 (+0.01) | 0.727 | 0.39 | +0.008 [-0.005, +0.023] | -0.003 | tie | tie |
| Model seed 44 (noise floor) | -0.003 [-0.011, +0.005] | -0.002 | -0.007 [-0.013, -0.002] | 7.18 (-0.01) | 0.735 | 0.39 | -0.006 [-0.016, +0.003] | -0.006 | tie | tie |

**Spike classifier and hybrid alert (classifier ≥ 0.10 OR regression ≥ 12)**

| Classifier | Val onset AUC | Val ΔAUC [95% CI] | Dev ΔAUC [95% CI] | Val onset recall @20% FA | Val hybrid: onsets caught / quiet-week alarms | Dev hybrid: onsets / alarms | Val F2 | Val trans. recall |
|---|---|---|---|---|---|---|---|---|
| Engine (committed) | 0.864 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 0.79 | 24/34 at 14.6% | 55/74 at 23.2% | 0.781 | 0.68 |
| Safe-baseline filter applied (A2) | 0.840 | -0.023 [-0.046, -0.002] | -0.004 [-0.024, +0.017] | 0.74 | 23/34 at 14.8% | 50/74 at 24.0% | 0.773 | 0.65 |
| Forecast-row lags refreshed (A1) | 0.874 | +0.010 [-0.016, +0.038] | +0.018 [-0.006, +0.045] | 0.82 | 27/34 at 14.8% | 54/74 at 23.3% | 0.798 | 0.77 |
| asof rows | 0.806 | -0.058 [-0.130, +0.011] | +0.026 [-0.019, +0.073] | 0.68 | 23/34 at 15.4% | 53/74 at 24.9% | 0.770 | 0.68 |
| asof rows + safe-baseline filter | 0.775 | -0.089 [-0.164, -0.023] | -0.011 [-0.049, +0.028] | 0.62 | 23/34 at 17.1% | 51/74 at 27.4% | 0.761 | 0.68 |
| Onset-only (asof, last < 20) | 0.787 | -0.076 [-0.147, -0.011] | +0.004 [-0.030, +0.040] | 0.62 | 23/34 at 15.0% | 53/74 at 27.8% | 0.772 | 0.68 |
| Onset-only + cross-site DA | 0.802 | -0.061 [-0.133, +0.004] | — | 0.65 | 23/34 at 16.9% | — | 0.762 | 0.68 |
| Onset-only + raw PN | 0.772 | -0.092 [-0.169, -0.029] | — | 0.65 | 23/34 at 15.6% | — | 0.769 | 0.68 |
| Onset-only + cross-site DA + raw PN | 0.785 | -0.078 [-0.159, -0.010] | — | 0.62 | 23/34 at 16.5% | — | 0.764 | 0.68 |
| Engine + cross-site DA | 0.851 | -0.012 [-0.053, +0.020] | — | 0.76 | 25/34 at 17.3% | — | 0.772 | 0.71 |
| asof rows + cross-site DA | 0.778 | -0.086 [-0.165, -0.011] | — | 0.65 | 22/34 at 17.3% | — | 0.753 | 0.65 |

**What the experiments say**

- **Level forecast: the clip is the dominant problem.** No clip gives +0.10 R² on both
  windows, and a q = 0.99 clip gets most of it. Computing the existing quantile over real
  rows (A3) is a smaller fix that is significant in both windows. The online bias
  correction, the persistence blend and the ×1.2 scale help only under the committed clip:
  on top of no clip, bias correction is −0.024 (validation) and +0.004 (2015–2018), and a
  20% persistence blend is −0.009. They compensate for the clip and are not worth porting
  once it is relaxed. This matches the parent's Spark re-tune (commit `e76647bd`,
  `results/tuning_provenance/retune_spark/`): 7 of 10 sites chose no quantile clip, and
  after that per-site weights could not be told apart from one shared rule.
- **Structural changes are ties for R²:** the per-site feature subsets, monotone
  constraints, recency weights, hyperparameter nudges, OAD, ONI, discharge, PN, satellite
  and even all environmental features together all stay inside ±0.02, as in the pre-fix
  ablation. ONI's publication lag, the single discharge gauge and the all-years OAD encoder
  (A7) therefore cannot be moving the forecast.
- **Losses and target transforms make things worse.** log1p, sqrt, Huber, Tweedie and the
  change target all lose R² and, more clearly, onset AUC (−0.05 to −0.19 on 2015–2018):
  compressing the tail removes exactly the signal onsets need.
- **Onsets.** For the regression, horizon-consistent rows plus cross-site DA (and raw PN)
  raise onset AUC on validation (+0.064 to +0.077, CIs exclude 0) and transition recall
  (0.39 → 0.61–0.65), but only +0.002 to +0.017 on 2015–2018, and MAE rises by about 0.55
  because those runs train on real rows only. Refreshing the forecast-row lags (A1) helps
  onset ranking in both windows (+0.014 [+0.001, +0.033] and +0.018) at no cost.
- **Spike classifier.** The engine's pooled classifier is the best single onset ranker on
  validation (AUC 0.864), but its lead over the horizon-consistent and onset-only
  variants (−0.06 to −0.09 on validation) reverses on 2015–2018 (−0.01 to +0.03). With 34
  and 74 onsets these differences are noise. Applying the documented safe-baseline filter
  (A2) slightly *hurts* in both windows (−0.023 [−0.046, −0.002], −0.004), so the fix for
  A2 is the documentation, not the code. Refreshing the classifier's forecast-row lags is
  the only change positive in both windows; on validation the hybrid alert then catches
  27 instead of 24 of 34 onsets at the same false-alarm rate (14.8%).
- **Adding 10 × P(spike) to the level forecast** improves R² and onset ranking in both
  windows (+0.020 / +0.041 R²; +0.057 / +0.026 onset AUC). One number can carry most of
  the alert information without a separate output.
- **The ESP offshore pDA and NEMO mooring sources** from the pre-cleanup experiments were
  not re-tested: their input files (`ChaBa ESP database.xlsx`, `ChaBa_pCO2ALLdata.mat`) no
  longer exist locally or on Hyak.


## 3b. One pooled model: declared comparison

The user asked for one model instead of per-site picking, with the model family chosen
by a small comparison declared in advance. **This subsection was written and committed
before any pooled model was run**; results are reported below it in full, for every
candidate, on validation and on the holdout.

**Rows (same for every candidate).** Pooled across all 10 sites. One training row per
real DA measurement (site-day, daily max), built exactly like a forecast row: features
as of the measurement date − 7 days (`asof` construction, A1). A row is usable at an
anchor once its measurement date is on or before the anchor. No gap-filled rows, no
clipping, no per-site weights, quantiles, feature subsets or hyperparameters.

**Features (same for every candidate).** Imputed by the training median and
MinMax-scaled (the engine's transformer), refit at every anchor.
- **F1 (core):** the 16 persistence features, calendar (sin/cos day of year, month),
  `oni`, `pdo`, `discharge`, `beuti`, `modis-sst`, `sst-anom`, `modis-flr`, `pn`, `pn_log`,
  the 16 OAD features, and a one-hot site indicator. The project's six zero-importance
  columns stay dropped.
- **F2:** F1 plus the cross-site DA features (`nb_*`, `coast_*`) and the raw-PN lead
  features (`pn_last_log`, `pn_max14_log`, `pn_age_days`, `pn_nb_max14_log`, `pn_rise_log`).

**Two outputs per family.**
- **Onset:** P(DA ≥ 20) from a classifier trained only on rows whose last known value
  at their own cutoff is < 20 (the population it is scored on), with balanced class
  weights.
- **Level:** a regressor of DA (µg/g) trained on all rows, floored at 0.

**Candidates.** One fixed hyperparameter set each, project defaults:
1. **Linear:** logistic regression (L2, C = 1, balanced) / Ridge (α = 1, the project's
   `linear_regression_alpha`).
2. **Random Forest:** the project's `rf_base_params` (400 trees, depth 12, leaf 3, max
   features 0.85); classifier with `class_weight="balanced_subsample"`.
3. **XGBoost:** classifier with the project's `spike_classifier_params` (300 trees,
   depth 4); regressor with the global `xgb_base_params` (400 trees, depth 6); no
   monotone constraints.
4. **Baselines:** naive persistence for the level, and the last-value rule (rank
   candidate rows by last known value) for onsets.

The current engine (per-site ensemble plus the committed pooled classifier and hybrid
alert) is shown on the same rows for reference.

**Metrics.** Primary: onset AUC on rows with last known < 20, onset recall at 10% and
20% quiet-week false-alarm rates, and the gain over the last-value rule at the same
alert count. Secondary: level R² [95% CI], MAE, spike F2 (actual ≥ 20, forecast ≥ 12),
and the level forecast's own onset AUC (can one output serve both purposes?).

**Selection rule (validation only).**
1. The family with the highest validation onset AUC wins, unless its paired ΔAUC CI
   against a simpler family includes zero; then the simpler family wins (linear < RF <
   XGBoost).
2. F2 is preferred over F1 only if it raises that family's onset AUC with a paired CI
   that excludes zero.
3. The holdout is run once for every cell and reported in full; it is not used to choose.

**Validation result (recorded before the holdout run).** 635 rows, 520 with last known
< 20, 34 onsets. Source: `results/experiments/pooled_val.csv`, `pooled_val_pairwise.csv`.

| Candidate | Onset AUC | ΔAUC vs last-value [95% CI] | Onset recall @10% / @20% FA | Gain vs last-value @20% | Level R² [95% CI] | MAE | F2 (≥ 12) | Level output as onset score: AUC |
|---|---|---|---|---|---|---|---|---|
| Last-value rule / naive persistence | 0.651 | — | 0.41 / 0.62 | — | 0.213 [−0.214, 0.518] | 6.83 | 0.742 | 0.651 |
| Current engine (classifier / per-site ensemble) | 0.864 | +0.213 [+0.085, +0.346] | 0.56 / 0.79 | +6.0 | 0.324 [0.208, 0.452] | 7.19 | 0.734 | 0.738 |
| F1 linear | 0.740 | +0.089 [−0.040, +0.225] | 0.50 / 0.62 | +0.0 | 0.364 [0.192, 0.522] | 8.03 | 0.728 | 0.712 |
| F1 RF | 0.759 | +0.108 [−0.022, +0.244] | 0.53 / 0.62 | +0.0 | 0.437 [0.218, 0.587] | 7.05 | 0.767 | 0.752 |
| F1 XGBoost | 0.774 | +0.123 [−0.002, +0.255] | 0.50 / 0.62 | +0.0 | 0.378 [0.143, 0.552] | 7.47 | 0.764 | 0.773 |
| F2 linear | 0.773 | +0.122 [+0.009, +0.251] | 0.56 / 0.62 | +0.0 | 0.359 [0.186, 0.523] | 8.31 | 0.731 | 0.779 |
| F2 RF | 0.749 | +0.098 [−0.036, +0.237] | 0.56 / 0.59 | −1.0 | 0.455 [0.249, 0.605] | 6.94 | 0.781 | 0.759 |
| F2 XGBoost | 0.786 | +0.135 [+0.009, +0.268] | 0.50 / 0.62 | +0.0 | 0.427 [0.213, 0.586] | 7.12 | 0.775 | 0.766 |

Pairwise onset ΔAUC on F1: RF − linear +0.018 [−0.023, +0.065], XGBoost − linear
+0.034 [−0.006, +0.077], XGBoost − RF +0.015 [−0.022, +0.058]. F2 − F1: linear +0.033
[−0.006, +0.073], RF −0.009, XGBoost +0.012 (all CIs include 0).

**The declared rule selects F1 linear** (logistic onset classifier + Ridge level): the
three families are statistically tied on onsets, so the simplest wins, and F2 does not
clear its bar.

**Holdout result (run once, after the validation pick was committed).** 409 rows, 319
with last known < 20, 27 onsets. Source: `results/experiments/pooled_holdout.csv`.

| Candidate | Onset AUC | ΔAUC vs last-value [95% CI] | Onset recall @10% / @20% FA | Gain vs last-value @20% | Level R² [95% CI] | MAE | F2 (≥ 12) | Level output as onset score: AUC |
|---|---|---|---|---|---|---|---|---|
| Last-value rule / naive persistence | 0.550 | — | 0.30 / 0.37 | — | 0.225 [−0.267, 0.516] | 7.89 | 0.674 | 0.550 |
| Current engine (classifier / per-site ensemble) | 0.758 | +0.208 [+0.044, +0.371] | 0.33 / 0.52 | +3.9 | 0.497 [0.364, 0.604] | 7.60 | 0.729 | 0.747 |
| **F1 linear (declared pick)** | 0.744 | +0.195 [+0.023, +0.358] | 0.30 / 0.52 | +3.9 | 0.447 [0.286, 0.583] | 8.35 | 0.661 | 0.679 |
| F1 RF | 0.755 | +0.205 [+0.041, +0.366] | 0.37 / 0.56 | +4.8 | 0.471 [0.164, 0.639] | 7.88 | 0.721 | 0.764 |
| F1 XGBoost | 0.799 | +0.250 [+0.091, +0.400] | 0.44 / 0.59 | +5.6 | 0.456 [0.185, 0.619] | 8.17 | 0.712 | 0.760 |
| F2 linear | 0.787 | +0.237 [+0.075, +0.391] | 0.37 / 0.63 | +6.5 | 0.454 [0.297, 0.586] | 8.36 | 0.664 | 0.732 |
| F2 RF | 0.730 | +0.180 [+0.028, +0.334] | 0.41 / 0.52 | +3.9 | 0.468 [0.181, 0.632] | 7.82 | 0.723 | 0.737 |
| F2 XGBoost | 0.790 | +0.240 [+0.087, +0.384] | 0.48 / 0.59 | +5.6 | 0.402 [0.042, 0.600] | 8.41 | 0.713 | 0.745 |

Pairwise onset ΔAUC on the holdout, F1: RF − linear +0.011 [−0.074, +0.095], XGBoost −
linear +0.055 [−0.022, +0.135], XGBoost − RF +0.044 [−0.003, +0.091].

**All three windows** (2015–2018 run with the same code; `results/experiments/pooled_dev.csv`):

| Onset AUC (rows with last known < 20) | Validation (34 onsets) | 2015–2018 (74) | Holdout (27) |
|---|---:|---:|---:|
| Last-value rule | 0.651 | 0.508 | 0.550 |
| Engine regression (per-site ensemble as the score) | 0.738 | 0.729 | 0.747 |
| Engine pooled classifier (committed) | **0.864** | 0.736 | 0.758 |
| Engine pooled classifier, refreshed lags (S4) | **0.874** | **0.754** | 0.773 |
| F1 linear (declared pick) | 0.740 | 0.736 | 0.744 |
| F1 RF | 0.759 | 0.722 | 0.755 |
| F1 XGBoost | 0.774 | 0.737 | **0.799** |
| F2 linear | 0.773 | 0.727 | 0.787 |
| F2 RF | 0.749 | 0.742 | 0.730 |
| F2 XGBoost | 0.786 | 0.742 | 0.790 |

| Level R² | Validation | 2015–2018 | Holdout |
|---|---:|---:|---:|
| Naive persistence | 0.213 | −0.111 | 0.225 |
| Engine, committed (per-site, clipped) | 0.324 | 0.230 | 0.497 |
| Engine, no clip (S2) | 0.426 | 0.320 | **0.550** |
| F1 linear (Ridge) | 0.364 | 0.276 | 0.447 |
| F1 RF | 0.437 | 0.284 | 0.471 |
| F1 XGBoost | 0.378 | 0.261 | 0.456 |
| F2 RF | **0.455** | 0.280 | 0.468 |

Pooled level models against the unclipped per-site ensemble: validation +0.011
[−0.135, +0.147] (F1 RF) and +0.028 [−0.107, +0.161] (F2 RF); holdout −0.079
[−0.263, +0.055] (F1 RF), −0.103 [−0.213, +0.028] (Ridge).

**Reading.**
- **Family choice barely matters for onsets.** On F1, the declared comparison, no
  family-vs-family difference is significant in any window. XGBoost has the highest point
  estimate on validation and the holdout, and ties on 2015–2018; within F2, XGBoost beats
  RF on the holdout (+0.060 [+0.012, +0.108]) but not elsewhere. The declared pick,
  one pooled logistic regression, ties the engine's classifier on 2015–2018 (0.736 vs
  0.736) and on the holdout (0.744 vs 0.758), and trails it on validation (0.740 vs 0.864).
- **The onset half of the engine is already "one model".** The engine's spike classifier
  is a single pooled XGBoost across all sites with one hyperparameter set and uniform
  features. It is the best or tied onset ranker in every window, and refreshing its
  forecast-row lags (S4) makes it the best in two of three. So per-site picking only ever
  lived in the level regression.
- **For the level forecast, one pooled model is not proven equivalent.** It ties the
  unclipped per-site ensemble on validation (RF +0.01 to +0.03) but trails it by about 0.08
  on the holdout, with CIs including 0. The parent's re-tune found that the per-site
  *weights* do not matter once the clip is gone; whether per-site *training* (separate
  models, per-site hyperparameters and feature subsets) matters is not settled by 409 rows.
- **One output or two?** For tree models, the pooled level forecast ranks onsets about as
  well as the dedicated classifier (holdout RF 0.764 vs 0.755, XGBoost 0.760 vs 0.799).
  For the engine, its per-site regression is within 0.01 of its classifier on 2015–2018
  and the holdout, but 0.13 behind on validation. A single level model can rank onsets
  (one output). A calibrated probability for the alert threshold still needs the
  classifier, so two outputs from one feature pipeline is the practical design.


## 4. Shortlist and one-time holdout check

**Shortlist, fixed from validation (with the 2015–2018 window as a consistency check)
and committed before the holdout was run.** Each is checked alone against the
committed engine on the 409 holdout rows.

| # | Change | Why it is on the list (validation; dev) | Kind |
|---|---|---|---|
| S1 | Clip quantile over real rows only | ΔR² +0.017 [+0.004, +0.030]; +0.057 [+0.007, +0.102] | engine fix (rescoring) |
| S2 | No clip | ΔR² +0.102 [−0.029, +0.207]; +0.090 [+0.030, +0.149] | reference for the parent's re-tune (rescoring) |
| S3 | Level forecast + 10 × P(spike) | ΔR² +0.020 / +0.041; Δ onset AUC +0.057 [+0.019, +0.113] / +0.026 | one output for level and onset (rescoring) |
| S4 | Classifier with forecast-row lags refreshed to the anchor (A1) | Δ onset AUC +0.010 / +0.018; hybrid 27 vs 24 of 34 onsets at the same alarm rate | engine fix (classifier) |
| S5 | Regression on horizon-consistent rows + cross-site DA + raw PN | Δ onset AUC +0.077 [+0.015, +0.154] / +0.017; transition recall 0.39 → 0.61 / 0.58 → 0.71; ΔR² tie; MAE +0.55 | engine change (retrain) |

The declared pooled comparison (§3b) is run on the holdout separately, for every cell.

**Holdout results (409 rows, run once).** The harness baseline reproduces the committed
holdout predictions exactly (max |difference| 0.0). Sources:
`results/experiments/holdout_shortlist.csv`, `holdout_shortlist_classifier.csv`.

| # | Change | ΔR² [95% CI] | MAE (Δ) | F2 | Trans. recall | Δ onset AUC [95% CI] | Same direction as validation? |
|---|---|---|---|---|---|---|---|
| S0 | Committed engine | R² 0.497 | 7.60 | 0.729 | 0.55 | AUC 0.747 (regression) | — |
| S1 | Clip quantile over real rows | +0.046 [−0.002, +0.110] | 7.43 (−0.17) | 0.729 | 0.55 | 0 | yes (val +0.017, dev +0.057) |
| S2 | No clip | +0.053 [−0.013, +0.118] | 7.39 (−0.22) | 0.729 | 0.55 | 0 | yes (val +0.102, dev +0.090) |
| S3 | Level + 10 × P(spike) | −0.000 [−0.067, +0.037] | 8.41 (+0.81) | 0.721 | 0.59 | +0.016 [−0.008, +0.046] | partly: onset yes, R² no, MAE worse |
| S5 | Horizon-consistent rows + cross-site DA + raw PN | −0.010 [−0.080, +0.046] (−0.063 both unclipped) | 8.00 (+0.39) | 0.721 | 0.62 | +0.014 [−0.042, +0.081] | no for R²/MAE; onset small |

| # | Spike classifier inside the hybrid alert | Onset AUC | ΔAUC [95% CI] | Hybrid: onsets caught / quiet-week alarms | Spike recall / precision / F2 | Trans. recall |
|---|---|---:|---|---|---|---:|
| S0 | Committed classifier | 0.758 | — | 18/27 at 30.5% | 0.893 / 0.381 / 0.704 | 0.69 |
| S4 | Forecast-row lags refreshed to the anchor | 0.773 | +0.015 [−0.033, +0.065] | 19/27 at 29.1% | 0.905 / 0.392 / 0.717 | 0.72 |

**Verdict.** S1/S2 (the clip) and S4 (refreshed forecast-row lags for the classifier) hold
up in all three windows. S3 helps onset ranking but costs MAE and R² on the holdout, so it
is not worth porting as is. S5 does not hold up: its validation onset gain shrinks to
noise and it costs R² and MAE.


## 5. Reproducing

All experiment code is in `scripts/experiments/`; nothing in `forecasting/` or
`config/` was changed.

| Script | Purpose |
|---|---|
| `harness.py` | One-change variants of the engine's forecast (`--window` val, dev or holdout; `--variants a,b,a+b`; `--verify`) |
| `pooled_study.py` | Declared pooled comparison; `--ablation F1:<family>` for the feature-group diagnostic |
| `rescore.py` | Clip / blend / bias / threshold rescoring and paired comparisons → `results/experiments/summary_*.csv` |
| `pooled_report.py` | Full pooled table, pairwise ΔAUC, declared selection rule, ablation table |
| `holdout_check.py` | The one-time holdout check (S1–S5 only) |
| `diagnostics.py` | §2 breakdowns of the committed predictions |
| `doc_tables.py`, `report_tables.py` | Markdown tables for this page |
| `run_experiments.sbatch`, `run_pooled.sbatch` | Hyak (`ckpt`) jobs; outputs on `/gscratch/scrubbed/ac283/experiments/` |

```bash
# Hyak, from a worktree of this branch (a window takes ~3 min per variant on 16 CPUs)
WINDOWS=val VARIANTS=baseline,refresh EXP_OUT=/gscratch/scrubbed/$USER/experiments/raw \
  sbatch --cpus-per-task=16 --mem=32G scripts/experiments/run_experiments.sbatch
WINDOWS=val EXP_OUT=/gscratch/scrubbed/$USER/experiments/raw \
  sbatch --cpus-per-task=16 --mem=32G scripts/experiments/run_pooled.sbatch
# laptop: rescoring and reports from the copied raw directory
python scripts/experiments/rescore.py --window val --raw-dir <raw>
python scripts/experiments/pooled_report.py --window val --raw-dir <raw>
python scripts/experiments/holdout_check.py --raw-dir <raw>
```

Hyak jobs: round 1 42629724–27, onset round 42630120–21, pooled 42630238–39 and 42632004,
shortlist holdout 42631685, ablation 42632793–95.

