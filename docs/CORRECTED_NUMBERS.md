# Reported numbers

Every metric DATect reports, the file it comes from, and how it got there. The
README, CLAUDE.md and the dashboard (`GET /api/metrics/summary`) quote this page;
when a re-run changes a number, change it here first. The paper's tables still
predate the leakage fixes and are updated from this page in one pass.

Every number below comes from commit `1c9206c6` (one random forest with shared
settings at every beach; see "One random forest at every beach" below) unless its
section names another commit. Hyak golden job 43275770
(`scripts/eval/golden_baseline.sbatch`) produced the headline, 5-seed, paper-sample
and spike numbers, and job 43275774 the all-years run behind the error band. The
component ablation, the per-site configuration check, the feature-group and
perturbation studies still describe the retired per-site configuration at
`93518c7d` (jobs 42671544, 42671548, 42674365, 42674366); the spike classifier did
not change, so its importance (job 42671547) still holds. Every model is seeded
with `MODEL_SEED = 42`.

## Headline: deterministic chronological evaluation

Every real DA measurement in the window is a test point (no sampling), each
forecast trained only on data up to `test_date - 7 days`. 95% CIs are row-level
bootstrap. Spike = actual DA ≥ 20 µg/g; the regression alert fires at a predicted
≥ 12 µg/g.

| Window | N | Spikes | R² [95% CI] | MAE µg/g [95% CI] | RMSE | Spike recall | Precision | F2 |
|---|---:|---:|---|---|---:|---:|---:|---:|
| **Holdout 2022–2023** | **409** | 84 | **0.473 [0.200, 0.644]** | **7.34 [5.99, 8.85]** | 16.23 | **0.845** | 0.470 | **0.729** |
| Validation 2019–2021 | 635 | 114 | 0.409 [0.186, 0.611] | 6.57 [5.53, 7.72] | 15.69 | 0.842 | 0.545 | 0.759 |

Naive persistence on the same rows: R² 0.225 (holdout) and 0.213 (validation).

**Read the pooled R² with two caveats.** It is carried by a few large values: on
the holdout, Coos Bay alone scores 0.796 and the other sites together 0.153
(validation: 0.278 and 0.412). On a log scale (R² of `log1p` values), which weighs
every row, the holdout scores 0.413 [0.256, 0.558] and validation 0.387 [0.264, 0.500].

Source: `results/chronological/chronological_regression_ensemble_{20220101_20240101,20190101_20220101}.{json,parquet}`.
`tests/test_chronological_eval_smoke.py` pins the holdout row.

### Spike onsets

An **onset** is a test point whose last value known at the anchor was below 20 and
whose measurement reached 20; the other below-20 points are **quiet weeks**. Points
already at or above 20 are left out, because carrying the last value forward
catches those for free. The **hybrid alert** (`spike_alert`) is the pooled spike
classifier's probability ≥ 0.10 OR the forecast ≥ 12. AUC ranks onsets
against quiet weeks (`forecasting.metrics.onset_scores`; the chronological JSON's
`overall.onset` block).

| Window | Onsets / quiet weeks | Hybrid alert: onsets caught, quiet weeks alarmed | Classifier AUC | Forecast ≥ 12: caught, alarmed | Forecast AUC | Last-value AUC |
|---|---|---|---:|---|---:|---:|
| Holdout 2022–2023 | 27 / 292 | **20 (0.74)**, 30.5% | **0.773** | 14, 16.4% | 0.724 | 0.550 |
| Validation 2019–2021 | 34 / 486 | **27 (0.79)**, 15.2% | **0.874** | 20, 9.3% | 0.749 | 0.651 |

The hybrid alert's recall, precision and F2 over all spikes are 0.917 / 0.387 /
0.720 on the holdout and 0.939 / 0.495 / 0.796 on validation. For scale: alerting
on the highest last known values, with as many below-20 alerts as the hybrid
raises, catches 14 of the 27 holdout onsets and 20 of the 34 validation onsets.
The hybrid alert's lead over that rule (6 and 7 onsets) is what the models add for
early warning; the forecast alone (≥ 12) catches about half the onsets (14 of 27,
20 of 34).

**The classifier's score is not a calibrated chance.** It ranks weeks, but its
value does not say how often a jump follows: on the holdout, scores of 0.35–0.60
were followed by a jump 15% of the time and scores of 0.60 or more 39%. The
dashboard therefore prints it as a score to two decimals, never as a percentage,
beside what weeks at that level were followed by. Weeks whose last known value
was below 20, followed by a jump to ≥ 20:

| Score | Holdout 2022–2023 | Validation 2019–2021 |
|---|---:|---:|
| Below 0.10 | 10 of 224 | 8 of 432 |
| 0.10–0.35 | 7 of 57 | 12 of 51 |
| 0.35–0.60 | 3 of 20 | 7 of 20 |
| 0.60+ | 7 of 18 | 7 of 17 |

Bins are left-closed. The rows sum to the onsets and quiet weeks above (27 of 319,
34 of 520). Source: `results/conformal_band/band_predictions.parquet`
(`spike_probability`, `naive_prediction` < 20, `actual_da` ≥ 20).

### Per-site holdout (2022–2023)

| Site | State | N | Spikes | R² [95% CI] | MAE | log R² |
|---|---|---:|---:|---|---:|---:|
| Coos Bay | OR | 35 | 14 | 0.796 [0.628, 0.915] | 11.25 | 0.855 |
| Copalis | WA | 60 | 6 | 0.682 [0.480, 0.819] | 2.88 | 0.769 |
| Kalaloch | WA | 35 | 1 | 0.666 [0.408, 0.810] | 1.86 | 0.647 |
| Quinault | WA | 55 | 9 | 0.666 [0.383, 0.841] | 3.03 | 0.639 |
| Twin Harbors | WA | 42 | 8 | 0.609 [0.324, 0.775] | 3.93 | 0.540 |
| Long Beach | WA | 44 | 8 | 0.552 [0.207, 0.861] | 3.76 | 0.661 |
| Clatsop Beach | OR | 48 | 16 | 0.537 [0.357, 0.693] | 5.87 | 0.685 |
| Newport | OR | 48 | 16 | 0.041 [−0.722, 0.425] | 22.16 | −0.075 |
| Gold Beach | OR | 40 | 6 | −0.449 [−4.735, 0.196] | 13.17 | −0.535 |
| Cannon Beach | OR | 2 | — | too few measurements in the window | — | — |

8 of the 9 evaluable sites have positive R². All five Washington sites score
0.55–0.68; the two southern Oregon sites score lowest (Newport 0.04, Gold Beach
below zero), consistent with their near-zero persistence ceilings (below).

### Error band on the forecast line

An 80% split-conformal band from the model's own past errors (log1p scale,
pooled across sites; the most recent 250 errors known at the anchor from
forecasts in the same predicted category, [0,5) [5,20) [20,∞), or from every
forecast while that category has fewer than 30; chosen on validation). Protocol
and full results: `docs/ERROR_BAND.md`.

| Window | 50% band | 80% band | 90% band | Median 80% width |
|---|---:|---:|---:|---:|
| Holdout 2022–2023 | 0.521 | **0.812** | 0.917 | 15.7 µg/g |
| Validation 2019–2021 | 0.532 | 0.811 | 0.907 | 11.5 µg/g |

Calibrated overall and roughly by forecast size (holdout 0.77 below 5, 0.86 from
5 to 20, 0.79 from 20 up), not by site: holdout 80% coverage is 0.45 at Gold
Beach and 0.48 at Newport, 0.79–0.97 elsewhere. The declared one-model test kept
the spike classifier: on the holdout the band's P(DA ≥ 20) catches 17 of 27
onsets at 25.7% quiet alarms (hybrid alert: 20 at 30.5%), onset AUC 0.706 vs
0.773, difference −0.068 [−0.179, +0.036].

Source: `results/conformal_band/band_eval.json`, from the all-years run
`results/chronological/chronological_regression_ensemble_20030101_20240101.parquet`
(Hyak job 43275774, commit `1c9206c6`; its 2019–2023 rows equal the headline
predictions exactly).

## Confirmation: 5-seed bootstrap

`multi_seed_baseline.py`, seeds 42–46, a random 20% of all real measurements
(N = 1,054 per seed), same configuration. Source: `results/multi_seed/baseline_seed{42..46}_predictions.parquet`.

| Subset | R² mean ± std (min–max) | MAE | Spike recall | F2 |
|---|---|---:|---:|---:|
| All years | 0.407 ± 0.035 (0.350–0.441) | 6.75 ± 0.26 | 0.867 ± 0.022 | 0.760 ± 0.014 |
| Holdout rows only (N 100–133 per seed) | 0.322 ± 0.136 | | | |
| Validation rows only (N 171–183 per seed) | 0.582 ± 0.097 | | | |

The retired per-site configuration scored 0.448 ± 0.034 (MAE 6.32) on the same
samples; the difference is in the years before 2019 (see "One random forest at
every beach"). The holdout subset's seed-to-seed spread (± 0.136) is why a
single seed is never reported alone; the deterministic evaluation above has no
sampling noise.

## Paper sample (seed 123, 40%, N = 2,105)

`eval_paper_metrics.py --seed 123 --sample-fraction 0.40` (all years, 2003–2023).
Source: `results/paper_metrics/` (`table1_model_comparison.csv`,
`table2_per_site.csv`, `table3_classification.csv`, `paper_metrics.json`).

**Table 1: models**

| Model | R² [95% CI] | MAE [95% CI] | RMSE | ΔR² of the random forest vs model [95% CI] |
|---|---|---|---:|---|
| **Random forest (served)** | **0.447 [0.335, 0.542]** | **7.04 [6.36, 7.74]** | 17.74 | — |
| XGBoost (same features and settings) | 0.295 [0.075, 0.462] | 6.78 [6.00, 7.60] | 20.03 | +0.152 [+0.050, +0.296] |
| Ridge | 0.417 [0.291, 0.525] | 6.88 [6.16, 7.56] | 18.21 | +0.030 [−0.027, +0.091] |
| Naive persistence | 0.166 [−0.128, 0.402] | 6.62 [5.78, 7.54] | 21.78 | +0.281 [+0.111, +0.513] |

The random forest beats XGBoost on the same features (the paired CI excludes
zero) and is statistically tied with Ridge. It has the highest MAE of the four:
most measurements are low and stay low, which persistence copies exactly, and the
forest over-forecasts low weeks in the early years of this all-years sample (see
"One random forest at every beach"). The retired per-site configuration scored
R² 0.464 [0.367, 0.552], MAE 6.61 on the same 2,105 points.

**Table 2: per site**

| Site | N | R² [95% CI] | MAE | 4-category accuracy |
|---|---:|---|---:|---:|
| Kalaloch | 222 | 0.828 [0.578, 0.909] | 2.85 | 0.788 |
| Clatsop Beach | 309 | 0.681 [0.487, 0.813] | 6.41 | 0.631 |
| Quinault | 196 | 0.678 [0.505, 0.777] | 2.97 | 0.791 |
| Long Beach | 209 | 0.578 [0.413, 0.710] | 4.48 | 0.775 |
| Twin Harbors | 216 | 0.564 [0.392, 0.847] | 5.18 | 0.741 |
| Coos Bay | 110 | 0.527 [0.341, 0.653] | 22.37 | 0.627 |
| Copalis | 275 | 0.525 [0.321, 0.673] | 4.62 | 0.433 |
| Cannon Beach | 118 | 0.188 [−1.919, 0.319] | 3.14 | 0.839 |
| Gold Beach | 242 | 0.135 [−0.222, 0.355] | 12.80 | 0.756 |
| Newport | 208 | 0.002 [−0.808, 0.410] | 11.36 | 0.519 |

Copalis's 4-category accuracy (0.43; per-site configuration 0.79) is the early-year
over-forecast: before 2016 the forest forecasts 5–8 µg/g there in weeks that
measured below 5.

**Table 3: four categories** (left-closed bins [0,5) [5,20) [20,40) [40,∞))

| Category | N | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Low | 1,370 | 0.883 | 0.726 | 0.797 |
| Moderate | 457 | 0.424 | 0.650 | 0.513 |
| High | 160 | 0.399 | 0.406 | 0.403 |
| Extreme | 118 | 0.597 | 0.576 | 0.586 |
| Macro average | 2,105 | 0.576 | 0.590 | 0.575 |

Overall accuracy 0.677 (per-site configuration 0.723).

## Spike detection and transitions (seed 123, 40%)

`spike_detection_eval.py --seed 123 --sample-fraction 0.40` on the same 2,105
predictions: 278 spikes, of which 93 are **transitions** (the site's previous
measurement was below 20; unlike an onset above, that measurement may fall
inside the 7-day forecast gap). Here the regression rows alert at a predicted
≥ 20 unless noted; the classifier alerts at probability ≥ 0.10; the hybrid alert
is classifier ≥ 0.10 OR forecast ≥ 12, exactly as the engine's `spike_alert`.

| Alert | Spike recall | Precision | F2 | Transition recall |
|---|---:|---:|---:|---:|
| Naive persistence (≥ 20) | 0.687 | 0.668 | 0.683 | 0.129 |
| Forecast (≥ 20) | 0.662 | 0.664 | 0.662 | 0.247 |
| Forecast (≥ 12) | 0.845 | 0.519 | 0.751 | — |
| Spike classifier (p ≥ 0.10) | 0.896 | 0.484 | 0.766 | 0.710 |
| **Hybrid alert** | **0.921** | **0.461** | **0.768** | **0.785** |

On the deterministic 2022–2023 holdout (409 rows, 84 spikes, 29 transitions) the
hybrid alert catches 22 of the 29 transitions (0.759; scored with the snippet in
[`QUICKSTART_FOR_REVIEWERS.md`](QUICKSTART_FOR_REVIEWERS.md)).

The forecast's mean on transition rows is 19.8 µg/g against a mean actual 50.2:
regression shrinks toward the mean and rarely crosses 20 at onset,
which is why the separate classifier exists. Sources:
`results/spike_metrics_seed123/{spike_metrics_by_model,per_site_metrics,threshold_sweep}.csv`,
log in `results/final_verification/spike_detection_eval.txt`.

## Exploratory analyses (full data, not out-of-sample)

**Environmental lead signals before a spike** (`scripts/eval/spike_env_lead_analysis.py`):
224 raw-observation transitions (previous measurement < 20, this one ≥ 20) at 10
sites. Largest Wilcoxon effect sizes |r| over the 1–4 week leads: Pseudo-nitzschia
0.56 (lower than the site baseline before a spike), SST anomaly 0.45, PDO 0.43,
discharge 0.35 (lower), SST 0.30, BEUTI 0.24. ONI shows no lead signal
(p ≥ 0.43 at every lead). Nominal p-values, no multiple-comparison correction.
Sources: `results/env_lead/{spike_events,statistical_tests}.csv`,
`results/final_verification/spike_env_lead_analysis.txt`.

**Persistence ceiling** (`scripts/eval/autocorrelation_diagnostic.py`): squared
lag-1 autocorrelation of consecutive measurements at most 28 days apart, the R² a
last-value forecast could reach. Washington sites 0.73–0.81 (mean 0.76); Clatsop
Beach 0.71; Coos Bay 0.25; Cannon Beach 0.23; Newport and Gold Beach 0.05.
Source: `results/final_verification/autocorrelation_diagnostic.txt`.

## Component ablation (seed 42, 20%, N = 1,054)

**This section and the per-site, feature-group and perturbation studies below
measure the retired per-site configuration at `93518c7d`** (the scripts pin
`DATECT_USE_PER_SITE_MODELS=true`). They are the record of what that
configuration did and were not re-run for the random forest; "Full DATect" here
is the per-site configuration, not the served model.

`paper_ablation_study.py` (Hyak job 42671544), each run in a fresh process. Source:
`results/ablation/paper_ablation_results.json`, log `results/final_verification/paper_ablation_study.txt`.
This sample includes 2022–2023 rows, so it is a diagnostic, never used to choose a
configuration.

| Run | R² | ΔR² | MAE |
|---|---:|---:|---:|
| Full DATect | 0.459 | — | 6.35 |
| No gap-filled training rows | 0.428 | −0.031 | 6.98 |
| No per-site configuration | 0.364 | −0.095 | 6.65 |
| No observation-order lags | 0.459 | −0.000 | 6.34 |
| No derived features (`pn_log`) | 0.458 | −0.001 | 6.36 |
| No OAD features (16) | 0.433 | −0.026 | 6.79 |

With the clip gone, the per-site configuration (−0.095), the gap-filled training
rows (−0.031) and the OAD features (−0.026) carry weight on this sample; before the
clip removal all three were within ±0.01. ΔR² is paired on identical test points,
but it is one seed; see the deterministic per-site check below.

**Per-site configuration on the deterministic windows** (`chronological_eval.py` with
`DATECT_USE_PER_SITE_MODELS=false`, Hyak job 42671548;
`results/chronological_no_per_site/`). "Off" meant, at `93518c7d`, equal XGB/RF
weights, no per-site feature subsets and no per-site hyperparameters; since
`1c9206c6` off is the random forest alone, the served model.

| Window | Per-site on | Per-site off | ΔR² off − on [95% CI] | Δ log R² |
|---|---|---|---|---|
| Validation 2019–2021 (N = 635) | R² 0.423, MAE 6.55, F2 0.745 | R² 0.344, MAE 6.72, F2 0.735 | −0.079 [−0.258, +0.012] | +0.039 [−0.021, +0.101] |
| Holdout 2022–2023 (N = 409) | R² 0.516, MAE 7.44, F2 0.737 | R² 0.419, MAE 7.61, F2 0.717 | −0.097 [−0.268, +0.005] | +0.033 [−0.025, +0.090] |

The per-site configuration raises R² in both windows, through the large values,
while the shared configuration scores slightly better on a log scale; every interval
includes zero. Onset detection is the same either way (hybrid alert 27/34 and 19/27).
The validation window was the per-site weights' tuning objective, so only the
holdout row is a clean comparison. Reported only; the holdout is not used to choose
a configuration.

**Feature groups** (`paper_sensitivity_study.py --study features`, same sample, Hyak
job 42674365; `results/ablation/paper_features_results.json`): no group moves R² by
more than 0.007, and dropping every environmental predictor at once costs −0.010.

| Group dropped | R² | ΔR² |
|---|---:|---:|
| Satellite (SST, SST anomaly, FLH) | 0.465 | +0.006 |
| Climate indices (PDO, ONI) | 0.452 | −0.007 |
| Upwelling (BEUTI) | 0.457 | −0.001 |
| River discharge | 0.458 | −0.001 |
| Pseudo-nitzschia (count, log count) | 0.458 | −0.001 |
| Rolling DA statistics (7) | 0.463 | +0.005 |
| Calendar (month, day-of-year sin/cos) | 0.464 | +0.005 |
| All environmental (satellite, climate, BEUTI, discharge, PN) | 0.449 | −0.010 |

**Design perturbations** (`paper_sensitivity_study.py --study perturbations`, same
sample, Hyak job 42674366; `results/ablation/paper_perturbations_results.json`):

| Perturbation | R² | ΔR² | MAE |
|---|---:|---:|---:|
| Baseline (full DATect) | 0.459 | — | 6.35 |
| XGBoost at every site | 0.433 | −0.026 | 6.37 |
| Random Forest at every site | 0.428 | −0.031 | 6.92 |
| No per-site config | 0.364 | −0.095 | 6.65 |
| RF shallow (depth 6, 200 trees) | 0.459 | −0.000 | 6.37 |
| RF deep (depth 16, 600 trees) | 0.459 | +0.000 | 6.34 |
| RF more trees (800) | 0.459 | +0.000 | 6.35 |
| RF less regularized (leaf 1) | 0.459 | +0.000 | 6.22 |
| All features at every site | 0.480 | +0.021 | 6.22 |
| Minimal features (4) | 0.377 | −0.081 | 6.62 |
| Clip at the 0.95 training quantile everywhere | 0.358 | −0.101 | 6.70 |
| Clip at the 0.99 training quantile everywhere | 0.439 | −0.020 | 6.42 |
| No monotonic constraints | 0.461 | +0.002 | 6.32 |

Putting a clip back costs R² at either quantile, confirming its removal. The
per-site feature subsets are the one per-site choice that hurts here: every feature
at every site scores +0.021. The per-site XGBoost/RF weights add about 0.03 over
either model alone. RF hyperparameters and monotone constraints make no difference.

**Spike-classifier gain importance** (`spike_classifier_importance.py`: the pooled
classifier retrained as the engine does at 60 anchors per seed, seeds 42–46, Hyak
job 42671547; `results/spike_classifier/importance_by_seed.csv`). The three rolling
DA maxima carry 57% of the gain; PDO is the strongest environmental feature.

| Feature | Mean gain | Range over seeds |
|---|---:|---|
| `raw_obs_roll_max_4` | 0.327 | 0.314–0.350 |
| `raw_obs_roll_max_8` | 0.155 | 0.133–0.169 |
| `raw_obs_roll_max_12` | 0.091 | 0.083–0.099 |
| `pdo` | 0.031 | 0.030–0.032 |
| `raw_obs_roll_mean_4` | 0.025 | 0.022–0.027 |
| `da_raw_prev_obs_1` | 0.020 | 0.019–0.021 |

**Per-site R² by seed** (`seed_stability_table.py` over `results/multi_seed`;
`results/multi_seed/per_site_r2_by_seed.csv`, the random forest): Washington means
0.47–0.71 (std 0.06–0.20), Clatsop Beach 0.66 ± 0.04, Coos Bay 0.39 ± 0.21, the
other Oregon sites −0.36 to 0.23.

## How the headline got here

| # | Protocol | Holdout R² | N | Why it moved |
|---|---|---|---:|---|
| 1 | Single seed 42, hand-tuned config | 0.492 | ~160 | Lucky seed |
| 2 | 5 seeds (42–46), hand-tuned | 0.386 ± 0.145 | ~160/seed | Honest seed noise; the hand-tuned values had seen part of the validation window |
| 3 | 5 seeds, grid-search winners (leak-free) | 0.433 ± 0.092 | ~160/seed | Tuned with chronological CV on pre-2022 data only |
| 4 | Deterministic chronological (published) | 0.485 [0.330, 0.604] | 404 | All holdout measurements; no sampling noise |
| 5 | Cleanup fixes (`0330150b`…`d53fb6d5`) | 0.441 [0.257, 0.571] | 409 | Spike rule, non-detects, PN parsing, mae070 OAD, classifier encoding (see below) |
| 6 | One measurement per site-day (`a2de8dac`) | 0.419 [0.247, 0.546] | 409 | Same-day duplicates scored at their daily max |
| 7 | Leakage fixes (`e14db2eb`, `2688fb8d`) | 0.497 [0.365, 0.601] | 409 | Training persistence features no longer contain the target; only weeks that end by the anchor (see below) |
| 8 | No clip; forecast-row lags (`0f4466d2`, `c099f20b`) | 0.516 [0.308, 0.654] | 409 | Forecasts no longer capped below real highs; forecast rows built like training rows (see below) |
| **9** | **One random forest at every beach (`1c9206c6`)** | **0.473 [0.200, 0.644]** | **409** | Per-beach model picks, feature lists and hyperparameters replaced by one shared configuration; a tie with step 8 (see below) |

**Why 0.485 became 0.419.** About 1,200 site-days have several same-day DA
samples that often disagree (Gold Beach 2019-01-24 recorded 0 and 190 µg/g; the
CSVs carry no species or station to tell them apart). Training already used the
weekly maximum, but the evaluation scored each forecast against whichever sample
sorted first, so the published target depended on file order. Scoring the
unchanged pre-cleanup predictions against daily-max targets gives a holdout R² of
0.419 (validation 0.332), the same as the post-cleanup model (0.419 / 0.329). So
the cleanup's model and data fixes are neutral on the holdout, and the entire
move from the published 0.485 comes from the target definition. Step 5's
0.441 is the same arbitrary pick landing on different duplicates after the fixes
reordered rows. The daily max raises 123 targets (65 validation, 58 holdout),
lowers none, and turns 16 more holdout measurements into spikes (68 → 84).

The validation window followed the same path: 0.384 (published) → 0.312 → 0.329.

## One random forest at every beach (`1c9206c6`)

The per-site configuration picked XGBoost, a random forest or a blend per beach,
with its own feature list and hyperparameters, from a grid search. A declared
comparison ([`FORECAST_EXPERIMENTS.md`](FORECAST_EXPERIMENTS.md) §3c) asked which single
algorithm to use at every beach with shared features and settings. On validation
the random forest beat XGBoost (ΔR² XGBoost − forest −0.160 [−0.405, −0.029]); the
holdout, read once after the pick, agreed (−0.132 [−0.292, −0.024]). `1c9206c6`
made it the served model (`DATECT_USE_PER_SITE_MODELS=false`, weights (0, 1, 0));
`true` restores the per-site configuration. Its predictions equal the comparison
run's forest column exactly (635, 409 and 3,510 of 3,510 rows).

| | Per-site configuration (`93518c7d`) | Random forest (`1c9206c6`) | ΔR² forest − per-site [95% CI] |
|---|---|---|---|
| Validation 2019–2021 (N = 635) | R² 0.423, MAE 6.55, onsets 27/34 | R² 0.409, MAE 6.57, onsets 27/34 | −0.014 [−0.095, +0.042] |
| Holdout 2022–2023 (N = 409) | R² 0.516, MAE 7.44, onsets 19/27 | R² 0.473, MAE 7.34, onsets 20/27 | −0.043 [−0.154, +0.030] |
| 5 seeds, all years (20%) | R² 0.448 ± 0.034, MAE 6.32 | R² 0.407 ± 0.035, MAE 6.75 | — |
| Paper sample (seed 123, 40%, all years) | R² 0.464, MAE 6.61 | R² 0.447, MAE 7.04 | — |

Paired intervals from `scripts/eval/one_algorithm_compare.py` (2,000 row resamples).
Both deterministic windows are ties, with log-scale R² up (holdout 0.370 → 0.413,
validation 0.343 → 0.387) and the hybrid alert unchanged or one onset better.

**Where it costs: before 2019.** The all-years run (every real measurement,
2005–2023) by period, per-site configuration → forest:

| Years | N | R² | MAE µg/g | Mean error when the measurement is below 5 |
|---|---:|---|---|---|
| 2005–2010 | 805 | 0.039 → −0.190 | 2.25 → 2.95 | +1.6 → +2.5 |
| 2011–2015 | 961 | 0.566 → 0.559 | 5.70 → 6.30 | +2.5 → +3.3 |
| 2016–2018 | 700 | 0.269 → 0.213 | 11.65 → 12.11 | +7.4 → +10.6 |
| 2019–2021 | 635 | 0.423 → 0.409 | 6.55 → 6.57 | +2.9 → +2.9 |
| 2022–2023 | 409 | 0.516 → 0.473 | 7.44 → 7.34 | +5.0 → +5.3 |

The forest over-forecasts quiet weeks more in the early years, mostly at Copalis
(MAE +3.3 µg/g over 2005–2015), Clatsop Beach (+1.6) and Long Beach (+1.4), three
beaches where the per-site configuration used XGBoost. At Copalis in 2005–2010
(median 341 training rows) the forest's median forecast is 6.4 µg/g against
measurements averaging 1.2; XGBoost on the same shared features forecasts 0.3, so
this is the forest, not the shared feature list. The comparison chose on
2019–2021 and showed 2015–2018 as a second look (forest 0.275 vs per-site
0.314); the years before 2015 were not part of it. The 5-seed and paper samples
draw from all years, which is why they fall while the deterministic windows tie.

**Error band.** Re-run on the forest's errors, the declared band protocol
(`docs/ERROR_BAND.md`) now chooses the most recent 250 errors with Mondrian
calibration (validation interval score 3.521; the previous choice, 500 errors
without Mondrian, 3.523). `backend/band.py` now serves Mondrian calibration,
equal to the evaluated band on all 3,480 rows that have one.

## Clip removal and forecast-row lags (after the leakage fixes)

A forecast-pipeline audit ([`FORECAST_EXPERIMENTS.md`](FORECAST_EXPERIMENTS.md); code on
branch `claude/forecast-experiments-b1c907`) checked the engine against its own description and
ran small, pre-declared experiments on validation and 2015–2018 before one holdout
check. Its fixes are separate commits, evaluated with the golden runs
`port-c1-neutral`, `port-c2-noclip` and `port-c4-lags` (Hyak jobs 42671541–43).

| Step | Holdout R² | Validation R² | 5-seed pooled | Holdout onsets (hybrid) | Validation onsets |
|---|---|---|---|---|---|
| Before (`c0a0b2ae`) | 0.497 [0.365, 0.601] | 0.324 [0.207, 0.450] | 0.381 ± 0.026 | 18/27, 30.5% alarmed | 24/34, 14.6% |
| + `0f4466d2` no clip | 0.550 [0.373, 0.668] | 0.426 [0.221, 0.628] | 0.456 ± 0.033 | 18/27, 30.5% | 24/34, 14.6% |
| + `c099f20b` forecast-row lags (current) | 0.516 [0.308, 0.654] | 0.423 [0.216, 0.626] | 0.448 ± 0.034 | 19/27, 29.8% | 27/34, 15.0% |

**`3f9bfba6`: the classifier's safe-baseline filter did nothing.** It was meant to
train the spike classifier only on weeks whose previous measurement was below 20,
but it compared the MinMax-scaled column (always in [0, 1]) with 20, so it kept
every row. Applied as documented it lowered validation onset AUC by 0.023
[−0.046, −0.002], so the filter was deleted and the paper and docs now describe the
classifier that runs. All 18 golden outputs are byte-identical to the previous run.

**`0f4466d2`: forecasts were capped below real highs.** XGBoost and RF forecasts were
clipped at the per-site 0.95/0.97 quantile of the training targets, about 59% of
which are gap-filled (decay-extrapolated, often ~0). At a 2019 anchor Coos Bay was
capped at 81 µg/g while its real 95th percentile was 145. Removing the clip raised
R² by 0.102 [−0.031, +0.207] (validation) and 0.053 [−0.012, +0.115] (holdout), MAE
7.19 → 6.55 and 7.60 → 7.39, with log-scale R² and every spike metric unchanged.
Most of the holdout gain is Coos Bay (0.670 → 0.789); the other sites together
moved 0.292 → 0.304. The Spark re-tune chose no quantile clip at 7 of 10 sites,
though 5 of those kept a hard ceiling, so only 2 chose no clip at all. The ensemble
weights, chosen jointly with the clip, are unchanged.

**`c099f20b`: forecast rows saw older information than training rows.** A training
row for week T sees measurements up to the end of week T−1; the forecast row was the
last weekly row whose week ended by the anchor, so its lags and rolling statistics
stopped 13–19 days before the target (only `last_observed_da_raw` and
`weeks_since_last_spike` were refreshed). `da_raw_prev_obs_1` equalled the last
measurement in 100% of training rows but in 49% of forecast rows. Every persistence
feature is now rebuilt from daily measurements up to the anchor with the training
definitions (exact on 2,977 of 2,977 sampled training rows;
`TestTestRowMatchesTrainingRows`). Effect, validation / holdout:

| | ΔR² [95% CI] | Δ log R² | Onsets caught | Classifier onset AUC | Hybrid F2 |
|---|---|---|---|---|---|
| Validation | −0.003 [−0.028, +0.020] | +0.016 [−0.001, +0.034] | 24 → 27 of 34 | 0.864 → 0.874 | 0.781 → 0.797 |
| Holdout | −0.034 [−0.090, +0.005] | −0.008 [−0.028, +0.012] | 18 → 19 of 27 | 0.758 → 0.773 | 0.704 → 0.714 |

The holdout R² dip is Newport (+6,836 of the +6,868 change in squared error), whose
measurements swing between 0 and 100 from one sample to the next; the refreshed lags
follow those swings. The fix was kept for correctness and for the onset gain in both
windows, which the audit had measured before the holdout check (onset AUC +0.010 /
+0.018 / +0.015 on validation / 2015–2018 / holdout).

**Reporting (`93518c7d`).** The chronological JSON now carries the onset scores and
log-scale R² shown above.

## Leakage fixes (after the cleanup)

A data-pipeline audit found two problems; each fix is its own commit, evaluated
with the deterministic windows (Hyak jobs 42601540 and 42601539).

| Step | Holdout R² | Validation R² | 5-seed pooled |
|---|---|---|---|
| Before (`a2de8dac`) | 0.419 [0.247, 0.546] | 0.329 [0.203, 0.465] | 0.345 ± 0.047 |
| + `e14db2eb` | 0.458 [0.306, 0.570] | 0.354 [0.242, 0.472] | — |
| + `2688fb8d` | 0.497 [0.365, 0.601] | 0.324 [0.207, 0.450] | 0.381 ± 0.026 |

**`e14db2eb`: training persistence features described the target.** On training
rows, `last_observed_da_raw` and `weeks_since_last_spike` were forward-filled
*including the row's own measurement*, so on all 4,447 real-measurement rows the
"last observation" equalled the target (the rolling statistics already excluded
the row and are unchanged). The model learned to copy it; at test time the same feature holds the
previous measurement, so forecasts behaved like persistence and lagged every onset
(ensemble transition recall 0.086 on the seed-123 sample). This was a train/test
mismatch, not test-time leakage, which is why fixing it raises skill. Transition
recall rose to 0.237.

**`2688fb8d`: weeks that straddle the anchor.** Weekly rows are labelled with
their Monday and aggregate Monday–Sunday, so a row dated on or before the anchor
could contain up to six days after it: the persistence features and naive
baseline came from that partial week for 14 of 409 holdout points, and the anchor
row's weekly PN maximum included post-anchor samples for about 23% of evaluation
points. Training now uses only weeks with
`date + 6 days <= anchor` (`week_complete_by`), and the test row's persistence
features and the naive baseline come from raw daily measurements up to the anchor.
This removed real future information; validation fell 0.03 and the holdout rose
0.04, both well inside the CIs.

Both changes are covered by `tests/test_leakage_audit.py`.

## Changes in the October 2026 cleanup

Behavior-changing commits, each with before/after numbers in its message.
Everything else in the cleanup was checked byte-identical against the
pre-cleanup golden run (`compare_golden.py`).

| Commit | Change | Effect on reported numbers |
|---|---|---|
| `81718fff` | Models seeded by `MODEL_SEED = 42`, separate from the sampling seed | None: published runs already trained with 42 in loky workers |
| `9c435f59` | Per-anchor XGB tuning off by default (`DATECT_PER_ANCHOR_TUNING`) | None for reported numbers (all ran with it off). Tuning changes 23% of predictions without improving skill (pooled R² 0.414 on vs 0.417 off) |
| `af778bd1` | One spike rule: actual ≥ 20, alert ≥ 12; left-closed categories | Spike counts and `weeks_since_last_spike` |
| `6ac5ea78` | `"<1"` → LOD/2 = 0.5 (339 readings); PN `"12,345"` parsed (1,604 values); dates stripped | 5 more holdout and 3 more validation test points; training targets |
| `a3368361` | OAD features from the `mae070` checkpoint | Within noise (see ablation) |
| `92c1f47c` | Dashboard forecasts call the evaluated core; intervals use `reg:quantileerror` | Dashboard only |
| `0330150b` | Spike classifier scores test rows with its own pooled encoding | Spike probability, hybrid alert, transition recall |
| `d53fb6d5` | Non-zero intervals at RF-only sites; removed a no-op ablation | Dashboard intervals only |
| `a2de8dac` | One DA measurement per site-day (daily max) | Evaluation targets (above) |
| `80e6f530` | Persistence ceiling uses only pairs ≤ 28 days apart, as the paper describes | With `a2de8dac`'s one-per-day data, Clatsop Beach 0.25 → 0.71 |

`spike_env_lead_analysis.py` was restored (`b428bae9`): the paper cited it after
it had been deleted. With one measurement per site-day it finds 224 transitions;
the same loader without the collapse gives the previously published 385, so 161
of those were two same-day samples read as a crossing.

## Experiments that did not help

Measured before the cleanup (mae050 OAD checkpoint, old parsing); conclusions
only. Code for the removed experiments is at git tag `archive/pre-cleanup`;
details in [`OAD_INTEGRATION_RESULTS.md`](OAD_INTEGRATION_RESULTS.md) §7–8.

| Experiment | Result | Disposition |
|---|---|---|
| OAD regional anomaly features (16 columns) | ΔR² within noise, including SW Washington | Kept (not harmful); re-measured in the ablation above |
| Optuna search over 18 per-site hyperparameters | Validation +0.07 R², holdout −0.16 | Discarded (overfit) |
| Grid search over ensemble weights × clip quantile × clip max (chronological CV, pre-2022) | Held up on the holdout | Weights promoted to `config/tuned_hyperparameters.json` and still used; the clip was removed in `0f4466d2` |
| Five extra feature sets (lagged PN, BEUTI derivatives, NeMO mooring, ESP offshore pDA, NDBC wind) | Each \|ΔR²\| < 0.003 | Removed |
| Per-anchor XGB grid tuning | 23% of predictions change, no skill gain | Off by default (`9c435f59`) |

Two conclusions from this work still stand: OAD tracks *Pseudo-nitzschia* at the
offshore NEMO mooring but adds nothing at the beaches (see
[`../oad/OAD_STORYLINE.md`](../oad/OAD_STORYLINE.md), Chapter 9), and the existing
feature set is at a flat optimum where new in-situ data, not more derived
features, is the likely route to better skill.

**Forecast-experiments audit (2026-10, after the leakage fixes).** Each was declared
on validation and 2015–2018 before a single holdout check
([`FORECAST_EXPERIMENTS.md`](FORECAST_EXPERIMENTS.md)). None was adopted.

| Experiment | Result |
|---|---|
| log1p / sqrt targets, Tweedie, Poisson and Huber losses, predicting the change | Tie or worse in R² and MAE |
| Online bias correction; blending with persistence | Only compensated for the clip; nothing left once it is gone |
| Training rows built exactly like forecast rows, with cross-site DA and raw PN | Validation onset AUC +0.077, holdout +0.014 [−0.042, +0.081]; holdout MAE +0.39 |
| Level forecast + 10 × spike probability as one output | Onset AUC up, holdout MAE +0.81 |
| Onset-only classifier (trained on below-20 weeks); the safe-baseline filter as documented | Lower onset AUC (−0.076, −0.023) |
| Pooled linear vs Random Forest vs XGBoost, declared comparison | No significant family difference for onsets; the engine's pooled XGBoost classifier best or tied in every window |
| One pooled level model vs the per-site ensemble | Holdout −0.079 [−0.263, +0.055]: unsettled |
| Per-site weight and clip picks from the Spark re-tune | Tie one shared rule on the holdout |
| Recency weights, shorter training windows, monotone constraints off, per-site feature subsets, hyperparameter nudges | Ties |
| Dropping environmental feature groups, scored on onsets | No group lowers onset AUC by more than 0.004; rolling DA history is the only group that matters |

## Reproducing

```bash
# Hyak: chronological (both windows), multi-seed 42-46, paper metrics (seed 123),
# spike eval, in a disposable worktree
bash scripts/eval/golden_baseline.sh 1c9206c6 rf-1c9206c6
# The all-years run behind the error band, then (laptop) the band protocol
python scripts/eval/chronological_eval.py --window 2003-01-01:2024-01-01
.venv/bin/python scripts/eval/conformal_band_eval.py
# The retired per-site configuration's ablation (pins DATECT_USE_PER_SITE_MODELS=true)
bash scripts/eval/golden_baseline.sh 93518c7d ablation-port-93518c7d scripts/eval/paper_ablation_study.sbatch
# Appendix studies, one job per STUDY (features, perturbations, classifier), from a worktree
sbatch --export=ALL,STUDY=features scripts/eval/paper_sensitivity_study.sbatch
# The retired per-site configuration, both windows
DATECT_USE_PER_SITE_MODELS=true python scripts/eval/chronological_eval.py --window 2022-01-01:2024-01-01 --out-dir results/chronological_per_site

# Laptop, after copying the golden run's results/retro_seed123/ (the seed-123 retrospective
# parquets spike_detection_eval.py writes and reloads; not the dashboard's precomputed store)
.venv/bin/python scripts/eval/spike_detection_eval.py --seed 123 --sample-fraction 0.40
.venv/bin/python scripts/eval/spike_env_lead_analysis.py
.venv/bin/python scripts/eval/autocorrelation_diagnostic.py
.venv/bin/python scripts/eval/seed_stability_table.py
```
