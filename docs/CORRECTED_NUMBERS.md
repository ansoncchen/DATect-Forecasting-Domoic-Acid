# Reported numbers

Every metric DATect reports, the file it comes from, and how it got there. The
README, CLAUDE.md and the dashboard (`GET /api/metrics/summary`) quote this page;
when a re-run changes a number, change it here first. The paper's tables still
predate the leakage fixes and are updated from this page in one pass.

Every number below comes from commit `93518c7d` (forecasts no longer clipped,
forecast-row lags built like training rows; see "Clip removal and forecast-row
lags" below). Hyak golden job 42671543 (`scripts/eval/golden_baseline.sbatch`)
produced the headline, 5-seed, paper-sample and spike numbers; jobs 42671544
(ablation), 42671548 (per-site configuration off), 42674365 and 42674366 (feature
groups, perturbations) and 42671547 (classifier importance) the rest. Every
model is seeded with `MODEL_SEED = 42`.

## Headline: deterministic chronological evaluation

Every real DA measurement in the window is a test point (no sampling), each
forecast trained only on data up to `test_date - 7 days`. 95% CIs are row-level
bootstrap. Spike = actual DA ≥ 20 µg/g; the regression alert fires at a predicted
≥ 12 µg/g.

| Window | N | Spikes | R² [95% CI] | MAE µg/g [95% CI] | RMSE | Spike recall | Precision | F2 |
|---|---:|---:|---|---|---:|---:|---:|---:|
| **Holdout 2022–2023** | **409** | 84 | **0.516 [0.308, 0.654]** | **7.44 [6.20, 8.87]** | 15.56 | **0.845** | 0.486 | **0.737** |
| Validation 2019–2021 | 635 | 114 | 0.423 [0.216, 0.626] | 6.55 [5.51, 7.69] | 15.50 | 0.816 | 0.554 | 0.745 |

Naive persistence on the same rows: R² 0.225 (holdout) and 0.213 (validation).

**Read the pooled R² with two caveats.** It is carried by a few large values: on
the holdout, Coos Bay alone scores 0.798 and the other sites together 0.233
(validation: 0.329 and 0.425). On a log scale (R² of `log1p` values), which weighs
every row, the holdout scores 0.370 [0.234, 0.494] and validation 0.343 [0.231, 0.443].

Source: `results/chronological/chronological_regression_ensemble_{20220101_20240101,20190101_20220101}.{json,parquet}`.
`tests/test_chronological_eval_smoke.py` pins the holdout row.

### Spike onsets

An **onset** is a test point whose last value known at the anchor was below 20 and
whose measurement reached 20; the other below-20 points are **quiet weeks**. Points
already at or above 20 are left out, because carrying the last value forward
catches those for free. The **hybrid alert** (`spike_alert`) is the pooled spike
classifier's probability ≥ 0.10 OR the ensemble forecast ≥ 12. AUC ranks onsets
against quiet weeks (`forecasting.metrics.onset_scores`; the chronological JSON's
`overall.onset` block).

| Window | Onsets / quiet weeks | Hybrid alert: onsets caught, quiet weeks alarmed | Classifier AUC | Regression ≥ 12: caught, alarmed | Ensemble AUC | Last-value AUC |
|---|---|---|---:|---|---:|---:|
| Holdout 2022–2023 | 27 / 292 | **19 (0.70)**, 29.8% | **0.773** | 14, 14.4% | 0.744 | 0.550 |
| Validation 2019–2021 | 34 / 486 | **27 (0.79)**, 15.0% | **0.874** | 15, 8.2% | 0.752 | 0.651 |

The hybrid alert's recall, precision and F2 over all spikes are 0.905 / 0.388 /
0.714 on the holdout and 0.939 / 0.498 / 0.797 on validation. For scale: alerting
on the highest last known values, with as many below-20 alerts as the hybrid
raises, catches 14 of the 27 holdout onsets and 20 of the 34 validation onsets.
The classifier's lead over that rule (5 and 7 onsets) is what the model adds for
early warning; the regression alert alone (≥ 12) catches about half the onsets.

### Per-site holdout (2022–2023)

| Site | State | N | Spikes | R² [95% CI] | MAE | log R² |
|---|---|---:|---:|---|---:|---:|
| Coos Bay | OR | 35 | 14 | 0.798 [0.639, 0.938] | 10.85 | 0.868 |
| Kalaloch | WA | 35 | 1 | 0.667 [0.413, 0.808] | 1.87 | 0.651 |
| Quinault | WA | 55 | 9 | 0.642 [0.355, 0.824] | 3.08 | 0.592 |
| Twin Harbors | WA | 42 | 8 | 0.630 [0.332, 0.793] | 3.80 | 0.509 |
| Copalis | WA | 60 | 6 | 0.587 [0.334, 0.759] | 3.41 | 0.697 |
| Clatsop Beach | OR | 48 | 16 | 0.538 [0.351, 0.652] | 6.17 | 0.678 |
| Long Beach | WA | 44 | 8 | 0.503 [0.131, 0.836] | 4.09 | 0.585 |
| Newport | OR | 48 | 16 | 0.129 [−0.348, 0.410] | 22.12 | −0.132 |
| Gold Beach | OR | 40 | 6 | −0.208 [−2.815, 0.202] | 13.15 | −0.701 |
| Cannon Beach | OR | 2 | — | too few measurements in the window | — | — |

8 of the 9 evaluable sites have positive R². All five Washington sites score
0.50–0.67; the two southern Oregon sites score lowest (Newport 0.13, Gold Beach
below zero), consistent with their near-zero persistence ceilings (below).

### Error band on the forecast line

An 80% split-conformal band from the model's own past errors (log1p scale,
pooled across sites, most recent 500 errors known at the anchor; chosen on
validation). Protocol and full results: `docs/ERROR_BAND.md`.

| Window | 50% band | 80% band | 90% band | Median 80% width |
|---|---:|---:|---:|---:|
| Holdout 2022–2023 | 0.484 | **0.802** | 0.900 | 15.1 µg/g |
| Validation 2019–2021 | 0.502 | 0.819 | 0.915 | 10.3 µg/g |

Calibrated overall and by forecast size, not by site: holdout 80% coverage is
0.375 at Gold Beach and 0.479 at Newport, 0.85–0.93 elsewhere. The declared
one-model test kept the spike classifier: on the holdout the band's
P(DA ≥ 20) catches 16 of 27 onsets at 27.1% quiet alarms (hybrid alert: 19 at
29.8%), onset AUC 0.747 vs 0.773, difference −0.027 [−0.126, +0.067].

Source: `results/conformal_band/band_eval.json`, from the all-years run
`results/chronological/chronological_regression_ensemble_20030101_20240101.parquet`
(Hyak job 42762246, commit `595ab4a4`; its 2019–2023 rows equal the headline
predictions exactly).

## Confirmation: 5-seed bootstrap

`multi_seed_baseline.py`, seeds 42–46, a random 20% of all real measurements
(N = 1,054 per seed), same configuration. Source: `results/multi_seed/baseline_seed{42..46}_predictions.parquet`.

| Subset | R² mean ± std (min–max) | MAE | Spike recall | F2 |
|---|---|---:|---:|---:|
| All years | 0.448 ± 0.034 (0.416–0.502) | 6.32 ± 0.28 | 0.845 ± 0.019 | 0.754 ± 0.014 |
| Holdout rows only (N 100–133 per seed) | 0.429 ± 0.121 | | | |
| Validation rows only (N 171–183 per seed) | 0.578 ± 0.082 | | | |

Every seed improved over the run before the clip removal (0.381 ± 0.026). The
holdout subset's seed-to-seed spread (± 0.121) is why a single seed is never
reported alone; the deterministic evaluation above has no sampling noise.

## Paper sample (seed 123, 40%, N = 2,105)

`eval_paper_metrics.py --seed 123 --sample-fraction 0.40` (all years, 2003–2023).
Source: `results/paper_metrics/` (`table1_model_comparison.csv`,
`table2_per_site.csv`, `table3_classification.csv`, `paper_metrics.json`).

**Table 1: models**

| Model | R² [95% CI] | MAE [95% CI] | RMSE | ΔR² of ensemble vs model [95% CI] |
|---|---|---|---:|---|
| **Ensemble (XGB + RF)** | **0.464 [0.367, 0.552]** | **6.61 [5.95, 7.31]** | 17.47 | — |
| XGBoost | 0.408 [0.334, 0.488] | 6.89 [6.20, 7.63] | 18.36 | +0.056 [+0.020, +0.091] |
| Random Forest | 0.443 [0.344, 0.539] | 7.13 [6.47, 7.84] | 17.81 | +0.021 [−0.022, +0.075] |
| Ridge | 0.393 [0.289, 0.493] | 6.92 [6.18, 7.62] | 18.59 | +0.071 [+0.027, +0.119] |
| Naive persistence | 0.166 [−0.128, 0.402] | 6.62 [5.78, 7.54] | 21.78 | +0.297 [+0.099, +0.584] |

The ensemble beats XGBoost alone and Ridge (paired CIs exclude zero) and is
statistically tied with Random Forest. Naive persistence matches the ensemble's
MAE: most measurements are low and stay low, which persistence copies exactly.

**Table 2: per site**

| Site | N | R² [95% CI] | MAE | 4-category accuracy |
|---|---:|---|---:|---:|
| Kalaloch | 222 | 0.816 [0.587, 0.890] | 3.13 | 0.752 |
| Clatsop Beach | 309 | 0.698 [0.477, 0.835] | 5.43 | 0.735 |
| Quinault | 196 | 0.681 [0.535, 0.777] | 3.00 | 0.765 |
| Copalis | 275 | 0.678 [0.535, 0.800] | 3.01 | 0.793 |
| Twin Harbors | 216 | 0.580 [0.416, 0.851] | 4.96 | 0.773 |
| Long Beach | 209 | 0.560 [0.389, 0.700] | 3.97 | 0.770 |
| Coos Bay | 110 | 0.509 [0.353, 0.626] | 22.41 | 0.591 |
| Newport | 208 | 0.198 [−0.073, 0.392] | 11.21 | 0.438 |
| Cannon Beach | 118 | 0.183 [−0.947, 0.336] | 2.94 | 0.856 |
| Gold Beach | 242 | 0.134 [0.024, 0.267] | 12.75 | 0.723 |

**Table 3: four categories** (left-closed bins [0,5) [5,20) [20,40) [40,∞))

| Category | N | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Low | 1,370 | 0.870 | 0.828 | 0.849 |
| Moderate | 457 | 0.494 | 0.578 | 0.532 |
| High | 160 | 0.378 | 0.406 | 0.392 |
| Extreme | 118 | 0.621 | 0.500 | 0.554 |
| Macro average | 2,105 | 0.591 | 0.578 | 0.582 |

Overall accuracy 0.723.

## Spike detection and transitions (seed 123, 40%)

`spike_detection_eval.py --seed 123 --sample-fraction 0.40` on the same 2,105
predictions: 278 spikes, of which 93 are **transitions** (the site's previous
measurement was below 20; unlike an onset above, that measurement may fall
inside the 7-day forecast gap). Here the regression rows alert at a predicted
≥ 20 unless noted; the classifier alerts at probability ≥ 0.10; the hybrid alert
is classifier ≥ 0.10 OR ensemble ≥ 12, exactly as the engine's `spike_alert`.

| Alert | Spike recall | Precision | F2 | Transition recall |
|---|---:|---:|---:|---:|
| Naive persistence (≥ 20) | 0.687 | 0.668 | 0.683 | 0.129 |
| Ensemble regression (≥ 20) | 0.637 | 0.663 | 0.642 | 0.172 |
| Ensemble regression (≥ 12) | 0.827 | 0.520 | 0.740 | — |
| Spike classifier (p ≥ 0.10) | 0.896 | 0.484 | 0.766 | 0.710 |
| **Hybrid alert** | **0.921** | **0.460** | **0.767** | **0.785** |

On the deterministic 2022–2023 holdout (409 rows, 84 spikes, 29 transitions) the
hybrid alert catches 21 of the 29 transitions (0.724; scored with the snippet in
[`QUICKSTART_FOR_REVIEWERS.md`](QUICKSTART_FOR_REVIEWERS.md)).

The ensemble's mean prediction on transition rows is 15.7 µg/g against a mean
actual 50.2: regression shrinks toward the mean and rarely crosses 20 at onset,
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
`results/chronological_no_per_site/`). "Off" means equal XGB/RF weights, no
per-site feature subsets and no per-site hyperparameters.

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
`results/multi_seed/per_site_r2_by_seed.csv`): Washington means 0.54–0.76 (std
0.05–0.16), Clatsop Beach 0.60 ± 0.11, Coos Bay 0.38 ± 0.16, the other Oregon
sites −0.04 to 0.22.

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
| **8** | **No clip; forecast-row lags (`0f4466d2`, `c099f20b`)** | **0.516 [0.308, 0.654]** | **409** | Forecasts no longer capped below real highs; forecast rows built like training rows (see below) |

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
moved 0.292 → 0.304. The Spark re-tune independently chose "no clip" at 7 of 10
sites. The ensemble weights, chosen jointly with the clip, are unchanged.

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
bash scripts/eval/golden_baseline.sh 93518c7d port-c4-lags
bash scripts/eval/golden_baseline.sh 93518c7d ablation-port-93518c7d scripts/eval/paper_ablation_study.sbatch
# Appendix studies, one job per STUDY (features, perturbations, classifier), from a worktree
sbatch --export=ALL,STUDY=features scripts/eval/paper_sensitivity_study.sbatch
# Per-site configuration off, both windows
DATECT_USE_PER_SITE_MODELS=false python scripts/eval/chronological_eval.py --window 2022-01-01:2024-01-01 --out-dir results/chronological_no_per_site

# Laptop, after copying the golden run's results/retro_seed123/ (the seed-123 retrospective
# parquets spike_detection_eval.py writes and reloads; not the dashboard's precomputed store)
.venv/bin/python scripts/eval/spike_detection_eval.py --seed 123 --sample-fraction 0.40
.venv/bin/python scripts/eval/spike_env_lead_analysis.py
.venv/bin/python scripts/eval/autocorrelation_diagnostic.py
.venv/bin/python scripts/eval/seed_stability_table.py
```
