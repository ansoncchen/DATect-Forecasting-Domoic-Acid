# Reported numbers

Every metric DATect reports, the file it comes from, and how it got there. The
README, CLAUDE.md, the dashboard (`GET /api/metrics/summary`) and the paper quote
this page; when a re-run changes a number, change it here first.

The headline, 5-seed, paper-sample and spike-detection numbers come from the re-run
after the leakage fixes, at commit `2688fb8d` (Hyak golden job 42601539,
`scripts/eval/golden_baseline.sbatch`). The ablation and sensitivity studies were
run before those fixes, at `a2de8dac`, and are marked as such; they will be
re-run after the clip re-tune. Every model is seeded with `MODEL_SEED = 42`.

## Headline: deterministic chronological evaluation

Every real DA measurement in the window is a test point (no sampling), each
forecast trained only on data up to `test_date - 7 days`. 95% CIs are row-level
bootstrap. Spike = actual DA ≥ 20 µg/g; the regression alert fires at a predicted
≥ 12 µg/g.

| Window | N | Spikes | R² [95% CI] | MAE µg/g [95% CI] | RMSE | Spike recall | Precision | F2 |
|---|---:|---:|---|---|---:|---:|---:|---:|
| **Holdout 2022–2023** | **409** | 84 | **0.497 [0.365, 0.601]** | **7.60 [6.34, 9.06]** | 15.87 | **0.833** | 0.486 | **0.729** |
| Validation 2019–2021 | 635 | 114 | 0.324 [0.207, 0.450] | 7.19 [6.07, 8.42] | 16.78 | 0.798 | 0.555 | 0.734 |

Naive persistence on the same rows: R² 0.225 (holdout) and 0.213 (validation).

Source: `results/chronological/chronological_regression_ensemble_{20220101_20240101,20190101_20220101}.{json,parquet}`.
`tests/test_chronological_eval_smoke.py` pins the holdout row.

### Per-site holdout (2022–2023)

| Site | State | N | Spikes | R² [95% CI] | MAE |
|---|---|---:|---:|---|---:|
| Coos Bay | OR | 35 | 14 | 0.670 [0.506, 0.892] | 13.76 |
| Twin Harbors | WA | 42 | 8 | 0.655 [0.359, 0.816] | 3.73 |
| Kalaloch | WA | 35 | 1 | 0.650 [0.358, 0.804] | 1.93 |
| Copalis | WA | 60 | 6 | 0.576 [0.338, 0.752] | 3.43 |
| Quinault | WA | 55 | 9 | 0.568 [0.263, 0.768] | 3.50 |
| Clatsop Beach | OR | 48 | 16 | 0.554 [0.371, 0.658] | 6.23 |
| Long Beach | WA | 44 | 8 | 0.502 [0.097, 0.834] | 4.13 |
| Newport | OR | 48 | 16 | 0.216 [−0.056, 0.392] | 20.92 |
| Gold Beach | OR | 40 | 6 | −0.146 [−2.029, 0.155] | 13.01 |
| Cannon Beach | OR | 2 | — | too few measurements in the window | — |

8 of the 9 evaluable sites have positive R². All five Washington sites score
0.50–0.66; the two southern Oregon sites score lowest (Newport 0.22, Gold Beach
below zero), consistent with their near-zero persistence ceilings (below).

## Confirmation: 5-seed bootstrap

`multi_seed_baseline.py`, seeds 42–46, a random 20% of all real measurements
(N = 1,054 per seed), same configuration. Source: `results/multi_seed/baseline_seed{42..46}_predictions.parquet`.

| Subset | R² mean ± std (min–max) | MAE | Spike recall | F2 |
|---|---|---:|---:|---:|
| All years | 0.381 ± 0.026 (0.353–0.409) | 6.60 ± 0.30 | 0.832 ± 0.021 | 0.751 ± 0.018 |
| Holdout rows only (N 100–133 per seed) | 0.458 ± 0.093 | | | |
| Validation rows only (N 171–183 per seed) | 0.398 ± 0.062 | | | |

All five seeds improved over the pre-fix run (0.345 ± 0.047). The holdout
subset's seed-to-seed spread (± 0.093) is why a single seed is never
reported alone; the deterministic evaluation above has no sampling noise.

## Paper sample (seed 123, 40%, N = 2,105)

`eval_paper_metrics.py --seed 123 --sample-fraction 0.40` (all years, 2003–2023).
Source: `results/paper_metrics/` (`table1_model_comparison.csv`,
`table2_per_site.csv`, `table3_classification.csv`, `paper_metrics.json`).

**Table 1: models**

| Model | R² [95% CI] | MAE [95% CI] | RMSE | ΔR² of ensemble vs model [95% CI] |
|---|---|---|---:|---|
| **Ensemble (XGB + RF)** | **0.377 [0.312, 0.440]** | **6.99 [6.25, 7.73]** | 18.83 | — |
| XGBoost | 0.357 [0.300, 0.420] | 7.13 [6.42, 7.92] | 19.13 | +0.020 [+0.003, +0.038] |
| Random Forest | 0.385 [0.318, 0.450] | 7.43 [6.73, 8.16] | 18.71 | −0.008 [−0.027, +0.013] |
| Ridge | 0.362 [0.304, 0.423] | 7.18 [6.41, 7.90] | 19.05 | +0.015 [−0.008, +0.040] |
| Naive persistence | 0.166 [−0.128, 0.402] | 6.62 [5.78, 7.54] | 21.78 | +0.211 [−0.002, +0.505] |

The ensemble beats XGBoost alone (paired CI excludes zero) and is statistically
tied with Random Forest and Ridge. Naive persistence has the lowest MAE: most
measurements are low and stay low, which persistence copies exactly.

**Table 2: per site**

| Site | N | R² [95% CI] | MAE | 4-category accuracy |
|---|---:|---|---:|---:|
| Copalis | 275 | 0.614 [0.476, 0.756] | 3.08 | 0.789 |
| Kalaloch | 222 | 0.585 [0.460, 0.647] | 3.75 | 0.752 |
| Twin Harbors | 216 | 0.470 [0.337, 0.759] | 5.25 | 0.782 |
| Long Beach | 209 | 0.455 [0.306, 0.656] | 3.98 | 0.775 |
| Quinault | 196 | 0.413 [0.253, 0.725] | 3.39 | 0.755 |
| Clatsop Beach | 309 | 0.412 [0.299, 0.514] | 6.45 | 0.722 |
| Coos Bay | 110 | 0.355 [0.228, 0.470] | 24.79 | 0.582 |
| Newport | 208 | 0.245 [0.043, 0.368] | 11.04 | 0.428 |
| Gold Beach | 242 | 0.162 [0.054, 0.283] | 12.66 | 0.744 |
| Cannon Beach | 118 | −0.022 [−0.072, 0.128] | 2.75 | 0.890 |

**Table 3: four categories** (left-closed bins [0,5) [5,20) [20,40) [40,∞))

| Category | N | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Low | 1,370 | 0.871 | 0.839 | 0.855 |
| Moderate | 457 | 0.510 | 0.580 | 0.543 |
| High | 160 | 0.349 | 0.425 | 0.383 |
| Extreme | 118 | 0.594 | 0.348 | 0.439 |
| Macro average | 2,105 | 0.581 | 0.548 | 0.555 |

Overall accuracy 0.724 (0.778 before the leakage fixes): the corrected model
predicts more mid-range values, which helps R² and costs low-category hits.

## Spike detection and transitions (seed 123, 40%)

`spike_detection_eval.py --seed 123 --sample-fraction 0.40` on the same 2,105
predictions: 278 spikes, of which 93 are **transitions** (the site's previous
test observation was below 20). Here the regression rows alert at a predicted
≥ 20; the classifier alerts at probability ≥ 0.10; the hybrid alert is
classifier ≥ 0.10 OR ensemble ≥ 12, exactly as the engine's `spike_alert`.

| Alert | Spike recall | Precision | F2 | Transition recall |
|---|---:|---:|---:|---:|
| Naive persistence (≥ 20) | 0.687 | 0.668 | 0.683 | 0.129 |
| Ensemble regression (≥ 20) | 0.619 | 0.652 | 0.625 | 0.237 |
| Ensemble regression (≥ 12) | 0.813 | 0.536 | 0.737 | — |
| Spike classifier (p ≥ 0.10) | 0.885 | 0.480 | 0.757 | 0.699 |
| **Hybrid alert** | **0.910** | **0.466** | **0.764** | **0.753** |

On the deterministic 2022–2023 holdout (409 rows, 84 spikes, 29 transitions) the
hybrid alert has recall 0.893, precision 0.381, F2 0.704 and catches 20 of the 29
transitions (0.690); scored with the snippet in
[`QUICKSTART_FOR_REVIEWERS.md`](QUICKSTART_FOR_REVIEWERS.md). These match the
pre-fix values exactly by coincidence: 20 alerts switched on and 20 switched off,
one of each on a real spike.

The ensemble's mean prediction on transition rows is 15.3 µg/g against a mean
actual 50.2 (10.8 before the leakage fixes): regression shrinks toward the mean
and rarely crosses 20 at onset, which is why the separate classifier exists. Sources:
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

**Pre-fix.** Everything from here through the classifier importance table was run at
`a2de8dac`, before the leakage fixes (see "Leakage fixes" below), and will be
re-run after the clip re-tune. Treat the deltas as indicative.

`paper_ablation_study.py` (Hyak job 42597354), each run in a fresh process. Source:
`results/ablation/paper_ablation_results.json`, log `results/final_verification/paper_ablation_study.txt`.

| Run | R² | ΔR² | MAE |
|---|---:|---:|---:|
| Full DATect | 0.332 | — | 6.29 |
| No gap-filled training rows | 0.339 | +0.007 | 6.39 |
| No per-site configuration | 0.327 | -0.005 | 6.14 |
| No observation-order lags | 0.335 | +0.003 | 6.29 |
| No derived features (`pn_log`) | 0.332 | +0.000 | 6.29 |
| No OAD features (16) | 0.335 | +0.003 | 6.27 |

Every component is within ±0.01 R² of the full system on this sample, including
the per-site configuration (pre-cleanup: −0.187). ΔR² is paired on identical test
points, but it is one seed; see the per-site check below.

**Per-site configuration on the deterministic windows** (`chronological_eval.py` with
`DATECT_USE_PER_SITE_MODELS=false`, Hyak job 42599199;
`results/chronological_no_per_site/`):

| Window | Per-site on | Per-site off |
|---|---|---|
| Validation 2019–2021 (N = 635) | R² 0.329, MAE 6.82, F2 0.737 | R² 0.368 [0.114, 0.593], MAE 6.37, F2 0.739 |
| Holdout 2022–2023 (N = 409) | R² 0.419, MAE 7.49, F2 0.693 | R² 0.343 [0.069, 0.520], MAE 7.55, F2 0.688 |

The per-site configuration hurts on validation (−0.039) and helps on the holdout
(+0.076), both inside the CIs: roughly neutral overall, as the single-seed ablation
says. Reported only; the holdout is not used to choose a configuration.

**Feature groups** (`paper_sensitivity_study.py --study features`, same sample;
`results/ablation/paper_features_results.json`): no group moves R² by more than
0.004, and dropping every environmental predictor at once costs −0.004.

| Group dropped | R² | ΔR² |
|---|---:|---:|
| Satellite (SST, SST anomaly, FLH) | 0.335 | +0.003 |
| Climate indices (PDO, ONI) | 0.333 | +0.001 |
| Upwelling (BEUTI) | 0.332 | -0.000 |
| River discharge | 0.332 | -0.000 |
| Pseudo-nitzschia (count, log count) | 0.332 | +0.000 |
| Rolling DA statistics (7) | 0.331 | -0.001 |
| Calendar (month, day-of-year sin/cos) | 0.334 | +0.002 |
| All environmental (satellite, climate, BEUTI, discharge, PN) | 0.328 | -0.004 |

**Design perturbations** (`paper_sensitivity_study.py --study perturbations`, same
sample, Hyak job 42597854; `results/ablation/paper_perturbations_results.json`):

| Perturbation | R² | ΔR² | MAE |
|---|---:|---:|---:|
| Baseline (full DATect) | 0.332 | — | 6.29 |
| XGBoost at every site | 0.331 | -0.001 | 6.40 |
| Random Forest at every site | 0.323 | -0.009 | 6.23 |
| No per-site config | 0.327 | -0.005 | 6.14 |
| RF shallow (depth 6, 200 trees) | 0.332 | +0.000 | 6.27 |
| RF deep (depth 16, 600 trees) | 0.332 | -0.000 | 6.29 |
| RF more trees (800) | 0.332 | +0.000 | 6.29 |
| RF less regularized (leaf 1) | 0.332 | -0.000 | 6.29 |
| All features at every site | 0.347 | +0.015 | 6.25 |
| Minimal features (4) | 0.333 | +0.001 | 6.49 |
| Clip at q = 0.99 everywhere | 0.371 | +0.039 | 6.13 |
| No prediction clipping | 0.388 | +0.055 | 6.08 |
| No monotonic constraints | 0.332 | +0.000 | 6.29 |

**Known issue: the per-site clip quantiles are stale.** They were chosen by the
weights × clip grid search against the pre-`a2de8dac` targets. Against daily-max
targets, capping forecasts at the 95th–97th percentile of training targets cuts off
real spikes: without clipping the development-sample R² rises from 0.332 to 0.388
(MAE 6.29 → 6.08). "No per-site configuration" (−0.005) nets this out against the
per-site weights and hyperparameters, which still add about +0.04. This sample
includes 2022–2023 rows, so it must not be used to pick new values. Plan (decided
2026-10-29): re-run the constrained grid search on pre-2022 chronological CV with the
corrected targets as the first Spark + MLflow workload, check the holdout once, then
re-run the golden baseline and update every number here. A test run of that
pipeline on the pre-fix features matched the committed predictions exactly; the
re-tune that counts runs on the leakage-fixed features.

**Spike-classifier gain importance** (`spike_classifier_importance.py`: the pooled
classifier retrained as the engine does at 60 anchors per seed, seeds 42–46;
`results/spike_classifier/importance_by_seed.csv`). The three rolling DA maxima
carry 57% of the gain; PDO is the strongest environmental feature.

| Feature | Mean gain | Range over seeds |
|---|---:|---|
| `raw_obs_roll_max_4` | 0.323 | 0.311–0.343 |
| `raw_obs_roll_max_8` | 0.160 | 0.142–0.173 |
| `raw_obs_roll_max_12` | 0.090 | 0.082–0.097 |
| `pdo` | 0.031 | 0.031–0.032 |
| `raw_obs_roll_mean_4` | 0.025 | 0.023–0.029 |
| `da_raw_prev_obs_1` | 0.020 | 0.019–0.021 |

**Per-site R² by seed** (`seed_stability_table.py` over `results/multi_seed`, post-fix;
`results/multi_seed/per_site_r2_by_seed.csv`): Washington means 0.37–0.73 (std
0.05–0.14), Clatsop Beach 0.44 ± 0.04, Coos Bay 0.28 ± 0.05, the other Oregon
sites −0.03 to 0.18.

## How the headline got here

| # | Protocol | Holdout R² | N | Why it moved |
|---|---|---|---:|---|
| 1 | Single seed 42, hand-tuned config | 0.492 | ~160 | Lucky seed |
| 2 | 5 seeds (42–46), hand-tuned | 0.386 ± 0.145 | ~160/seed | Honest seed noise; the hand-tuned values had seen part of the validation window |
| 3 | 5 seeds, grid-search winners (leak-free) | 0.433 ± 0.092 | ~160/seed | Tuned with chronological CV on pre-2022 data only |
| 4 | Deterministic chronological (published) | 0.485 [0.330, 0.604] | 404 | All holdout measurements; no sampling noise |
| 5 | Cleanup fixes (`0330150b`…`d53fb6d5`) | 0.441 [0.257, 0.571] | 409 | Spike rule, non-detects, PN parsing, mae070 OAD, classifier encoding (see below) |
| 6 | One measurement per site-day (`a2de8dac`) | 0.419 [0.247, 0.546] | 409 | Same-day duplicates scored at their daily max |
| **7** | **Leakage fixes (`e14db2eb`, `2688fb8d`)** | **0.497 [0.365, 0.601]** | **409** | Training persistence features no longer contain the target; only weeks that end by the anchor (see below) |

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

## Leakage fixes (after the cleanup)

A data-pipeline audit found two problems; each fix is its own commit, evaluated
with the deterministic windows (Hyak jobs 42601540 and 42601539).

| Step | Holdout R² | Validation R² | 5-seed pooled |
|---|---|---|---|
| Before (`a2de8dac`) | 0.419 [0.247, 0.546] | 0.329 [0.203, 0.465] | 0.345 ± 0.047 |
| + `e14db2eb` | 0.458 [0.306, 0.570] | 0.354 [0.242, 0.472] | — |
| + `2688fb8d` (current) | 0.497 [0.365, 0.601] | 0.324 [0.207, 0.450] | 0.381 ± 0.026 |

**`e14db2eb`: training persistence features described the target.** On training
rows, `last_observed_da_raw` and `weeks_since_last_spike` were forward-filled
*including the row's own measurement*, so on all 4,447 real-measurement rows the
"last observation" equalled the target (the rolling statistics already excluded
the row and are unchanged). The model learned to copy it; at test time the same feature holds the
previous measurement, so forecasts behaved like persistence and lagged every onset
(ensemble transition recall 0.086 on the seed-123 sample). This was a train/test
mismatch, not test-time leakage, which is why fixing it raises skill. Transition
recall is now 0.237.

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
| Grid search over ensemble weights × clip quantile × clip max (chronological CV, pre-2022) | Held up on the holdout | **Promoted** to `config/tuned_hyperparameters.json`; the current configuration |
| Five extra feature sets (lagged PN, BEUTI derivatives, NeMO mooring, ESP offshore pDA, NDBC wind) | Each \|ΔR²\| < 0.003 | Removed |
| Per-anchor XGB grid tuning | 23% of predictions change, no skill gain | Off by default (`9c435f59`) |

Two conclusions from this work still stand: OAD tracks *Pseudo-nitzschia* at the
offshore NEMO mooring but adds nothing at the beaches (see
[`../oad/OAD_STORYLINE.md`](../oad/OAD_STORYLINE.md), Chapter 9), and the existing
feature set is at a flat optimum where new in-situ data, not more derived
features, is the likely route to better skill.

## Reproducing

```bash
# Hyak: chronological (both windows), multi-seed 42-46, paper metrics (seed 123),
# spike eval, in a disposable worktree
bash scripts/eval/golden_baseline.sh 2688fb8d post-leakfix
# Pre-fix ablation (to be re-run after the clip re-tune)
bash scripts/eval/golden_baseline.sh a2de8dac ablation-post-dedup scripts/eval/paper_ablation_study.sbatch
# Appendix studies, one job per STUDY (features, perturbations, classifier), from a worktree
sbatch --export=ALL,STUDY=features scripts/eval/paper_sensitivity_study.sbatch

# Laptop, from the seed-123 cache copied to results/retro_seed123/
.venv/bin/python scripts/eval/spike_detection_eval.py --seed 123 --sample-fraction 0.40
.venv/bin/python scripts/eval/spike_env_lead_analysis.py
.venv/bin/python scripts/eval/autocorrelation_diagnostic.py
.venv/bin/python scripts/eval/seed_stability_table.py
```
