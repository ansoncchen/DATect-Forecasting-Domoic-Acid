# Reported numbers

Every metric DATect reports, the file it comes from, and how it got there. The
README, CLAUDE.md, the dashboard (`GET /api/metrics/summary`) and the paper quote
this page; when a re-run changes a number, change it here first.

Current numbers come from the October 2026 cleanup re-run at commit `a2de8dac`
(Hyak golden job 42541704, `scripts/eval/golden_baseline.sbatch`), plus two local
passes over its cached predictions (`spike_detection_eval.py`,
`spike_env_lead_analysis.py`). Every model is seeded with `MODEL_SEED = 42`.

## Headline: deterministic chronological evaluation

Every real DA measurement in the window is a test point (no sampling), each
forecast trained only on data up to `test_date - 7 days`. 95% CIs are row-level
bootstrap. Spike = actual DA ≥ 20 µg/g; the regression alert fires at a predicted
≥ 12 µg/g.

| Window | N | Spikes | R² [95% CI] | MAE µg/g [95% CI] | RMSE | Spike recall | Precision | F2 |
|---|---:|---:|---|---|---:|---:|---:|---:|
| **Holdout 2022–2023** | **409** | 84 | **0.419 [0.247, 0.546]** | **7.49 [6.08, 9.07]** | 17.06 | **0.774** | 0.489 | **0.693** |
| Validation 2019–2021 | 635 | 114 | 0.329 [0.203, 0.465] | 6.82 [5.70, 8.04] | 16.71 | 0.763 | 0.649 | 0.737 |

Source: `results/chronological/chronological_regression_ensemble_{20220101_20240101,20190101_20220101}.{json,parquet}`.
`tests/test_chronological_eval_smoke.py` pins the holdout row.

### Per-site holdout (2022–2023)

| Site | State | N | Spikes | R² [95% CI] | MAE |
|---|---|---:|---:|---|---:|
| Quinault | WA | 55 | 9 | 0.708 [0.518, 0.832] | 2.96 |
| Coos Bay | OR | 35 | 14 | 0.634 [0.451, 0.869] | 13.84 |
| Twin Harbors | WA | 42 | 8 | 0.596 [0.342, 0.755] | 3.82 |
| Copalis | WA | 60 | 6 | 0.589 [0.383, 0.758] | 3.25 |
| Kalaloch | WA | 35 | 1 | 0.537 [0.331, 0.727] | 2.10 |
| Long Beach | WA | 44 | 8 | 0.526 [0.204, 0.843] | 3.63 |
| Clatsop Beach | OR | 48 | 16 | 0.500 [0.300, 0.693] | 5.76 |
| Newport | OR | 48 | 16 | 0.041 [−0.321, 0.289] | 22.62 |
| Gold Beach | OR | 40 | 6 | −0.273 [−2.285, 0.058] | 11.66 |
| Cannon Beach | OR | 2 | — | too few measurements in the window | — |

8 of the 9 evaluable sites have positive R². All five Washington sites score
0.53–0.71; the two southern Oregon sites (Newport, Gold Beach) are near or below
zero, consistent with their near-zero persistence ceilings (below).

## Confirmation: 5-seed bootstrap

`multi_seed_baseline.py`, seeds 42–46, a random 20% of all real measurements
(N = 1,054 per seed), same configuration. Source: `results/multi_seed/baseline_seed{42..46}_predictions.parquet`.

| Subset | R² mean ± std (min–max) | MAE | Spike recall | F2 |
|---|---|---:|---:|---:|
| All years | 0.345 ± 0.047 (0.292–0.395) | 6.18 ± 0.38 | 0.793 ± 0.027 | 0.737 ± 0.019 |
| Holdout rows only (N 100–133 per seed) | 0.377 ± 0.105 | | | |
| Validation rows only (N 171–183 per seed) | 0.417 ± 0.071 | | | |

The holdout subset's seed-to-seed spread (± 0.105) is why a single seed is never
reported alone; the deterministic evaluation above has no sampling noise.

## Paper sample (seed 123, 40%, N = 2,105)

`eval_paper_metrics.py --seed 123 --sample-fraction 0.40` (all years, 2003–2023).
Source: `results/paper_metrics/` (`table1_model_comparison.csv`,
`table2_per_site.csv`, `table3_classification.csv`, `paper_metrics.json`).

**Table 1: models**

| Model | R² [95% CI] | MAE [95% CI] | RMSE | ΔR² of ensemble vs model [95% CI] |
|---|---|---|---:|---|
| **Ensemble (XGB + RF)** | **0.337 [0.257, 0.416]** | **6.53 [5.78, 7.29]** | 19.42 | — |
| XGBoost | 0.331 [0.257, 0.410] | 6.78 [6.04, 7.58] | 19.51 | +0.006 [−0.006, +0.017] |
| Random Forest | 0.330 [0.247, 0.415] | 6.42 [5.67, 7.23] | 19.52 | +0.007 [−0.009, +0.021] |
| Ridge | 0.325 [0.242, 0.410] | 6.61 [5.84, 7.37] | 19.59 | +0.012 [−0.007, +0.032] |
| Naive persistence | 0.150 [−0.142, 0.384] | 6.68 [5.83, 7.60] | 22.00 | +0.188 [−0.019, +0.484] |

The ensemble's lead over XGBoost, Random Forest and Ridge is within the paired
bootstrap CI: the four learners are statistically tied on this sample.

**Table 2: per site**

| Site | N | R² [95% CI] | MAE | 4-category accuracy |
|---|---:|---|---:|---:|
| Copalis | 275 | 0.737 [0.649, 0.824] | 2.15 | 0.855 |
| Kalaloch | 222 | 0.603 [0.468, 0.674] | 2.98 | 0.865 |
| Twin Harbors | 216 | 0.486 [0.359, 0.768] | 4.67 | 0.792 |
| Long Beach | 209 | 0.484 [0.354, 0.656] | 3.71 | 0.809 |
| Quinault | 196 | 0.462 [0.289, 0.774] | 3.05 | 0.781 |
| Clatsop Beach | 309 | 0.455 [0.361, 0.548] | 5.77 | 0.783 |
| Coos Bay | 110 | 0.311 [0.156, 0.440] | 24.91 | 0.582 |
| Newport | 208 | 0.085 [−0.176, 0.275] | 11.21 | 0.548 |
| Gold Beach | 242 | 0.062 [−0.040, 0.207] | 12.16 | 0.789 |
| Cannon Beach | 118 | −0.021 [−0.070, 0.145] | 2.72 | 0.898 |

**Table 3: four categories** (left-closed bins [0,5) [5,20) [20,40) [40,∞))

| Category | N | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Low | 1,370 | 0.871 | 0.914 | 0.892 |
| Moderate | 457 | 0.603 | 0.626 | 0.614 |
| High | 160 | 0.438 | 0.375 | 0.404 |
| Extreme | 118 | 0.696 | 0.331 | 0.448 |
| Macro average | 2,105 | 0.652 | 0.561 | 0.590 |

Overall accuracy 0.778.

## Spike detection and transitions (seed 123, 40%)

`spike_detection_eval.py --seed 123 --sample-fraction 0.40` on the same 2,105
predictions: 278 spikes, of which 93 are **transitions** (the site's previous
test observation was below 20). Here the regression rows alert at a predicted
≥ 20; the classifier alerts at probability ≥ 0.10; the hybrid alert is
classifier ≥ 0.10 OR ensemble ≥ 12, exactly as the engine's `spike_alert`.

| Alert | Spike recall | Precision | F2 | Transition recall |
|---|---:|---:|---:|---:|
| Naive persistence (≥ 20) | 0.701 | 0.666 | 0.694 | 0.118 |
| Ensemble regression (≥ 20) | 0.511 | 0.736 | 0.544 | 0.086 |
| Ensemble regression (≥ 12) | 0.755 | 0.572 | 0.710 | — |
| Spike classifier (p ≥ 0.10) | 0.910 | 0.483 | 0.773 | 0.731 |
| **Hybrid alert** | **0.921** | **0.474** | **0.775** | **0.763** |

On the deterministic 2022–2023 holdout (409 rows, 84 spikes, 29 transitions) the
hybrid alert has recall 0.893, precision 0.381, F2 0.704 and catches 20 of the 29
transitions (0.690); scored with the snippet in
[`QUICKSTART_FOR_REVIEWERS.md`](QUICKSTART_FOR_REVIEWERS.md).

The ensemble's mean prediction on transition rows is 10.8 µg/g against a mean
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

## Component ablation (seed 42, 20%)

<!-- ABLATION: filled from Hyak job 42597354 (results/ablation/paper_ablation_results.json) -->

## How the headline got here

| # | Protocol | Holdout R² | N | Why it moved |
|---|---|---|---:|---|
| 1 | Single seed 42, hand-tuned config | 0.492 | ~160 | Lucky seed |
| 2 | 5 seeds (42–46), hand-tuned | 0.386 ± 0.145 | ~160/seed | Honest seed noise; the hand-tuned values had seen part of the validation window |
| 3 | 5 seeds, grid-search winners (leak-free) | 0.433 ± 0.092 | ~160/seed | Tuned with chronological CV on pre-2022 data only |
| 4 | Deterministic chronological (published) | 0.485 [0.330, 0.604] | 404 | All holdout measurements; no sampling noise |
| 5 | Cleanup fixes (`0330150b`…`d53fb6d5`) | 0.441 [0.257, 0.571] | 409 | Spike rule, non-detects, PN parsing, mae070 OAD, classifier encoding (see below) |
| **6** | **One measurement per site-day (`a2de8dac`)** | **0.419 [0.247, 0.546]** | **409** | Same-day duplicates scored at their daily max |

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
bash scripts/eval/golden_baseline.sh a2de8dac post-dedup
bash scripts/eval/golden_baseline.sh a2de8dac ablation-post-dedup scripts/eval/paper_ablation_study.sbatch

# Laptop, from the seed-123 cache copied to results/retro_seed123/
.venv/bin/python scripts/eval/spike_detection_eval.py --seed 123 --sample-fraction 0.40
.venv/bin/python scripts/eval/spike_env_lead_analysis.py
.venv/bin/python scripts/eval/autocorrelation_diagnostic.py
```
