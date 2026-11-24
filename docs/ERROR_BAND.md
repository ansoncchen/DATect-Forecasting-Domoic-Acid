# Error band on the forecast line, and the one-model test

The regression line gives one number per forecast. This adds an 80% band
around it from the model's own past errors (split conformal prediction), and
uses the same errors to ask whether a separate spike classifier is still
needed. The protocol below was committed before any holdout row was scored.

Code: `forecasting/conformal.py`, `scripts/eval/conformal_band_eval.py`,
`tests/test_conformal.py`. Results: `results/conformal_band/`.

## How the band is built

- **Errors on the log1p scale:** `log1p(actual) - log1p(forecast)`. A miss of
  5 ug/g matters more at 3 than at 80, so the band is narrow when DA is low and
  widens multiplicatively when it is high.
- **Past only:** a forecast for date *t* (anchor *t - 7*) is calibrated by earlier
  forecasts whose measurement is dated on or before the anchor, the same
  cut-off the model uses. `tests/test_conformal.py` rewrites every later
  measurement and checks that the band does not change.
- **One rule for every site:** errors are pooled across sites.
- **Band:** the forecast shifted by the conformal quantiles of those errors
  (ranks floor((n+1)a/2) and ceil((n+1)(1-a/2)), a = 0.2), floored at 0. At
  least 30 past errors are required; earlier rows get no band.
- **P(DA >= 20):** the share of past errors that would carry this forecast to
  20 or above.

## Inputs

One chronological retrospective over every real DA measurement:

    python scripts/eval/chronological_eval.py --window 2003-01-01:2024-01-01

Its 2019-2023 rows must equal the committed headline predictions exactly
(`parity_with_committed` in `band_eval.json`).

## Declared protocol

**Chosen on validation [2019-01-01, 2022-01-01) only:**

1. **Band variant** among six: error window = most recent 250, most recent 500
   or all past errors; with or without Mondrian calibration (errors only from
   forecasts in the same predicted category [0,5) [5,20) [20,inf), falling
   back to all categories while one has fewer than 30). Among variants whose
   80% coverage is within 0.03 of 0.80, the lowest mean interval score (log1p
   Winkler score) wins; if none is within, the closest coverage.
2. **Alert threshold** on P(DA >= 20): the smallest threshold whose validation
   quiet-week alarm rate does not exceed the deployed hybrid alert's
   (classifier p >= 0.10 or forecast >= 12), so both alerts cost the same.

**Scored once on the holdout [2022-01-01, 2024-01-01):**

- Coverage of the 80% band overall, by predicted category, by site and at
  onsets; median width. Coverage at 50% and 90% as a calibration check.
- **One-model test.** Onsets are rows whose last value known at the anchor is
  below 20 and whose measurement reaches 20; quiet weeks are the other
  below-20 rows. The band replaces the pooled spike classifier only if all three hold:
  - the paired-bootstrap 95% CI (10,000 row resamples) of onset AUC, band
    P(DA >= 20) minus classifier probability, has a lower bound above -0.05;
  - the band alert catches at least the hybrid alert's onsets minus one;
  - at a quiet-week alarm rate at most 2 points above the hybrid alert's.

  Otherwise the classifier stays as a separately reported second output (the
  spike alert), and the band supplies the line's uncertainty and categories.

Onsets are the primary metric, so the rule leans toward keeping the
classifier: the band must be shown not to be clearly worse, not merely fail to
be shown worse.

## Known before the holdout was scored

On validation, from the committed headline predictions: the classifier ranks
onsets with AUC 0.874 (hybrid alert: 27 of 34 onsets, 15.0% of quiet weeks
alarmed), and the regression line itself with AUC 0.752. Pooled errors make
P(DA >= 20) rise with the forecast, so without Mondrian calibration the band
ranks onsets almost exactly as the line does (only the error set changes over
time); the classifier is expected to win.
The holdout classifier AUC (0.773) was already reported; the band's holdout
numbers were not computed before this commit.

## Results

### Current: re-run after the training-row fix (`edf64987`, 2026-11-16)

`edf64987` stopped training on weeks with no last known value
([`CORRECTED_NUMBERS.md`](CORRECTED_NUMBERS.md), "Training rows need a last known
value"), which changes the forecasts and so their errors. The band was
re-calibrated with the protocol above, unchanged. Hyak job 43283813 ran the
all-years retrospective at `edf64987`: 3,510 forecasts, whose 635 validation and
409 holdout rows equal the committed headline predictions exactly.

| Error window | Mondrian | 80% coverage | Interval score (log1p) |
|---|---|---:|---:|
| 250 | no | 0.813 | 3.551 |
| **250** | **yes** | **0.819** | **3.516** |
| 500 | no | 0.828 | 3.535 |
| 500 | yes | 0.809 | 3.569 |
| all | no | 0.746 | 3.782 |
| all | yes | 0.748 | 3.618 |

Four variants are within 0.03 of 0.80; the rule again picks the lowest interval
score, the most recent 250 errors with Mondrian calibration. The alert threshold
is P(DA >= 20) >= 0.088 (14.6% of validation quiet weeks; hybrid alert 14.8%).

| Nominal | Validation (N = 635) | Holdout (N = 409) |
|---:|---:|---:|
| 50% | 0.529 | 0.513 |
| **80%** | **0.819** | **0.807** |
| 90% | 0.904 | 0.919 |
| Median width of the 80% band | 10.8 µg/g | 15.6 µg/g |

Holdout coverage by predicted category: 0.756 below 5 (N = 160), 0.854 from 5 to
20 (N = 178), 0.803 from 20 up (N = 71). By site it is still uneven: 0.425 at
Gold Beach and 0.46 at Newport, 0.81–0.97 at the other sites with more than two
rows. At onsets the band contains the measurement 11 times in 27 (0.41).

**One-model test: the classifier stays.** On the holdout the band's
P(DA >= 20) >= 0.088 catches 16 of 27 onsets at 24.7% quiet alarms (hybrid
alert: 19 at 30.5%); onset AUC 0.685 vs 0.773, difference −0.088 [−0.201,
+0.021]. The CI's lower bound is below −0.05 and the band catches 16 where 18
were required, so two of the three conditions fail again.

### Random forest (`1c9206c6`, 2026-11-15)

The served model became one random forest at every beach
([`FORECAST_EXPERIMENTS.md`](FORECAST_EXPERIMENTS.md) §3c), so the band was
re-calibrated on its errors with the protocol above, unchanged. Hyak job 43275774
ran the all-years retrospective at `1c9206c6`: 3,510 forecasts, whose 635
validation and 409 holdout rows equal the committed headline predictions exactly.

| Error window | Mondrian | 80% coverage | Interval score (log1p) |
|---|---|---:|---:|
| 250 | no | 0.817 | 3.545 |
| **250** | **yes** | **0.811** | **3.521** |
| 500 | no | 0.831 | 3.523 |
| 500 | yes | 0.803 | 3.533 |
| all | no | 0.775 | 3.585 |
| all | yes | 0.754 | 3.612 |

Four variants are within 0.03 of 0.80 and their interval scores differ by at most
0.024; the rule picks the lowest, the most recent 250 errors with Mondrian
calibration. The alert threshold is P(DA >= 20) >= 0.092 (14.4% of validation
quiet weeks; hybrid alert 15.2%).

| Nominal | Validation (N = 635) | Holdout (N = 409) |
|---:|---:|---:|
| 50% | 0.532 | 0.521 |
| **80%** | **0.811** | **0.812** |
| 90% | 0.907 | 0.917 |
| Median width of the 80% band | 11.5 µg/g | 15.7 µg/g |

Holdout coverage by predicted category: 0.770 below 5 (N = 161), 0.858 from 5 to
20 (N = 176), 0.792 from 20 up (N = 72). By site it is still uneven: 0.45 at
Gold Beach and 0.48 at Newport, 0.79–0.97 at the other sites with more than two
rows. At onsets the band contains the measurement 11 times in 27 (0.41).

**One-model test: the classifier stays.** On the holdout the band's
P(DA >= 20) >= 0.092 catches 17 of 27 onsets at 25.7% quiet alarms (hybrid
alert: 20 at 30.5%); onset AUC 0.706 vs 0.773, difference −0.068 [−0.179,
+0.036]. The CI's lower bound is below −0.05 and the band catches 17 where 19
were required, so two of the three conditions fail again.

### First run (`595ab4a4`, per-site configuration)

Hyak job 42762246 ran the all-years retrospective at `595ab4a4` in 15 minutes
on 16 cores: 3,510 forecasts from 2005-10-17 to 2023-11-30 (earlier site-weeks
lack the training data). Its 635 validation and 409 holdout rows equal the
committed headline predictions exactly, in every column. Its pooled metrics
are not a reported number, because the years before 2019 overlap the tuning
period; the run only supplies past errors. 3,480 rows have a band (the first 30
have too little history).

Source: `results/conformal_band/band_eval.json` (all numbers below) and
`band_predictions.parquet` (every row with its band and P(DA >= 20)); input
`results/chronological/chronological_regression_ensemble_20030101_20240101.parquet`.

#### Chosen on validation

| Error window | Mondrian | 80% coverage | Interval score (log1p) |
|---|---|---:|---:|
| 250 | no | 0.813 | 3.537 |
| 250 | yes | 0.795 | 3.631 |
| **500** | **no** | **0.819** | **3.481** |
| 500 | yes | 0.775 | 3.657 |
| all | no | 0.735 | 3.698 |
| all | yes | 0.732 | 3.708 |

Four variants are within 0.03 of 0.80; the most recent 500 errors without
Mondrian has the lowest interval score. Using every past error under-covers
(0.73): errors from the early years, with little training data, describe a
different model. The alert threshold is P(DA >= 20) >= 0.084, which alarms
14.8% of validation quiet weeks (hybrid alert: 15.0%).

#### Coverage

| Nominal | Validation (N = 635) | Holdout (N = 409) |
|---|---:|---:|
| 50% | 0.502 | 0.484 |
| **80%** | **0.819** | **0.802** |
| 90% | 0.915 | 0.900 |
| Median width of the 80% band | 10.3 µg/g | 15.1 µg/g |

On the holdout the 80% band covers 0.801 of forecasts below 5 (N = 166), 0.811
from 5 to 20 (N = 175) and 0.779 from 20 up (N = 68).

**Where it is not calibrated:**

- **By site.** Pooled errors give every site the same relative width, so the
  band is too wide where the model is good and too narrow where it is not.
  Holdout coverage is 0.375 at Gold Beach (N = 40) and 0.479 at Newport
  (N = 48), the two sites with the lowest R², against 0.85-0.93 at the other
  sites with more than two rows. Validation showed the same pattern (0.629
  and 0.553), so this was visible before the holdout.
- **At onsets.** The band contains the measurement at 10 of the 27 holdout
  onsets (0.37; validation 0.41). An onset is a jump the forecast did not
  anticipate, so the measurement usually lies above the band. Warning of
  onsets is the spike alert's job, not the band's.

#### One-model test: the classifier stays

| Holdout (27 onsets, 292 quiet weeks) | Band P(DA >= 20) >= 0.084 | Hybrid alert |
|---|---:|---:|
| Onsets caught | 16 | **19** |
| Quiet weeks alarmed | 27.1% | 29.8% |
| Onset AUC | 0.747 | **0.773** |

Band minus classifier onset AUC: -0.027, 95% CI [-0.126, +0.067]. Two of the
three declared conditions fail: the CI's lower bound is below -0.05, and the
band catches 16 onsets where 18 were required. The alarm-rate condition holds.
So the pooled spike classifier stays as the separately reported spike alert,
and the band supplies the regression line's uncertainty.

The difference is not significant either way; the rule asked for evidence that
the band is not clearly worse, and the holdout's 27 onsets cannot give it. As
expected, the band's P(DA >= 20) ranks onsets almost exactly as the forecast
itself does (holdout AUC 0.747 vs 0.744), because without Mondrian it rises
with the forecast against a slowly changing error set.

## Serving

The dashboard shows this band with every forecast (`backend/band.py`):

- **Error history:** `results/chronological/chronological_regression_ensemble_20030101_20240101.parquet`,
  the all-years run above. A forecast anchored at *t* is calibrated by the rows of that
  run measured on or before *t*, exactly as `past_residual_bands` selects them;
  `tests/test_band_service.py` checks every banded forecast (3,480) against
  `band_predictions.parquet`: the same errors and P(DA >= 20), and edges equal to within
  1e-12 (`expm1`/`log1p` differ in the last bits across platforms).
- **Variant:** read from `band_eval.json` (`selection.chosen`): since `1c9206c6` the most
  recent 250 errors with Mondrian calibration, 80%, chosen again at `edf64987` (first run:
  500 errors, no Mondrian).
  `backend/band.py` keeps one error history per predicted category and selects errors
  exactly as `past_residual_bands` does: same-category errors known by the anchor if
  there are at least 30, else all errors, then the most recent 250. It lives in
  `backend/` rather than `forecasting/` because the band is computed per request, not
  stored, and `forecasting/` is part of the precomputed store's config hash. A forecast
  with fewer than 30 earlier errors (before about 2006) gets no band.
- **Category chances** (`forecasting.conformal.category_probabilities`) spread the same
  errors over [0,5) [5,20) [20,40) [40,inf); they are the line's, not the alert's.
- **Keep it current:** the history holds the engine's own forecasts, so it must be
  regenerated (one Hyak run of the command under Inputs) whenever the engine's
  predictions change, as for the golden baseline.
