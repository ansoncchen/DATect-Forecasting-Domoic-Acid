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
The holdout classifier AUC (0.773) is already published; the band's holdout
numbers were not computed before this commit.

## Results

Filled in after the run.
