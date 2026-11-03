# Dashboard API and error band Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the dashboard everything in section 6 of its design from the API: the 80% conformal error band on every forecast, graded alert levels, recent measurements, a track-record endpoint, per-site reliability and the protocol constants. No metric is left for the browser to compute.

**Architecture:**
- **Band at serve time.** It is computed from a committed error history (the all-years chronological run), using the `forecasting/conformal.py` primitives and the variant `docs/ERROR_BAND.md` chose on validation (most recent 500 errors, no Mondrian). Precomputed forecasts stay as they are.
- **Track record and reliability** come from the committed chronological window parquets joined with `results/conformal_band/band_predictions.parquet`, summarized in one backend module.
- **The enhanced forecast becomes regression-only**: the 4-category classifier is no longer called. Its category probabilities come from the band.

**Tech Stack:** FastAPI, pandas, numpy, scikit-learn metrics, Plotly (server-rendered figures), pytest with `fastapi.testclient`.

**Spec:** `docs/superpowers/specs/2026-11-03-dashboard-decision-support-design.md` (branch `claude/dashboard-redesign-26311d`), sections 3 and 6; `docs/ERROR_BAND.md`.

## Global Constraints

- **Thresholds.**
  - Spike = actual DA ≥ 20 µg/g (`config.SPIKE_THRESHOLD`).
  - The regression alert fires at forecast ≥ 12 (`config.SPIKE_REGRESSION_ALERT_THRESHOLD`).
  - The classifier alert fires at probability ≥ 0.10 (`config.SPIKE_ALERT_PROB_THRESHOLD`).
  - A value at a threshold counts.
- **Categories** are left-closed bins `[0,5) [5,20) [20,40) [40,inf)` (`config.DA_CATEGORY_BINS`), labelled `Low (<5)`, `Moderate (5–<20)`, `High (20–<40)`, `Extreme (≥40)`.
- **The band.**
  - It is 80%, on log1p errors.
  - Each forecast is calibrated only on forecasts whose measurement is dated on or before its anchor (`test_date - 7 days`).
  - It uses the most recent 500 errors, without Mondrian; read this from `results/conformal_band/band_eval.json` → `selection.chosen`.
  - At least 30 errors are required; otherwise there is no band.
- **Alert levels.**
  - Quiet = no alert condition holds.
  - Watch = today's hybrid alert (classifier ≥ 0.10 or forecast ≥ 12).
  - Warning = classifier ≥ 0.40.
  - 0.40 is a **placeholder**, labelled as such in `/api/config`, until the onset experiments choose tiers on validation.
  - No level when the last known value is already ≥ 20.
- **No reported number changes.** The engine's predictions are untouched. New code in `forecasting/` is new functions only, and `tests/test_chronological_eval_smoke.py` stays green.
- **Don't touch `frontend/`**; the dashboard-redesign session owns it. Keep the response keys the current dashboard reads: `regression.*`, `graphs.level_range.{type,gradient_plot}` and `graphs.category_range.{predicted_category,class_probabilities,category_labels}`.
- **API handlers are plain `def`.** Unknown sites go through `resolve_site`.
- **Expected side effect.** Edits to `forecasting/*.py`, `config.py`, the tuned JSON and `backend/visualizations.py` change the precomputed store's `config_hash`. Stored rows go stale and are computed on demand. The production store is rebuilt once, after the engine cleanup plan.
- **Tests:** `.venv/bin/python -m pytest tests/ -q` stays green and order-independent.

## Review Focus

1. **A forecast with no band.** Dates with fewer than 30 earlier errors (before about 2006), or an engine result of `None`, must return `band: null` and `category_probabilities: null`, not raise (Task 4 test).
2. **Last value already ≥ 20.** `alert.level` is `null` with `level_reason: "already_at_or_above_threshold"`, while the classifier and regression parts are still reported (Tasks 2 and 4).
3. **Site aliases in new endpoints.** `/api/sites/{site}/history` and `/api/track-record?site=` accept the same aliases and casing as the other endpoints, and return 404 for an unknown site (Task 6).
4. **Bad query values.** A window other than `holdout`/`validation` returns 422. `since` later than `until` returns an empty `rows` list, not 500 (Task 6).
5. **NaN or negative forecasts.** A negative forecast is banded from 0 (as `band_from_residuals` does). A NaN forecast gets no band (Task 1).

---

### Task 1: Error history and serve-time band

**Files:**
- Modify: `forecasting/conformal.py` (add `ErrorHistory`, `category_probabilities`)
- Create: `backend/band.py`
- Modify: `.dockerignore` (allow `results/conformal_band`)
- Test: `tests/test_conformal.py`, `tests/test_band_service.py`

**Interfaces:**
- Produces, in `forecasting/conformal.py`:
  - `ErrorHistory(frame)`. `frame` needs `date`, `actual_da` and `predicted_da`. It stores the date-sorted (stable) dates and log residuals.
  - `ErrorHistory.known_by(anchor_date, window: int | None = None) -> np.ndarray`: residuals of rows with `date <= anchor_date`, in date order; the last `window` of them when given. This is the same selection `past_residual_bands` makes.
  - `ErrorHistory.band(pred: float, anchor_date, coverage: float = 0.8, window: int | None = None, threshold: float = 20.0, min_calibration: int = 30) -> dict | None`. Returns `{"lo", "hi", "level", "n_calibration", "p_exceed"}`. Returns `None` for a NaN `pred` or fewer than `min_calibration` errors.
  - `category_probabilities(pred: float, residuals, edges=(5.0, 20.0, 40.0)) -> list[float]`. Gives the share of residuals that put `log1p(max(pred, 0)) + r` back through `expm1` into each left-closed bin. The list has 4 entries and sums to 1.
- Produces, in `backend/band.py`:
  - `ERROR_HISTORY_PATH = "results/chronological/chronological_regression_ensemble_20030101_20240101.parquet"`
  - `BAND_EVAL_PATH = "results/conformal_band/band_eval.json"`
  - `chosen_variant() -> dict` returns `{"coverage": 0.8, "window": 500}`. It raises `ValueError` if `band_eval.json` chose Mondrian.
  - `forecast_band(predicted_da: float, anchor_date) -> dict | None` returns `{"lo", "hi", "level", "n_calibration", "p_exceed", "category_probabilities"}`. It loads lazily and caches the history; `None` if the history file is missing.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_band_service.py
def test_serve_time_band_equals_the_evaluated_band():
    rows = pd.read_parquet("results/conformal_band/band_predictions.parquet").dropna(subset=["band_lo"])
    for _, r in rows.sample(50, random_state=0).iterrows():
        b = band.forecast_band(r.predicted_da, r.anchor_date)
        assert (b["lo"], b["hi"], b["p_exceed"]) == (r.band_lo, r.band_hi, r.p_exceed)  # exact
        assert b["n_calibration"] == r.n_calibration and b["level"] == 0.8

def test_no_band_before_thirty_errors_or_for_nan():
    assert band.forecast_band(5.0, pd.Timestamp("2005-01-01")) is None
    assert band.forecast_band(float("nan"), pd.Timestamp("2020-01-01")) is None

def test_category_probabilities_sum_to_one_and_respect_bins():
    p = band.forecast_band(10.0, pd.Timestamp("2021-06-01"))["category_probabilities"]
    assert len(p) == 4 and sum(p) == pytest.approx(1.0)
```

In `tests/test_conformal.py`:
- `test_category_probabilities_put_zero_errors_in_the_forecasts_bin`: `category_probabilities(20.0, np.zeros(10)) == [0, 0, 1, 0]`. 20 is in `[20, 40)`.
- `test_known_by_matches_past_residual_bands`: on `_frame()`, `len(ErrorHistory(f).known_by(anchor_i, 100)) == past_residual_bands(f, window=100).n_calibration[i]`.

- [ ] **Step 2: Run them and see them fail**

Run: `.venv/bin/python -m pytest tests/test_band_service.py tests/test_conformal.py -q`
Expected: FAIL. `ErrorHistory`, `category_probabilities` and `backend.band` don't exist yet.

- [ ] **Step 3: Implement**

Write `ErrorHistory` and `category_probabilities` in `forecasting/conformal.py`, reusing `log_residuals`, `band_from_residuals` and `exceedance_probability`. Don't change `past_residual_bands`. Then write `backend/band.py`, which caches an `ErrorHistory` (`functools.lru_cache`). Add `!results/conformal_band` to `.dockerignore` next to the other `results/` lines.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_band_service.py tests/test_conformal.py -q`
Expected: PASS. The 50 serve-time bands equal `band_predictions.parquet` exactly.

- [ ] **Step 5: Commit** with the message `band: serve-time conformal band from the committed error history`.

### Task 2: Alert levels

**Files:**
- Modify: `config/tuned_hyperparameters.json`. Add `"alert_warning_prob_cutoff": 0.4` to `global`, and a `provenance` note: "PLACEHOLDER from the dashboard design; the onset experiments will set tiers on validation".
- Modify: `config.py` (`ALERT_WARNING_PROB_CUTOFF`, next to `SPIKE_ALERT_PROB_THRESHOLD`)
- Modify: `forecasting/metrics.py` (add `alert_parts`)
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces: `alert_parts(classifier_probability: float | None, forecast: float, last_known: float | None, *, prob_cutoff: float, regression_cutoff: float, warning_cutoff: float, threshold: float = 20.0) -> dict`. Returns `{"classifier_probability", "by_classifier", "by_regression", "alert", "level", "level_reason"}`.
  - `level` is one of `"Quiet"`, `"Watch"`, `"Warning"`, or `None`.
  - `level_reason` is `None`, `"already_at_or_above_threshold"` or `"no_classifier"`.
  - A missing classifier probability counts as not firing.

- [ ] **Step 1: Write the failing test** `test_alert_parts_levels_and_boundaries`:

```python
kw = dict(prob_cutoff=0.10, regression_cutoff=12.0, warning_cutoff=0.40)
assert alert_parts(0.05, 3.0, 2.0, **kw)["level"] == "Quiet"
assert alert_parts(0.10, 3.0, 2.0, **kw)["level"] == "Watch"      # at the cutoff counts
assert alert_parts(0.05, 12.0, 2.0, **kw)["level"] == "Watch"     # regression rule alone
assert alert_parts(0.40, 3.0, 2.0, **kw)["level"] == "Warning"
done = alert_parts(0.90, 30.0, 20.0, **kw)
assert done["level"] is None and done["level_reason"] == "already_at_or_above_threshold"
assert done["alert"] is True and done["by_classifier"] and done["by_regression"]
```

- [ ] **Step 2: Run it and see it fail**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -q -k alert_parts`
Expected: FAIL (`alert_parts` doesn't exist yet).

- [ ] **Step 3: Implement `alert_parts`** with `exceeds` from the same module, plus the config and JSON keys.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_metrics.py tests/test_tuned_config.py -q`
Expected: PASS.

- [ ] **Step 5: Commit** with the message `alerts: Quiet/Watch/Warning levels (Warning cutoff is a placeholder)`.

### Task 3: Band chart

**Files:**
- Modify: `backend/visualizations.py`. Replace `generate_gradient_uncertainty_plot` with `generate_band_plot`; the only caller is `backend/api.py`.
- Test: `tests/test_visualizations_math.py`. Replace the two gradient-plot tests.

**Interfaces:**
- Produces: `generate_band_plot(band: dict, forecast: float, xgb_prediction: float | None = None, rf_prediction: float | None = None, threshold: float = 20.0) -> str` (Plotly JSON). The figure has:
  - a horizontal bar from `band["lo"]` to `band["hi"]`, named `80% range (past errors)` (the level comes from `band["level"]`);
  - a forecast marker named `Forecast`;
  - optional `XGBoost` and `Random Forest` markers;
  - a dashed vertical line at the threshold, annotated `20 µg/g spike`;
  - the x-axis title `DA (µg/g)`.

- [ ] **Step 1: Write the failing test** `test_band_plot_draws_range_forecast_and_threshold`. Parse the JSON and assert:
  - a trace named `80% range (past errors)` spans x ∈ [lo, hi];
  - the `Forecast` trace's x equals the forecast;
  - the x-axis title is `DA (µg/g)`;
  - a shape sits at x = 20;
  - no `μg/L` appears anywhere in the JSON.
- [ ] **Step 2: Run it and see it fail**

Run: `.venv/bin/python -m pytest tests/test_visualizations_math.py -q`
Expected: FAIL.
- [ ] **Step 3: Implement `generate_band_plot`**, delete `generate_gradient_uncertainty_plot` and update the import in `backend/api.py`.
- [ ] **Step 4: Run tests**

Run the same command. Expected: PASS.
- [ ] **Step 5: Commit** with the message `viz: draw the 80% conformal range, not the bootstrap interval`.

### Task 4: Regression-only enhanced forecast with band, alert and recent measurements

**Files:**
- Modify: `backend/api.py`. Change `generate_enhanced_forecast`; add `_raw_measurements`, `_recent_measurements` and `_forecast_payload`.
- Test: `tests/test_precomputed.py` (enhanced-endpoint tests), new `tests/test_forecast_payload.py`

**Interfaces:**
- Consumes:
  - `band.forecast_band` (Task 1);
  - `metrics.alert_parts` and `config.ALERT_WARNING_PROB_CUTOFF` (Task 2);
  - `generate_band_plot` (Task 3);
  - `_single_forecast(site, date, "regression", model_type)`, which already exists.
- Produces:
  - `_raw_measurements() -> pd.DataFrame`, cached with `lru_cache`. It wraps `load_raw_da_measurements()`, one row per site-day at the max.
  - `_recent_measurements(site: str, anchor_date, n: int = 6) -> list[dict]`: `[{"date": "YYYY-MM-DD", "value": float}]`, the newest last, all with `date <= anchor_date`.
  - `_forecast_payload(site: str, forecast_date: pd.Timestamp, model_type: str) -> dict`. Its keys:
    - `success`, `forecast_date`, `site`, `anchor_date`, `regression`, `ensemble`, `source`, `model_version`;
    - `band`, `band_p_exceed`, `category_probabilities`;
    - `alert` (the `alert_parts` dict);
    - `last_measurement` (`{"date", "value"}` or `None`), `recent_measurements`;
    - `graphs.level_range` = `{"type": "gradient_uncertainty", "gradient_plot": <band plot>, "band": band}`, omitted when there's no band;
    - `graphs.category_range` = `{"predicted_category": <forecast's bin>, "class_probabilities": <band-derived>, "category_labels": [...], "source": "range"}`, omitted when there's no band.
  - There is no `classification` key, and `_single_forecast` is called once per request.

- [ ] **Step 1: Write the failing tests** with a stub engine via `monkeypatch.setattr(api, "get_forecast_engine", ...)`, following `test_precomputed.py`:
  - `test_enhanced_forecast_calls_the_engine_once_and_returns_the_band`: the call count is 1; `band` equals `band.forecast_band(pred, anchor)`; `"classification" not in body`; `graphs.category_range.source == "range"`.
  - `test_enhanced_forecast_reports_alert_parts_and_level`: the stub returns `spike_probability=0.5` and `ensemble_prediction=3.0`, so `alert.level == "Warning"` and `alert.by_regression is False`.
  - `test_recent_measurements_stop_at_the_anchor`: at most 6 entries, all dated `<= anchor_date`, matching `load_raw_da_measurements()` for that site.
  - `test_no_band_for_early_dates_or_missing_results`: the stub returns `None`, or a 2004 anchor, giving `success` without a 500, `band is None`, and `"level_range" not in graphs`.
- [ ] **Step 2: Run them and see them fail**

Run: `.venv/bin/python -m pytest tests/test_forecast_payload.py tests/test_precomputed.py -q`
Expected: FAIL.
- [ ] **Step 3: Implement `_forecast_payload`**, and make `generate_enhanced_forecast` return it. The last known value for `alert_parts` is `last_measurement.value`.
- [ ] **Step 4: Run tests**

Run the same command. Expected: PASS.
- [ ] **Step 5: Commit** with the message `api: regression-only enhanced forecast with the band, alert levels and recent measurements`.

### Task 5: Precompute only what the API reads

**Files:**
- Modify: `datect_spark/forecast_serving.py` (the precompute task list becomes `("regression",)`), `backend/precomputed.py` (`write_sample`, `check_store` expectations)
- Test: `tests/test_forecast_serving.py`, `tests/test_precomputed.py`

**Interfaces:** the store's `forecasts` table holds `task == "regression"` rows only, 10,950 at full size.

- [ ] **Step 1: Write the failing test** `test_precompute_tasks_are_regression_only`: the task list the precompute iterates is `("regression",)`, and `write_sample` writes only regression rows.
- [ ] **Step 2: Run it and see it fail**

Run: `.venv/bin/python -m pytest tests/test_forecast_serving.py tests/test_precomputed.py -q`
Expected: FAIL.
- [ ] **Step 3: Change the task list and the sample writer.** Update the docstring counts: 10,950 entries, about half the old precompute time.
- [ ] **Step 4: Run tests**

Run the same command. Expected: PASS.
- [ ] **Step 5: Commit** with the message `serving: precompute regression forecasts only (the classifier call is gone)`.

### Task 6: Track record and site history

**Files:**
- Create: `backend/track_record.py`
- Modify: `backend/api.py` (two endpoints)
- Test: `tests/test_track_record.py`

**Interfaces:**
- Produces, in `backend/track_record.py`:
  - `window_rows(window: Literal["holdout", "validation"], site: str | None = None) -> pd.DataFrame`. It reads the chronological window parquet, whose file tag comes from `tuned_config.get_eval_windows()`, and left-joins `band_lo`, `band_hi` and `p_exceed` from `band_predictions.parquet` on `(date, site)`. Columns: `date, anchor_date, site, actual_da, predicted_da, naive_prediction, predicted_da_linear, spike_probability, spike_alert, band_lo, band_hi, p_exceed`.
  - `summarize(rows: pd.DataFrame) -> dict`. Keys:
    - `n`, `r2`, `log_r2`, `mae` (sklearn, like `chronological_eval.py`);
    - `band = {coverage, coverage_at_onsets, median_width}`;
    - `onsets = {n, caught, quiet, quiet_alarmed}` (from `metrics.onset_scores` on `naive_prediction` and `spike_alert`);
    - `alarms = {n, followed_by_jump, false_below_5, false_5_to_10, false_near_miss}`, counting weeks whose last known value was below 20. The `false_*` counts split the false alarms by measured DA: `[0,5)`, `[5,10)` and `[10,20)` (near misses);
    - `levels = {"Quiet"|"Watch"|"Warning": {weeks, followed_by_jump}}`, built with `alert_parts`.
  - `site_history(site: str, since: str = "2019-01-01", until: str | None = None) -> list[dict]`: rows from `band_predictions.parquet` with the same columns plus `alert` and `level`.
- Produces, in `backend/api.py`:
  - `GET /api/track-record?window=holdout&site=` returns `{"window": {"name", "start", "end"}, "summary", "per_site": {site: summary}, "rows"}`;
  - `GET /api/sites/{site}/history?since=&until=` returns `{"site", "rows"}`.

- [ ] **Step 1: Write the failing tests.** Pin the committed results; these numbers were verified on 2026-11-03.

```python
def test_holdout_summary_matches_the_reported_numbers():
    s = tr.summarize(tr.window_rows("holdout"))
    assert s["n"] == 409 and s["r2"] == pytest.approx(0.516, abs=5e-4)
    assert s["onsets"] == {"n": 27, "caught": 19, "quiet": 292, "quiet_alarmed": 87}
    assert s["alarms"] == {"n": 106, "followed_by_jump": 19, "false_below_5": 21,
                           "false_5_to_10": 18, "false_near_miss": 48}
    assert s["band"]["coverage"] == pytest.approx(0.802, abs=5e-4)
    assert s["band"]["median_width"] == pytest.approx(15.08, abs=0.01)
    assert s["levels"] == {"Quiet": {"weeks": 213, "followed_by_jump": 8},
                           "Watch": {"weeks": 73, "followed_by_jump": 9},
                           "Warning": {"weeks": 33, "followed_by_jump": 10}}

def test_validation_and_per_site_reliability():
    v = tr.summarize(tr.window_rows("validation"))
    assert v["onsets"]["caught"] == 27 and v["band"]["coverage"] == pytest.approx(0.819, abs=5e-4)
    assert (v["alarms"]["false_below_5"], v["alarms"]["false_5_to_10"], v["alarms"]["false_near_miss"]) == (23, 13, 37)
    g = tr.summarize(tr.window_rows("holdout", "Gold Beach"))
    assert g["n"] == 40 and g["band"]["coverage"] == pytest.approx(0.375)
```

Plus `test_recomputed_alert_equals_the_engines_spike_alert`. For every holdout and validation row, `alert_parts(spike_probability, predicted_da, naive_prediction, ...)["alert"] == spike_alert`. This proves the levels use the engine's own rule.

Endpoint tests:
- `test_track_record_rejects_unknown_window`: 422.
- `test_history_resolves_aliases_and_404s_unknown_sites`.
- `test_history_with_since_after_until_is_empty`.

- [ ] **Step 2: Run them and see them fail**

Run: `.venv/bin/python -m pytest tests/test_track_record.py -q`
Expected: FAIL.
- [ ] **Step 3: Implement the module and both endpoints** (`window` typed `Literal[...]`).
- [ ] **Step 4: Run tests**

Run the same command. Expected: PASS.
- [ ] **Step 5: Commit** with the message `api: track record and site history from the committed predictions, summaries computed server-side`.

### Task 7: Reliability in the summary, protocol constants in config, states fixed

**Files:**
- Modify: `backend/api.py` (`_load_metrics_summary`, `get_config`, `_compute_summary`'s region map)
- Modify: `config.py` (add `SITE_STATES`: Kalaloch, Quinault, Copalis, Twin Harbors and Long Beach → `"WA"`; Clatsop Beach, Cannon Beach, Newport, Coos Bay and Gold Beach → `"OR"`)
- Test: `tests/test_api_robustness.py`

**Interfaces:**
- `/api/metrics/summary` adds a `reliability` block: `{window: {"summary": summarize(...), "per_site": {...}}}` for `holdout` and `validation`. `source` lists the parquet and JSON files it was computed from.
- `/api/config` adds:
  - `regression_alert_threshold`, `classifier_alert_cutoff`, `band_level` (0.8);
  - `alert_levels`: a list of `{"name", "rule", "cutoff", "placeholder"}`;
  - `category_bins` (`[0, 5, 20, 40, null]`) and `category_labels`;
  - `eval_windows`, from `get_eval_windows()`;
  - `newest_measurement_date` (the max of `_raw_measurements().date`);
  - `sites_north_to_south`: the sites sorted by latitude, descending, taken from the site coordinates the map endpoint already uses;
  - `latest_forecast_week` (the max weekly `date` in `final_output.parquet`, the dashboard's default).

- [ ] **Step 1: Write the failing tests:**
  - `test_config_exposes_protocol_constants`: the values equal the config attributes; the Warning entry has `placeholder: true` and `cutoff == 0.4`; `sites_north_to_south` starts with `Kalaloch` and ends with `Gold Beach`.
  - `test_summary_reliability_per_site`: holdout Gold Beach coverage is 0.375 and Newport's is about 0.479.
  - `test_region_rollup_puts_clatsop_and_cannon_in_oregon`.
- [ ] **Step 2: Run them and see them fail**

Run: `.venv/bin/python -m pytest tests/test_api_robustness.py -q`
Expected: FAIL.
- [ ] **Step 3: Implement**, reusing `track_record.summarize`. `_compute_summary` reads `config.SITE_STATES`.
- [ ] **Step 4: Run tests**

Run the same command. Expected: PASS.
- [ ] **Step 5: Commit** with the message `api: per-site reliability, protocol constants in /api/config, Oregon sites fixed`.

### Task 8: All sites in one call

**Files:**
- Modify: `backend/api.py` (`GET /api/forecast/all`)
- Test: `tests/test_forecast_payload.py`

**Interfaces:**
- Consumes `_forecast_payload` (Task 4) and `latest_forecast_week` (Task 7).
- Produces: `GET /api/forecast/all?date=YYYY-MM-DD&model=ensemble` (the date defaults to the latest forecast week). It returns `{"date", "sites": [{"site", "predicted_da", "band", "alert", "last_measurement", "source"}]}`, ordered as `/api/config` `sites_north_to_south` (Task 7).

- [ ] **Step 1: Write the failing test** `test_all_sites_returns_every_site_for_one_week`. With a stub engine: 10 entries, each with `alert.level`, and the default date equals `latest_forecast_week`.
- [ ] **Step 2: Run it and see it fail**

Run: `.venv/bin/python -m pytest tests/test_forecast_payload.py -q -k all_sites`
Expected: FAIL.
- [ ] **Step 3: Implement** with a plain loop over sites. Each one is a store hit once the store is current.
- [ ] **Step 4: Run tests**

Run the same command. Expected: PASS.
- [ ] **Step 5: Commit** with the message `api: one week's forecast for every site in one call`.

### Task 9: Server-rendered charts say what they show

**Files:**
- Modify: `backend/api.py`. `/api/historical/{site}` returns raw measurements, `{"site", "series": "raw measurements, one per site-day at the max", "rows": [{"date", "value"}]}`. Drop the nonexistent `da-category` column.
- Modify: `backend/visualizations.py`. The waterfall and DA-vs-PN comparison figures get the subtitle `Weekly maximum; gaps filled by decay`.
- Test: `tests/test_api_robustness.py`, `tests/test_visualizations_math.py`

- [ ] **Step 1: Write the failing tests:**
  - `test_historical_returns_raw_measurements`: the row count equals `load_raw_da_measurements()` for the site.
  - `test_weekly_figures_are_labelled_gap_filled`: the title text contains `gaps filled`.
- [ ] **Step 2: Run them and see them fail**

Run: `.venv/bin/python -m pytest tests/test_api_robustness.py tests/test_visualizations_math.py -q`
Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests**

Run the same command. Expected: PASS.
- [ ] **Step 5: Commit** with the message `api: raw measurements on /api/historical; weekly charts labelled gap-filled`.

### Task 10: Docs and the full check

**Files:**
- Modify:
  - `docs/ERROR_BAND.md`: a "Serving" section covering the error history file, the chosen variant, the 30-error minimum, and that the history must be regenerated whenever predictions change;
  - `CLAUDE.md`: the single-forecast bullet (the dashboard shows the conformal band; the engine's quantile intervals are unused until the engine cleanup), the API section, the `SITE_STATES` note, and stale names from #23;
  - `docs/VISUALIZATIONS_GUIDE.md`: the interval chart;
  - `docs/DATA_ENGINEERING.md`: #23's renames (`DATECT_PRECOMPUTED_DIR`, `backend/precomputed.py`) and the status table.

- [ ] **Step 1: Find stale references.**

Run: `grep -rn "precomputed_forecasts\|DATECT_FORECAST_TABLE\|gradient_uncertainty_plot\|bootstrap Q05\|grid_search_weights_clip\|validate_tuned_on_holdout" --include=*.md . | grep -v "docs/superpowers\|CORRECTED_NUMBERS"`
Expected: a list to fix. Leave `CORRECTED_NUMBERS.md`'s history and the plan/spec files alone.
- [ ] **Step 2: Edit the docs.**
- [ ] **Step 3: Run the full suite and lint.**

Run: `.venv/bin/python -m pytest tests/ -q && .venv/bin/ruff check backend forecasting tests`
Expected: everything passes, including `test_chronological_eval_smoke.py`, which shows no reported number moved.
- [ ] **Step 4: Run a live smoke test.** Start the API (`.claude/launch.json`, preview tools), `POST /api/forecast/enhanced` for Newport 2023-06-05, and confirm it returns `band`, `alert.level`, and a band plot without `μg/L`.
- [ ] **Step 5: Commit** with the message `docs: serving the error band; sync names after #23`. Then open the PR; the user merges.

---

**Not in this plan** (the engine cleanup plan, together with batch 3):
- deleting the 4-category classifier, the logistic path and the bootstrap/quantile interval code;
- one forecast function and one blend function;
- one DA CSV reader;
- removing per-anchor tuning;
- stamping results JSONs with commit and config hash;
- rebuilding the production store.
