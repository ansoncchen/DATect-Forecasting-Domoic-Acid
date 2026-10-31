# Forecasting Pipeline Deep Dive

How the forecasting engine (`forecasting/raw_forecast_engine.py`, class `RawForecastEngine`) turns the processed dataset and the raw DA measurements into a 7-day-ahead forecast, and which safeguards keep it leak-free.

**Related docs:** [`DATA_PIPELINE_DETAILED.md`](DATA_PIPELINE_DETAILED.md) builds the input parquet. [`QUICKSTART_FOR_REVIEWERS.md`](QUICKSTART_FOR_REVIEWERS.md) lists the evaluation scripts. Current results: see [`CORRECTED_NUMBERS.md`](CORRECTED_NUMBERS.md).

## Where configuration lives

| Source | Holds |
|---|---|
| `config/tuned_hyperparameters.json` | Single source of truth for tuned values: XGB/RF base params, spike thresholds, per-anchor tuning switch, and every per-site block (XGB/RF overrides, feature subset, ensemble weights, clip quantile). Loaded by `forecasting/tuned_config.py`. |
| `config.py` | Paths, horizon (`FORECAST_HORIZON_DAYS = 7`), seeds, category bins, decay-fill constants, sampling fractions, and the `DATECT_*` env-var overrides. Reads the JSON's `global` block. |
| `forecasting/per_site_models.py` | Exposes the JSON's `per_site` block as `SITE_SPECIFIC_CONFIGS` and applies the env-var overlays (`DATECT_FEATURE_SUBSET_MODE`, `DATECT_OAD_ON_SMALL_N`, `DATECT_HPARAM_OVERRIDE_JSON`, `DATECT_CLIP_Q_OVERRIDE`). |

Because `config.py` and `per_site_models.py` read `DATECT_*` variables at import time, experiments that change them run the engine in a fresh subprocess (`scripts/eval/_subprocess.py`).

## 1. Inputs

- **Raw DA measurements** — `raw_data_forecaster.load_raw_da_measurements()` reads the 10 CSVs in `config.ORIGINAL_DA_FILES` (`data/raw/da-input/`). Two schemas are detected per file: `CollectDate` + `Domoic Result`, or `Harvest Month/Date/Year` + `Domoic Acid`. Values and dates are parsed by `forecasting/measurements.py` (below-detection `"<1"` becomes LOD/2 = 0.5; see the data pipeline doc). Several samples on one site-day (about 1,200 site-days; the CSVs record no species or station) collapse to their maximum, like the weekly MAX used for training targets. Output: `[date, site, da_raw]`, one row per site-day. These are the only evaluation targets.
- **Processed weekly dataset** — `data/processed/final_output.parquet` (Monday rows, environmental + satellite + OAD columns, and the gap-filled `da`/`pn`).

## 2. Feature frame (`_load_feature_frame` → `build_raw_feature_frame`)

1. `add_temporal_features` (`feature_utils.py`): `sin_day_of_year`, `cos_day_of_year`, `month`.
2. `aggregate_raw_to_weekly`: raw DA bucketed to the Monday of its week using **MAX**, then left-joined onto the processed rows by `(site, date)`. Most rows have `da_raw = NaN`.
3. `pn_log = log1p(pn)`.
4. Persistence: `last_observed_da_raw` (per-site forward fill of `da_raw`) and `weeks_since_last_spike` (weeks since `da_raw >= SPIKE_THRESHOLD`; 999 if none).
5. Rolling stats (`USE_ROLLING_FEATURES`), computed on `shift(1)` of `last_observed_da_raw` so the current row never contributes: window 4 → mean, std, max; windows 8 and 12 → std, max.
6. Observation-order lags (`raw_data_processor.py`, `LAG_FEATURES = [1, 2, 3, 4]`, override with `DATECT_LAG_FEATURES`): `da_raw_prev_obs_1..4` are the Nth most recent real measurements strictly before the row date; `da_raw_prev_obs_2_weeks_ago` and `_3_weeks_ago` give their age; `da_raw_prev_obs_diff_1_2` is the latest change. Grid-shift lags would be mostly NaN on sparse sampling.

After dropping metadata, targets and `ZERO_IMPORTANCE_FEATURES` (`lat`, `lon`, `modis-par`, `modis-k490`, `chla-anom`, `modis-chla`), the frame offers **44 candidate features**: 7 environmental/satellite (`oni`, `pdo`, `discharge`, `beuti`, `modis-flr`, `modis-sst`, `sst-anom`), `pn` and `pn_log`, 16 OAD columns, 3 calendar, 2 persistence, 7 rolling and 7 lag features. Per-site `feature_subset` lists then narrow this (Clatsop Beach has no subset and uses all 44).

## 3. Per-prediction split

Every prediction trains fresh models; nothing is pre-trained.

- **Anchor** = `test_date − FORECAST_HORIZON_DAYS` (7 days). The anchor is the knowledge cutoff.
- **Training rows** (`get_site_training_frame`): the site's rows with `date <= anchor`. With `USE_INTERPOLATED_TRAINING` (default true, `DATECT_USE_INTERPOLATED_TRAINING`), every row with a non-null gap-filled `da` is kept; the target is the real `da_raw` where one exists, otherwise `da`. `_is_interpolated` marks the filled rows. At least `MIN_TRAINING_SAMPLES = 10` rows are required.
- **Test row** (`get_site_anchor_row`): the processed row closest to, and not after, the anchor (at most 28 days earlier). Its `date` is reset to `test_date` so only the calendar features describe the target date.
- **Persistence recompute** (`recompute_test_row_persistence_features`): `last_observed_da_raw` and `weeks_since_last_spike` are overwritten from **real** training measurements only. Rolling and lag features are taken from the anchor row as-is; they already use only earlier observations.
- **Leak check** (`_verify_no_data_leakage`): raises if any training date is after the anchor or the test date is not after it.

## 4. Preprocessing (`create_transformer`)

Drop `date`, `site`, `da_raw`, `da`, `_is_interpolated`, the zero-importance list and (when `USE_PER_SITE_MODELS`) every column outside the site's `feature_subset`. `compute_site_drop_cols` logs a warning if a subset names a column the data lacks. Remaining numeric columns that are not entirely NaN go through `SimpleImputer(median)` → `MinMaxScaler`, fit on the training rows only.

## 5. Models (`raw_model_factory.py`)

All fitted models are seeded with `config.MODEL_SEED` (42).

| Model | Details |
|---|---|
| XGBoost | `global.xgb_base_params` merged with the site's `xgb_params`. With `USE_MONOTONIC_CONSTRAINTS` (default true), non-decreasing constraints on `last_observed_da_raw`, `raw_obs_roll_max_4`, `raw_obs_roll_mean_4`. |
| Random Forest | `global.rf_base_params` merged with the site's `rf_params` (set for Coos Bay, Cannon Beach, Gold Beach, Newport). |
| Naive persistence | Last real DA at or before the anchor (`PERSISTENCE_MAX_DAYS = None`, no age cap). External baseline, never blended. If a site has no prior real measurement the point is dropped. |
| Ridge (`linear`) | `LINEAR_REGRESSION_ALPHA` on the full preprocessed feature set; a competitor, not part of the ensemble. Logistic regression is its classification counterpart. |

**Post-processing** (`_postprocess`, applied to XGB, RF and Ridge): floor at 0. There is no upper clip: `prediction_clip_q`, `prediction_clip_max` and `PREDICTION_CLIP_Q` are null. The clip they once set capped forecasts at a quantile of the training `da_raw`, which is mostly gap-filled (Coos Bay q95 81 µg/g vs 145 over real measurements), and removing it raised R² in every window (`docs/CORRECTED_NUMBERS.md`). `DATECT_CLIP_Q_OVERRIDE=<q>` restores a clip at quantile *q* for ablations.

## 6. Ensemble weights

`ensemble = w_xgb · XGB + w_rf · RF` with per-site `(w_xgb, w_rf, w_naive)` from the JSON (`w_naive` is always 0). They are winners of the constrained grid search in `scripts/eval/grid_search_weights_clip.py`, which chose them jointly with the clip quantiles that were later removed. Current values:

| Site | w_xgb | w_rf |
|---|---|---|
| Copalis | 1.00 | 0.00 |
| Quinault | 1.00 | 0.00 |
| Long Beach | 1.00 | 0.00 |
| Clatsop Beach | 1.00 | 0.00 |
| Newport | 1.00 | 0.00 |
| Gold Beach | 1.00 | 0.00 |
| Twin Harbors | 0.25 | 0.75 |
| Kalaloch | 0.00 | 1.00 |
| Coos Bay | 0.00 | 1.00 |
| Cannon Beach | 0.00 | 1.00 |

The JSON is authoritative; re-check it after any re-tune. `RawForecastEngine._ensemble_weights()` serves these to evaluation, the spike alert and the API. With `USE_PER_SITE_MODELS=false` every site uses 0.5 / 0.5.

## 7. Per-anchor XGB tuning (opt-in)

Off by default: `config.PER_ANCHOR_TUNING` comes from `DATECT_PER_ANCHOR_TUNING`, defaulting to `global.per_anchor_tuning = false` in the JSON. Every reported number was produced with it off; the provenance note in the JSON records that it changed predictions without improving skill while costing minutes per forecast.

When enabled (retrospective path only, `_run_single_retrospective`), it runs if the training frame has at least `MIN_TRAINING_FOR_TUNING` (80) rows and the grid has more than one entry. It samples `CALIBRATION_FRACTION` (0.3) of the real training rows, capped at `MAX_CALIBRATION_ROWS` (20) and requiring `MIN_TUNING_SAMPLES` (10), runs a nested leak-free forecast for each with every grid entry (site `param_grid` first, then the global `PARAM_GRID`), and keeps the entry with the best R². Single forecasts never tune.

## 8. Spikes, alerts and categories

One convention everywhere (`forecasting/metrics.py`, `config.py`): **reaching a threshold counts**.

- **Spike**: actual DA `>= SPIKE_THRESHOLD` (20 µg/g). `metrics.exceeds()` is always `>=`; `metrics.detection_scores()` computes recall, precision, F1, F2, FNR, FPR from confusion counts with scikit-learn's formulas.
- **Regression alert**: ensemble prediction `>= SPIKE_REGRESSION_ALERT_THRESHOLD` (12 µg/g).
- **Hybrid spike alert** (`spike_alert` in each result): pooled spike-classifier probability `>= SPIKE_ALERT_PROB_THRESHOLD` (0.1) **or** the regression alert. The classifier (`ClassificationAdapter.train_spike_binary_classifier`) is an XGBoost model trained per prediction on real measurements from all sites up to the anchor (at least 30 rows), on every row, including weeks already above 20 (the lag and rolling DA features stay), without `last_observed_da_raw` / `weeks_since_last_spike`, with balanced class weights and `SPIKE_CLASSIFIER_PARAMS`. If it cannot be trained, the regression rule alone decides.
- **Risk categories**: `DA_CATEGORY_BINS` with `pd.cut(..., right=False)`: Low `[0, 5)`, Moderate `[5, 20)`, High `[20, 40)`, Extreme `>= 40`. A value on a boundary belongs to the higher category. `ClassificationAdapter.threshold_classify` bins the regression prediction into `predicted_category`. Classification-task single forecasts also train a dedicated balanced XGBoost 4-class classifier (`predicted_category_ml` plus class probabilities); with `model_type="logistic"` a logistic regression supplies the category instead.

## 9. Single forecasts (dashboard and `/api/forecast*`)

`generate_single_forecast(data_path, forecast_date, site, task, model_type)` calls `_run_single_raw_validation(..., return_internals=True)` — the same core every evaluation scores — so the dashboard shows the evaluated models. It then adds the selected model's prediction, the ensemble breakdown, categories, the spike alert and prediction intervals:

- `ensemble` / `xgboost`: three XGBoost models with `objective="reg:quantileerror"` and `quantile_alpha` 0.05, 0.50, 0.95, each post-processed. For `ensemble`, each quantile is blended as `w_xgb · q_XGB + w_rf · q_RF`, where `q_RF` is the same quantile of the Random Forest's individual tree predictions (RF has no quantile objective), so RF-only sites (Kalaloch, Coos Bay, Cannon Beach) also get a real band.
- `rf`, `linear`, `naive`, or if the quantile fit fails: bootstrap with `N_BOOTSTRAP_ITERATIONS` (100) resamples (`BOOTSTRAP_SUBSAMPLE_FRACTION = 1.0`), percentiles `CONFIDENCE_PERCENTILES = [5, 50, 95]`. Naive intervals have zero width.

Retrospective evaluation skips intervals; they do not affect reported metrics.

## 10. Retrospective evaluation (`run_retrospective_evaluation(task, model_type, max_test_samples)`)

1. Candidates: raw measurements on or after `MIN_TEST_DATE` whose site already has at least `max(ceil(HISTORY_REQUIREMENT_FRACTION · site total), MIN_TRAINING_SAMPLES)` measurements on or before the anchor (`HISTORY_REQUIREMENT_FRACTION = 0.33`).
2. Per-site sample of `ceil(TEST_SAMPLE_FRACTION · site total)` points (`TEST_SAMPLE_FRACTION = 0.20`), drawn with `RandomState(config.RANDOM_SEED)`.
3. Each point runs `_run_single_retrospective` in parallel (`joblib`, `N_JOBS = -1`, backend `PARALLEL_BACKEND`, default `loky`).
4. The ensemble column is blended with the per-site weights, `predicted_da` is set from the requested `model_type`, categories are derived, and duplicate `(date, site)` rows are dropped.

`scripts/eval/chronological_eval.py` instead scores every real measurement in a date window (validation `[2019-01-01, 2022-01-01)`, holdout `[2022-01-01, 2024-01-01)`, from the JSON's `eval_windows`).

**Seeds.** `RANDOM_SEED` (42) only chooses test points; evaluation scripts set it per run (123 for the paper sample, 42–46 for multi-seed). `MODEL_SEED` (42) seeds every fitted model, so serial, joblib-parallel and distributed runs train identical models regardless of the sampling seed.

## Leakage safeguards (summary)

- Training rows satisfy `date <= anchor`; `_verify_no_data_leakage` asserts it for every prediction.
- Test-row environmental features come from the last processed row at or before the anchor; satellite, climate-index and OAD columns carry their own reporting lags from dataset creation.
- `da_raw` and `da` never enter the feature matrix.
- Lags and rolling stats use only observations strictly before the row; test-row persistence and the naive baseline use real measurements only.
- Gap-filled `da` is a past-only exponential decay (see the data pipeline doc) and is used only as a training target, never as a test target.
- Imputer and scaler are fit on training rows only.

## Caching and serving

`precompute_cache.py` (run on Hyak) writes 8 task/model retrospectives (regression: ensemble, xgboost, rf, naive, linear; classification: ensemble, naive, logistic), spectral analyses and per-site correlation heatmaps under `cache/`. The API serves them when the cache is enabled and otherwise computes on demand. See [`DEPLOYMENT_GUIDE.md`](DEPLOYMENT_GUIDE.md).
