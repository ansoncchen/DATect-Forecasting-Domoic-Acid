# Visualizations Guide

Every chart the dashboard shows, where it is produced, and how to read it. Plotly figures for the Historical page are built in `backend/visualizations.py` (and the site map in `backend/api.py`); dashboard result charts are built in the frontend (`frontend/src/pages/Dashboard.jsx`, `frontend/src/utils/retrospective.js`).

Risk categories used by every chart: Low `[0, 5)`, Moderate `[5, 20)`, High `[20, 40)`, Extreme `>= 40` µg/g; a value on a boundary belongs to the higher category.

## Historical page (`frontend/src/pages/Historical.jsx`)

| Chart | Endpoint | Scope |
|---|---|---|
| Correlation heatmap | `/api/visualizations/correlation/{site}`, `/correlation/all` | single site or all |
| Sensitivity analysis | `/api/visualizations/sensitivity/{site}`, `/sensitivity/all` | single site or all |
| DA vs Pseudo-nitzschia | `/api/visualizations/comparison/{site}` | single site only |
| Waterfall plot | `/api/visualizations/waterfall` | all sites |
| Spectral analysis | `/api/visualizations/spectral/{site}`, `/spectral/all` | single site or all |
| Site map | `/api/visualizations/map` | all sites |

All except the site map read the weekly processed dataset, so their `da` and `pn` are the **gap-filled** weekly series, not raw measurements.

**Correlation heatmap** (`generate_correlation_heatmap`) — Pearson correlations among all numeric columns except `lat`/`lon`, after median-imputing missing feature values. Blue is positive, red negative, white near zero. Per-site heatmaps are served from `cache/visualizations/` when the pre-computed cache is enabled.

**Sensitivity analysis** (`generate_sensitivity_analysis`) — up to three bar charts against `da`:
1. absolute Pearson correlation of each variable with `da`;
2. Sobol first-order indices (top 10), computed with SALib on a linear-regression surrogate (only with at least 50 rows);
3. permutation importance of that linear model (30 repeats).
These describe linear association in the weekly table, not the forecasting models' behavior.

**DA vs Pseudo-nitzschia** (`generate_time_series_comparison`) — weekly DA (capped at 80 µg/g) and PN, each min-max scaled to 0–1 on one axis. Shows MODIS chlorophyll instead when a site has no PN data. Useful for seeing whether PN peaks lead DA peaks.

**Waterfall plot** (`generate_waterfall_plot`) — one line per site, stacked by latitude (north at top). Each baseline is DA = 0; height above it is proportional to DA. Grey reference bars mark 20, 50 and 100 µg/g. Shows how blooms line up along the coast in time.

**Spectral analysis** (`generate_spectral_analysis`) — power spectral density (Welch), periodogram, spectrogram, coherence, and a summary panel for the weekly DA series. When `cache/retrospective/regression_xgboost.*` and `regression_rf.parquet` exist, XGBoost and Random Forest retrospective predictions are overlaid and coherence measures how well each tracks the actual series by frequency. Served from `cache/spectral/` when available; computing it on the server is slow.

**Site map** (`get_site_map` in `backend/api.py`) — OpenStreetMap scatter of the 10 sites from `config.SITES`, colored by the last reading in each raw DA file (green Low, yellow Moderate, orange High, red Extreme, grey no data). Below-detection readings are shown as LOD/2.

## Dashboard: single forecast

Built from `POST /api/forecast/enhanced`, which runs a regression and a classification forecast for the chosen site, date and model.

- **Summary cards** — predicted DA, predicted category, and (when the spike classifier ran) spike probability with the hybrid spike alert.
- **Ensemble model breakdown** — XGBoost and Random Forest predictions with their site weights, the blended ensemble prediction, and naive persistence as a baseline.
- **Gradient uncertainty plot** (`generate_gradient_uncertainty_plot`) — shading from q05 to q95 (darkest at q50), with markers for the ensemble, XGBoost, Random Forest and, when supplied, the actual value. Intervals come from XGBoost quantile regression for `ensemble`/`xgboost` and from bootstrap for other models (see [`PIPELINE_DEEP_DIVE.md`](PIPELINE_DEEP_DIVE.md#9-single-forecasts-dashboard-and-apiforecast)). If the quantiles coincide (e.g. RF-only sites), a ±1 µg/g band is drawn for display only.
- **Category probability distribution** — bar chart of the four class probabilities from the classification forecast; the predicted category is highlighted.
- **Top feature importance** — `feature_importances_` of the fitted XGBoost model. At sites whose ensemble weight on XGBoost is 0 this does not describe the model that produced the forecast.

## Dashboard: retrospective results

Built from `POST /api/retrospective` (pre-computed cache in production) plus headline numbers from `GET /api/metrics/summary`, which reads `results/chronological` and `results/multi_seed`.

- **Performance summary** — R², MAE and spike recall / precision / F2 for the filtered results (regression), or accuracy and per-class recall / precision / F1 (classification); headline holdout metrics and per-site quality tiers from `/api/metrics/summary`.
- **Actual vs predicted time series** — per site over time, with the 20 µg/g regulatory line and markers for raised spike alerts. In classification mode, actual and predicted categories over time.
- **Scatter plot** — predicted vs actual, colored by time window (pre-2019 context, 2019–2021 validation, 2022–2023 holdout), with spike-threshold lines splitting caught spikes, missed spikes and false alarms. In classification mode, actual vs predicted category.
- **Per-site small multiples** — one predicted-vs-actual scatter per site (All Sites view).

Filtered results can be exported as CSV; charts can be saved as PNG from the Plotly toolbar. For current headline values see [`CORRECTED_NUMBERS.md`](CORRECTED_NUMBERS.md).

## Reading the charts

- Check the interval width before trusting a point forecast; a zero-width band means the interval method does not apply, not that the forecast is certain.
- Correlation, Sobol and permutation panels are exploratory statistics on gap-filled weekly data, not evidence of causation or of what the forecasting models rely on.
- Per-site retrospective metrics rest on few holdout measurements at some sites; read them as indicative.
