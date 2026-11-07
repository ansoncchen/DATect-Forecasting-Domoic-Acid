# Visualizations Guide

Every chart the dashboard shows, where it is produced, and how to read it. Plotly figures for the Data explorer page are built in `backend/visualizations.py` (and the site map in `backend/api.py`); the forecast and track-record charts are built in the frontend (`frontend/src/pages/`).

Risk categories used by every chart: Low `[0, 5)`, Moderate `[5, 20)`, High `[20, 40)`, Extreme `>= 40` µg/g; a value on a boundary belongs to the higher category.

## Data explorer (`frontend/src/pages/DataExplorer.jsx`)

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

**Site map** (`get_site_map` in `backend/api.py`) — OpenStreetMap scatter of the 10 sites from `config.SITES`, all in one colour; hovering a site gives the last reading in its raw DA file. Below-detection readings are shown as LOD/2. The API still colours the markers by that reading and titles the figure "Risk Levels"; `DataExplorer.jsx` overrides both until the figure itself changes.

## Forecast page (`frontend/src/pages/Forecast.jsx`)

One screen: pick a site and a forecast week, and the page shows that week's forecast. Thresholds, alert levels, category bins and the range level come from `GET /api/config`; the page computes no metric itself.

- **Forecast card** — the predicted DA and its range, from `POST /api/forecast/enhanced` (the `ensemble` regression forecast, with the conformal error band; see [`ERROR_BAND.md`](ERROR_BAND.md)). A forecast with no band shows its value and why there is no range; a failed one shows its error.
- **Spike risk card** — the hybrid spike alert level, mapped from the API's alert field, with the classifier's score to two decimals. The score is not a calibrated chance ([`CORRECTED_NUMBERS.md`](CORRECTED_NUMBERS.md)), so the card also says what weeks at that level were followed by. If the last measurement before the forecast was already at or above 20 µg/g, it says so instead of a level and shows no score. If the classifier is missing for a week, the card keeps the level the forecast sets and says the classifier is unavailable.
- **Latest measurement card** — the newest measurement before the forecast and its age; a stale or missing one is flagged.
- **Measurements and forecasts** — a chart of the site's measurements (`GET /api/sites/{site}/history`) with the forecast and its range at the chosen week and the 20 µg/g line. For a past week it stops at that week's anchor, and the "Show what happened" toggle adds the measurements that followed. Measurements outside the range are hollow. The line and range break across gaps of more than three weeks; a forecast with no neighbour on either side is a short bar with a small marker.
- **Metrics tiles** — selectable tiles in the order we suggest reading them (alert track record, range reliability, typical error, category chances from the range, fit, spike detection by the forecast, model parts, XGBoost's feature importances, range width, data and model). Reliability and track-record figures come from `GET /api/metrics/summary`, which reads `results/chronological` and `results/multi_seed`. Feature importances are the fitted XGBoost's only; where the XGBoost weight is 0 the tile says the forecast comes from the Random Forest and lists none.
- **All sites** — the same week's forecast for every site from `GET /api/forecast/all`, north to south, under a heading that names the week its rows belong to; a row selects its site. A week that is not precomputed is computed on demand and the page says so while it waits.

## Track record page (`frontend/src/pages/TrackRecord.jsx`)

Built from `GET /api/track-record` for the chosen window (holdout 2022–2023, validation 2019–2021) and optionally one site. Every panel scores real measurements only, each forecast made a week ahead from data up to its anchor.

- **Summary** — N, R², log R², MAE, how often the range held, jumps caught and alarms followed by a jump, for all sites or the selected one. "too few" replaces a figure the API does not report for lack of rows.
- **Forecast against measurement** — for all sites, a predicted-versus-actual scatter with each point's range and the 20 µg/g lines; for one site, its measurements and week-ahead forecasts over time (`GET /api/sites/{site}/history`, cut to the window).
- **Against simpler models** — the ensemble beside naive persistence and the linear model on the same measurements.
- **By site** — one row per site for the window; a row selects the site. Per-site figures rest on few measurements at some sites; read them as indicative.

The filtered rows can be downloaded as CSV; Plotly charts can be saved as PNG from their toolbar. For current headline values see [`CORRECTED_NUMBERS.md`](CORRECTED_NUMBERS.md).

## Reading the charts

- Check the range and how often it held at the site before trusting a point forecast. Where a range cannot be built the page says "No range" and why; that does not mean the forecast is certain.
- Correlation, Sobol and permutation panels are exploratory statistics on gap-filled weekly data, not evidence of causation or of what the forecasting models rely on.
- Per-site metrics rest on few holdout measurements at some sites; read them as indicative.
