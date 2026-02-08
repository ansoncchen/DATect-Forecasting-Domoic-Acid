# DATect Forecasting Pipeline — Complete Deep Dive

## What DATect Does

DATect forecasts **domoic acid (DA)** concentrations at 10 Pacific Coast monitoring sites. Domoic acid is a neurotoxin produced by *Pseudo-nitzschia* algae during harmful algal blooms (HABs). When shellfish accumulate DA above safety thresholds, fisheries must close. DATect aims to predict DA levels **7 days in advance** so coastal managers can act proactively.

The system predicts in 4 risk categories:
- **Low**: 0–5 µg/g (safe)
- **Moderate**: 5–20 µg/g (watch)
- **High**: 20–40 µg/g (concern — FDA action level is 20)
- **Extreme**: 40+ µg/g (severe)

---

## Phase 1: Data Ingestion

The pipeline starts with two completely separate data sources that get merged.

### 1A. Raw DA Measurements (`raw_data_forecaster.py → load_raw_da_measurements()`)

This reads the actual toxin measurement CSVs from `data/raw/da-input/` — one file per site. These are irregularly-spaced field samples collected by state agencies. The data has two different CSV schemas:

- **Washington sites** (Kalaloch, Quinault, Copalis, Twin Harbors, Long Beach): Use `CollectDate` + `Domoic Result` columns
- **Oregon sites** (Clatsop Beach, Cannon Beach, Newport, Coos Bay, Gold Beach): Use `Harvest Month/Date/Year` + `Domoic Acid` columns

The function detects the schema automatically, parses dates, filters out negative/invalid values, normalizes site names (e.g., `"twin-harbors"` → `"Twin Harbors"`), and produces a clean DataFrame with columns `[date, site, da_raw]`.

This data is **sparse and irregular** — samples might be days apart during bloom season but weeks apart in winter. This sparsity is a fundamental challenge that drives many design decisions.

### 1B. Processed Environmental Data (`dataset-creation.py`)

This is a separate 30-60 minute process that downloads and merges 21 years of satellite/environmental data into a single Parquet file (`data/processed/final_output.parquet`). It contains weekly-aligned rows with:

- **MODIS Satellite data** (8-day composites): Chlorophyll-a, SST, PAR, Fluorescence Line Height, K490 — each with site-specific bounding boxes
- **Anomaly data** (monthly): Chlorophyll-a anomaly, SST anomaly
- **Climate indices**: PDO (Pacific Decadal Oscillation), ONI (Oceanic Niño Index)
- **BEUTI**: Biologically Effective Upwelling Transport Index (daily → weekly)
- **Columbia River streamflow**: Daily discharge from USGS gauge 14246900

All data is aligned to Monday-based weekly periods with site-specific spatial averages.

---

## Phase 2: Feature Frame Construction

The `_load_feature_frame()` method in `RawForecastEngine` merges these two data sources into a single feature frame. Here's the exact sequence:

### Step 2A: Weekly Aggregation (`aggregate_raw_to_weekly()`)

Raw DA measurements are aggregated to Monday-aligned weekly buckets using **MAX** (not mean). If 3 samples were taken in a week with values [2, 8, 15], the weekly value is 15. This is conservative — it preserves the worst-case reading.

### Step 2B: Merge Raw onto Environmental (`build_raw_feature_frame()`)

The weekly raw DA values are LEFT-joined onto the processed environmental data by `[site, date]`. Most rows will have `da_raw = NaN` because environmental data is weekly but measurements are sparse. Only rows where a measurement happened that week have a non-null `da_raw`.

### Step 2C: Persistence Features

Three features are computed using forward-fill within each site:

1. **`last_observed_da_raw`**: The most recent actual DA measurement (forward-filled). This is the strongest single feature — DA values tend to persist.

2. **`weeks_since_last_raw`**: How many weeks since the last real measurement. When this is large, the persistence information is stale and less reliable.

3. **`weeks_since_last_spike`**: Weeks since DA was last above 20 µg/g. Captures recency of dangerous events.

### Step 2D: Rolling Statistics

When `USE_ROLLING_FEATURES=True`, computes rolling statistics over `shift(1)` of `last_observed_da_raw` (the shift prevents leakage — it uses T-1 and earlier, never the current row):

- **Windows**: 4, 8, 12 weeks
- **Statistics**: mean, std, max
- Creates features like `raw_obs_roll_mean_4`, `raw_obs_roll_std_8`, `raw_obs_roll_max_12`

### Step 2E: Observation-Order Lag Features (`raw_data_processor.py`)

This is one of the most important design decisions. Instead of traditional time-series grid-shift lags (which would be mostly NaN on sparse data), the system uses **observation-order lags**:

- `da_raw_prev_obs_1`: The most recent actual DA measurement before this row
- `da_raw_prev_obs_2`: The second most recent
- `da_raw_prev_obs_3`: The third most recent
- `da_raw_prev_obs_4`: The fourth most recent

Plus recency features:
- `da_raw_prev_obs_2_weeks_ago`: How many weeks ago was the 2nd most recent measurement
- `da_raw_prev_obs_3_weeks_ago`, `da_raw_prev_obs_4_weeks_ago`

And a trend feature:
- `da_raw_prev_obs_diff_1_2`: Difference between most recent and 2nd most recent (captures whether DA is rising or falling)

The algorithm iterates through each site, finds all non-NaN `da_raw` observations strictly before each row's date, and looks backward to find the Nth previous observation. This means lag 1 for a row in January might come from a December measurement 3 weeks earlier, while lag 1 for a row during bloom season might come from just last week.

### Step 2F: Temporal Features (`feature_utils.py → add_temporal_features()`)

Deterministic calendar features that are safe to compute for any date (no data leakage risk):

- `sin_day_of_year`, `cos_day_of_year`: Cyclic encoding of season (captures that day 365 is near day 1)
- `month`, `sin_month`, `cos_month`: Month encoding
- `quarter`: Season quarter
- `sin_week_of_year`, `cos_week_of_year`: Fine-grained seasonal cycle
- `is_bloom_season`: Binary flag for March–October (when *Pseudo-nitzschia* blooms are most likely)
- `days_since_start`: Linear trend (captures long-term changes like warming ocean)

---

## Phase 3: Per-Prediction Data Splitting

For every prediction point (whether realtime or retrospective), the system creates a completely fresh train/test split. **There is no pre-trained model** — a new model is trained from scratch for every single prediction.

### Step 3A: Anchor Date Calculation

For a forecast date (test_date), the **anchor date** = test_date - 7 days. This is the knowledge cutoff — the model can only use data available 7 days before the target date.

### Step 3B: Training Data (`get_site_training_frame()`)

All feature frame rows for this site where `date ≤ anchor_date` AND `da_raw` is not null. This ensures:
- Only past data is used
- Only rows with actual measurements (not interpolated/empty rows) form the training set
- The model learns from real signal, not imputed values

Requires at least `MIN_TRAINING_SAMPLES=10` rows.

### Step 3C: Test Row Construction (`get_site_anchor_row()`)

This is critical for leak-free forecasting. The test row is built from:

- **Environmental features** (SST, chlorophyll, etc.): Taken from the closest processed row **at or before the anchor date** — information that would actually be available at prediction time
- **Calendar features** (sin_day_of_year, month, etc.): Computed for the actual test_date, since these are deterministic
- The row's `date` field is set to `test_date` so temporal features are correct

The key insight: at the time of making a prediction, you'd have satellite data up to ~7 days ago, but you wouldn't have environmental data from the future test date.

### Step 3D: Persistence Feature Recomputation (`recompute_test_row_persistence_features()`)

The persistence features (`last_observed_da_raw`, `weeks_since_last_raw`, `weeks_since_last_spike`) were computed globally during feature frame construction, which means they might leak future information. This function overwrites them using **only the training data**:

- `last_observed_da_raw` = the most recent DA value in training data
- `weeks_since_last_raw` = time from test_date to that last measurement
- `weeks_since_last_spike` = time to last spike event in training data

Rolling features (raw_obs_roll_*) are NOT recomputed because they already use `shift(1)` and don't leak.

### Step 3E: Leakage Verification (`_verify_no_data_leakage()`)

Every single prediction calls this assertion function that checks:
1. No training data has a date after the anchor date
2. The test date is after the anchor date

If either check fails, it raises an `AssertionError` and the prediction is aborted.

---

## Phase 4: Feature Preprocessing (`feature_utils.py → create_transformer()`)

Before model training, features go through a preprocessing pipeline:

### Step 4A: Column Dropping

Several categories of columns are removed:
- `date`, `site` (non-features)
- `da_raw`, `da` (target variable — would be leakage)
- `ZERO_IMPORTANCE_FEATURES` from config (features proven to have near-zero predictive power in leak-free evaluation): `lat`, `lon`, `weeks_since_last_raw`, `is_bloom_season`, `quarter`, `raw_obs_roll_mean_12`, `modis-par`, `raw_obs_roll_mean_8`, `sin_week_of_year`, `cos_month`, `modis-k490`, `cos_week_of_year`, `da_raw_prev_obs_4_weeks_ago`
- Per-site feature subset enforcement (if the site has a `feature_subset` defined, only those features are kept)

### Step 4B: Imputation + Scaling

A sklearn `ColumnTransformer` with a pipeline:
1. **`SimpleImputer(strategy="median")`**: Fills NaN values with column medians (many environmental features have gaps due to cloud cover, sensor outages, etc.)
2. **`MinMaxScaler()`**: Scales all features to [0, 1] range (important for XGBoost and especially Ridge regression)

The transformer is fit on training data only, then applied to transform both training and test data.

---

## Phase 5: Model Training & Prediction

Three models are trained independently on the same preprocessed data:

### 5A: XGBoost Regressor

Built via `build_xgb_regressor()` with config params (n_estimators=400, max_depth=6, learning_rate=0.05, etc.). Per-site overrides from `per_site_models.py` can change every hyperparameter — for example, Kalaloch uses max_depth=2 with only 80 estimators (very conservative), while Long Beach uses max_depth=3 with 250 estimators (more aggressive).

### 5B: Random Forest Regressor

Built via `build_rf_regressor()` with separate params (n_estimators=400, max_depth=12, etc.). Some sites override with conservative RF params (max_depth=6, max_features=0.5) when RF performs poorly at that site.

### 5C: Naive Baseline (`get_last_known_raw_da()`)

Simply returns the most recent DA measurement at or before the anchor date. No ML involved. Surprisingly competitive for sites where DA values are slow-changing.

### 5D: Post-Processing (`_postprocess()`)

Every ML prediction goes through:
1. **Floor at 0**: DA concentrations can't be negative
2. **Quantile clipping**: Prediction capped at the 99th percentile (or site-specific quantile) of training DA values. Prevents wild overestimates.
3. **Hard ceiling**: Some sites have absolute caps (e.g., Kalaloch capped at 80 µg/g)

---

## Phase 6: Ensemble Blending

The three predictions are combined using **per-site weighted averaging**:

```
ensemble_prediction = w_xgb * xgb_pred + w_rf * rf_pred + w_naive * naive_pred
```

The weights are completely customized per site based on which models perform best there:

| Site | XGB | RF | Naive | Character |
|------|-----|-----|-------|-----------|
| **Twin Harbors** | 0.10 | 0.25 | **0.65** | Persistence-dominant — naive is king |
| **Kalaloch** | 0.20 | 0.40 | **0.40** | Persistence-dominant |
| **Copalis** | 0.25 | **0.45** | 0.30 | RF slightly favored |
| **Quinault** | 0.35 | 0.30 | 0.35 | Balanced three-way |
| **Long Beach** | **0.45** | 0.40 | 0.15 | ML-leaning |
| **Clatsop Beach** | **0.50** | 0.45 | 0.05 | ML-dominant |
| **Coos Bay** | 0.10 | **0.85** | 0.05 | RF-dominant |
| **Newport** | 0.25 | **0.65** | 0.10 | RF-leaning |
| **Gold Beach** | 0.40 | **0.57** | 0.03 | RF-leaning |
| **Cannon Beach** | **0.95** | 0.03 | 0.02 | XGB-desperate (all models struggle) |

The three site categories in `per_site_models.py` are:
- **Persistence-dominant** (Twin Harbors, Kalaloch, Copalis, Quinault): DA changes slowly, naive baseline is strong, ML is constrained
- **ML-leaning** (Long Beach, Clatsop Beach, Coos Bay): ML models outperform naive
- **Struggle sites** (Cannon Beach, Gold Beach, Newport): All models have negative R-squared — the data is fundamentally hard to predict. These sites get aggressive regularization and conservative clipping.

---

## Phase 7: Per-Anchor Tuning (Retrospective Only)

During retrospective evaluation (not realtime), the engine performs **per-anchor XGBoost hyperparameter tuning**:

1. Take a calibration sample: 30% of training data (capped at 20 rows)
2. For each parameter set in the site's `param_grid` (1-3 options):
   - Train XGBoost with those params
   - Evaluate on calibration rows (each calibration row gets its own nested train/test split)
   - Track R-squared score
3. Use the param set with the best R-squared

This is a lightweight form of cross-validation that adapts the model complexity to how much data is available at each anchor point. Early anchors (less data) might benefit from simpler models; later anchors (more data) can handle more complexity.

---

## Phase 8: Classification

Two classification approaches are available:

### 8A: Threshold Classification (`ClassificationAdapter.threshold_classify()`)

Simply bins the continuous DA prediction using the standard thresholds:
- < 5 → Low (0)
- 5–20 → Moderate (1)
- 20–40 → High (2)
- 40+ → Extreme (3)

This is the primary method and is always computed.

### 8B: Dedicated ML Classifier (`ClassificationAdapter.train_dedicated_classifier()`)

Trains a separate XGBoost classifier on the 4-category labels with balanced class weighting. Returns per-class probabilities (e.g., [0.72, 0.20, 0.06, 0.02]) for the UI probability bar chart. Uses `compute_class_weight("balanced")` to handle the severe class imbalance (most measurements are Low).

---

## Phase 9: Confidence Intervals

Two methods for uncertainty estimation:

### 9A: XGBoost Quantile Regression (preferred)

Uses XGBoost's `objective="reg:quantile"` with `quantile_alpha` of 0.05, 0.50, and 0.95 to directly estimate the 5th, 50th, and 95th percentile predictions. Three separate XGBoost models are trained — one per quantile.

### 9B: Bootstrap (fallback)

When quantile objectives aren't available, uses bootstrap resampling:
1. Resample training data with replacement (20 iterations by default)
2. Train the full model (or ensemble) on each resample
3. Collect predictions
4. Take percentiles of the prediction distribution

For ensemble bootstrap, each iteration trains both XGBoost and RF on the resampled data, then blends with the fixed naive prediction.

---

## Phase 10: Retrospective Evaluation (`run_retrospective_evaluation()`)

This is the validation loop that evaluates the pipeline on historical data:

1. **Filter candidate test points**: Raw measurements after `MIN_TEST_DATE` (2003) where the site has enough history (>= 33% of total measurements before the anchor)

2. **Per-site sampling**: ~20% of each site's total raw measurements are selected as test points (stratified by site to avoid imbalance)

3. **Parallel execution**: Using `joblib.Parallel`, each test measurement gets its own complete train/predict cycle (per-anchor tuning → training → prediction → classification → quantile intervals)

4. **Result aggregation**: Individual results are collected into a DataFrame, ensemble predictions are computed, and categories are derived. Duplicates (same date+site) are deduplicated.

5. **Metrics**: R-squared, MAE, and Spike F1 are computed and logged.

---

## Phase 11: Caching & Deployment

### `precompute_cache.py` (runs on Hyak cluster)

Pre-computes all 8 task/model combinations:
- regression x {ensemble, xgboost, rf, naive, linear}
- classification x {ensemble, naive, logistic}

Plus spectral analysis and correlation heatmaps. Saves everything as JSON + Parquet files in `cache/`. This takes hours on the cluster.

### The API (`backend/api.py`)

Checks for cached results first. If cache exists, serves pre-computed predictions instantly. If not, falls back to running the engine in realtime (which is slow).

---

## The Full Data Flow for One Prediction

```
1.  Raw CSV files → load_raw_da_measurements() → DataFrame[date, site, da_raw]
2.  Processed Parquet → read_parquet() → DataFrame[date, site, sst, chla, ...]
3.  aggregate_raw_to_weekly() → Weekly DA values (MAX aggregation)
4.  build_raw_feature_frame() → Merge raw DA onto env data + persistence + rolling + lag features
5.  add_temporal_features() → Calendar encodings (sin/cos day, month, season)
6.  get_site_training_frame() → Filter to site, date <= anchor, da_raw not null
7.  get_site_anchor_row() → Build test row with env features from anchor date
8.  recompute_test_row_persistence_features() → Overwrite leaky persistence features
9.  _verify_no_data_leakage() → Assert no temporal leaks
10. create_transformer() → Drop non-features → Median impute → MinMax scale
11. Per-anchor XGB tuning (if retrospective) → Find best hyperparams
12. Train XGBoost, Random Forest, get Naive → Three predictions
13. _postprocess() → Floor at 0, clip to quantile, apply hard ceiling
14. Weighted ensemble blend → w_xgb * XGB + w_rf * RF + w_naive * Naive
15. threshold_classify() → Map to risk category
16. Quantile regression or bootstrap → Confidence intervals
17. Return result dict to API
```

Every single prediction (steps 6-17) is independent — a fresh model trained from scratch with zero knowledge of the future. This is by far the most computationally expensive aspect: 500 test points x 10 sites x 3 models x per-anchor tuning = tens of thousands of model trainings per full retrospective run.
