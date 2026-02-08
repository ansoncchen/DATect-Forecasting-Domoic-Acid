# Forecast Pipeline Documentation

## Overview

This document describes the DATect forecasting pipeline from raw data ingestion to prediction generation. The pipeline enforces strict temporal safeguards to prevent data leakage.

## Pipeline Architecture

```
Raw Data Sources → Data Ingestion → Feature Engineering → Model Training → Prediction
       ↓                 ↓                 ↓                  ↓              ↓
   MODIS Satellite   dataset-creation.py  data_processor.py  model_factory.py  Results
   Climate Indices        ↓                    ↓                  ↓
Streamflow        final_output.parquet  Temporal validation  XGBoost/Ridge
   DA Measurements
```

## Stage 1: Raw Data Sources

### Satellite Data (MODIS-Aqua)

| Variable | Description | Temporal Resolution |
|----------|-------------|---------------------|
| Chlorophyll-a | Phytoplankton biomass | 8-day composite |
| SST | Sea surface temperature | 8-day composite |
| PAR | Photosynthetically available radiation | 8-day composite |
| Fluorescence | Phytoplankton stress indicator | 8-day composite |
| K490 | Diffuse attenuation coefficient | 8-day composite |
| CHLA anomaly | Chlorophyll deviation from climatology | Monthly |
| SST anomaly | Temperature deviation from climatology | Monthly |

**Temporal Buffer**: 7-day processing delay enforced to match operational constraints.

### Climate Indices

| Index | Description |
|-------|-------------|
| PDO | Pacific Decadal Oscillation |
| ONI | Oceanic Niño Index (ENSO) |
| BEUTI | Biologically Effective Upwelling Transport Index |

**Temporal Buffer**: 2-month reporting delay enforced for all climate indices and monthly anomalies.

### Other Data

- **Streamflow**: USGS Columbia River discharge (daily)
- **Domoic Acid**: State monitoring measurements (μg/g)
- **Pseudo-nitzschia**: Cell counts of toxin-producing diatoms

## Stage 2: Data Ingestion (`dataset-creation.py`)

The data ingestion pipeline:

1. **Downloads satellite data** monthly from NOAA ERDDAP
2. **Fetches climate indices** with 2-month temporal buffer
3. **Processes DA/PN measurements** with biological decay interpolation
4. **Aggregates to weekly resolution** using ISO weeks
5. **Applies temporal safeguards** (7-day satellite, 2-month climate delays)
6. **Outputs**: `data/processed/final_output.parquet`

### Biological Decay Interpolation

Short gaps in DA/PN data are filled using exponential decay:

```python
DA_MAX_GAP_WEEKS = 2   # Max gap to interpolate
DA_DECAY_RATE = 0.2    # Per-week decay rate
PN_MAX_GAP_WEEKS = 4
PN_DECAY_RATE = 0.3
```

Longer gaps are filled with zeros (assumes non-detection).

## Stage 3: Feature Engineering (`forecasting/data_processor.py`)

### Temporal Features

```python
# Cyclical encoding for seasonality
sin_day_of_year = sin(2π × day_of_year / 365)
cos_day_of_year = cos(2π × day_of_year / 365)
```

### Lag Features (Optional)

When enabled (`USE_LAG_FEATURES = True`):
- **Lag 1**: Previous week's value
- **Lag 3**: 3-week lag (captures bloom cycles)

All lag features respect temporal cutoffs to prevent leakage.

### Feature Processing

```python
# Numeric pipeline
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
```

## Stage 4: Temporal Validation (`forecasting/validation.py`)

### Critical Checks

1. **Chronological ordering**: Training data ≤ anchor_date
2. **Temporal buffer**: Minimum gap between train/test
3. **Future data quarantine**: No post-anchor data in features
4. **Per-forecast categories**: DA categories from training data only

### Anchor Date System

```python
anchor_date = forecast_date - FORECAST_HORIZON_DAYS  # Default: 7 days
training_data = data[data['date'] <= anchor_date]
```

## Stage 5: Model Training (`forecasting/model_factory.py`)

### XGBoost (Primary)

```python
# Regression parameters
XGB_REGRESSION_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}

# Classification parameters
XGB_CLASSIFICATION_PARAMS = {
    "n_estimators": 500,
    "max_depth": 7,
    "learning_rate": 0.03,
    ...
}
```

### Ridge Models (Alternative)

- **Regression**: Ridge Regression
- **Classification**: Logistic Regression
- Uses the full feature set as a linear competitor to the nonlinear models.

### Naive Baseline

- Uses the most recent raw DA measurement at or before the anchor date.
- Optional max lookback can be set via `PERSISTENCE_MAX_DAYS`.

## Stage 6: Prediction Generation (`forecasting/forecast_engine.py`)

### Single Forecast Workflow

```python
def generate_single_forecast(site, date, task, model_type):
    # 1. Calculate anchor date
    anchor_date = forecast_date - FORECAST_HORIZON_DAYS
    
    # 2. Create lag features with temporal cutoff
    features = create_lag_features_safe(data, anchor_date)
    
    # 3. Split train/test chronologically
    train_data = data[data['date'] <= anchor_date]
    
    # 4. Create DA categories from training data only
    categories = create_da_categories_safe(train_data['da'])
    
    # 5. Fit transformer on training data only
    transformer.fit(X_train)
    
    # 6. Train model and predict
    model.fit(X_train_processed, y_train)
    prediction = model.predict(X_forecast)
    
    # 7. Generate quantile/bootstrap confidence intervals (configurable)
    bootstrap_quantiles = generate_confidence_intervals(...)
    
    return prediction, bootstrap_quantiles
```

### Retrospective Evaluation

For validation, the system runs hundreds of forecasts:

```python
def run_retrospective_evaluation(n_anchors=500, min_test_date="2008-01-01"):
    # Generate random anchor points per site
    # Run leak-free forecast for each anchor
    # Compare predictions to actual measurements
    # Aggregate performance metrics
```

## DA Risk Categories

| Category | Range (μg/g) | Label |
|----------|--------------|-------|
| Low | 0-5 | 0 |
| Moderate | 5-20 | 1 |
| High | 20-40 | 2 |
| Extreme | >40 | 3 |

Categories are created per-forecast from training data only to prevent target leakage.

## Configuration (`config.py`)

Key parameters:

```python
FORECAST_HORIZON_WEEKS = 1          # Weeks ahead to forecast
FORECAST_MODEL = "ensemble"         # Primary model ("ensemble", "naive", "linear")
FORECAST_TASK = "regression"        # or "classification"
N_RANDOM_ANCHORS = 500              # Retrospective evaluation points
N_BOOTSTRAP_ITERATIONS = 20         # Confidence interval samples
USE_LAG_FEATURES = False            # Lag features toggle
USE_ROLLING_FEATURES = False        # Rolling statistics toggle
```

## Temporal Safeguards Summary

| Safeguard | Implementation |
|-----------|----------------|
| Chronological split | Training ≤ anchor_date |
| Temporal buffer | 1-day minimum between train/test |
| Satellite delay | 7-day processing buffer |
| Climate delay | 2-month reporting buffer |
| Lag feature cutoffs | No future data in lags |
| Per-forecast categories | Categories from training only |
| Cross-site consistency | Same rules for all 10 sites |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/forecast` | POST | Single forecast |
| `/api/forecast/enhanced` | POST | Regression + classification |
| `/api/retrospective` | POST | Run retrospective evaluation |
| `/api/historical/{site}` | GET | Historical DA data |
| `/api/visualizations/*` | GET | Various analysis plots |
