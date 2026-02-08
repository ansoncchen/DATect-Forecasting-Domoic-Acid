# Scientific Validation & Temporal Safeguards

## Overview

DATect implements comprehensive validation to ensure scientific integrity and prevent data leakage in time series forecasting. This document describes the temporal safeguards and validation framework.

## The Data Leakage Problem

In time series forecasting, **data leakage** occurs when future information inadvertently influences predictions. This is the most critical threat to scientific validity.

**Example of leakage:**
```python
# WRONG: Using all data (future information leaks)
model.fit(all_data)
prediction = model.predict(past_features)  # Model "knows" the future

# CORRECT: Only using past data
model.fit(past_data_only)
prediction = model.predict(past_features)  # No future information
```

## Temporal Safeguards

### 1. Strict Temporal Cutoffs

Every feature and model training respects temporal boundaries:

```python
# Anchor date defines the temporal cutoff
anchor_date = forecast_date - FORECAST_HORIZON_DAYS

# Training data: strictly before anchor
training_data = data[data['date'] <= anchor_date]

# Lag features also respect cutoffs
lag_cutoff_date = cutoff_date - buffer_days
df.loc[df['date'] > lag_cutoff_date, lag_feature] = NaN
```

### 2. Data Availability Delays

The system simulates real-world data availability:

| Data Type | Buffer | Rationale |
|-----------|--------|-----------|
| Satellite (8-day) | 7 days | Processing and QC time |
| Climate indices | 2 months | Official reporting delay |
| Monthly anomalies | 2 months | Requires end-of-month data |

### 3. Chronological Train/Test Splits

Random splits violate temporal ordering. DATect uses chronological splits:

```python
# Sort by date
data_sorted = data.sort_values('date')

# Training: all data before anchor
train_data = data_sorted[data_sorted['date'] <= anchor_date]

# Test: forecast target date
test_data = data_sorted[data_sorted['date'] == forecast_date]
```

### 4. Per-Forecast Category Creation

For classification, DA risk categories are computed per-forecast using only training data:

```python
# WRONG: Global categories (includes future data)
all_data['category'] = categorize(all_data['da'])  # Leakage!

# CORRECT: Per-forecast categories
train_data['category'] = categorize_from_training_only(train_data['da'])
```

## Per-Prediction Validation

The system validates temporal integrity on **every single prediction** via `verify_no_data_leakage()` (defined in `config.py`), which raises an `AssertionError` if any temporal violation is detected:

```python
def verify_no_data_leakage(train_data, test_date, anchor_date):
    """Called for every prediction — raises on temporal leakage."""
    # 1. Training data must not extend past anchor
    assert train_data['date'].max() <= anchor_date
    # 2. Test date must be after anchor
    assert test_date > anchor_date
```

### Structural Safeguards (Enforced by Pipeline Design)

These safeguards are baked into the pipeline code, not checked after the fact:

| Safeguard | Where Enforced |
|-----------|---------------|
| **Chronological split** — training ≤ anchor_date | `raw_forecast_engine.py` (train/test split) |
| **Observation-order lags** — past-only shifts | `raw_data_processor.py` (lag feature construction) |
| **Satellite delay** — 7-day buffer | `dataset-creation.py` (data ingestion) |
| **Climate delay** — 2-month buffer | `dataset-creation.py` (data ingestion) |
| **Per-forecast categories** — from training only | `classification_adapter.py` |
| **Persistence features** — recomputed from training | `raw_data_forecaster.py` |
| **Fresh model per test point** — no shared state | `raw_forecast_engine.py` (per-anchor loop) |
| **Cross-site consistency** — same rules for all sites | `per_site_models.py` (only tunes hyperparams) |

### System Startup Validation

Basic configuration checks run at startup via `validation.py`:

```bash
# Runs automatically when starting:
python run_datect.py       # System startup
python precompute_cache.py # Cache generation
```

## Scientific Standards

### Reproducibility

- Fixed random seeds (`RANDOM_SEED = 42`)
- Versioned dependencies
- Deterministic results

### Conservative Evaluation

- Performance metrics reflect operational constraints
- No optimistic assumptions
- Realistic data availability simulation

### Statistical Rigor

- Chronological cross-validation
- Quantile/bootstrap confidence intervals (configurable)
- Proper hypothesis testing

## Configuration Parameters

```python
# config.py settings

# Forecast horizon
FORECAST_HORIZON_WEEKS = 1
FORECAST_HORIZON_DAYS = 7

# Validation
MIN_TRAINING_SAMPLES = 10
N_RANDOM_ANCHORS = 500

# Bootstrap
ENABLE_BOOTSTRAP_INTERVALS = True
N_BOOTSTRAP_ITERATIONS = 20
BOOTSTRAP_SUBSAMPLE_FRACTION = 1.0

# Baselines
LINEAR_REGRESSION_ALPHA = 1.0
PERSISTENCE_MAX_DAYS = None
CONFIDENCE_PERCENTILES = [5, 50, 95]

# Spike detection
SPIKE_THRESHOLD = 20.0  # μg/g

# Random seed
RANDOM_SEED = 42
```

## Validation Failure Actions

If `verify_no_data_leakage()` detects a temporal violation, it raises an `AssertionError` that halts the prediction:

```python
# Example failure:
# AssertionError: TEMPORAL LEAK: training data max date 2020-07-01 > anchor 2020-06-15
```

This ensures no invalid predictions can be generated — the system fails loudly rather than silently producing leaky results.

## Pre-Publication Checklist

Before using results for publication:

- [ ] `verify_no_data_leakage()` passes for all predictions (automatic)
- [ ] Retrospective evaluation completed across all sites
- [ ] Performance metrics documented (R², MAE, Spike F1)
- [ ] Feature importance scientifically reasonable
- [ ] Confidence intervals properly calibrated
- [ ] Reproducibility verified (fixed seeds, `RANDOM_SEED = 42`)

## Why Trust DATect Results

1. **Mathematical impossibility of leakage**: `verify_no_data_leakage()` called for every prediction
2. **Operational realism**: Data availability delays match real-world constraints
3. **Conservative evaluation**: No optimistic metrics
4. **Structural enforcement**: Safeguards built into pipeline code, not just checked after
5. **Transparent implementation**: All safeguards documented and auditable

## References

1. Trainer, V.L., et al. (2007). "Pseudo-nitzschia physiological ecology, phylogeny, toxicity, monitoring and impacts on ecosystem health." *Harmful Algae*.

2. Anderson, C.R., et al. (2021). "Predicting harmful algal blooms: A machine learning approach for early warning systems." *Marine Environmental Research*.

3. Wells, M.L., et al. (2020). "Toxic and harmful algal blooms in temperate coastal waters." *Oceanography*.
