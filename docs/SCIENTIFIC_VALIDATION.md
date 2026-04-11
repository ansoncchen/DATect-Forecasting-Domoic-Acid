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

### 4. Fixed Regulatory Thresholds

For threshold-based classification, DATect uses fixed DA category cut points from
`config.DA_CATEGORY_BINS` (`0, 5, 20, 40, inf`). Because these bins are
regulatory constants rather than values estimated from the full dataset, they do
not introduce target leakage.

## Per-Prediction Validation

The system validates temporal integrity on **every single prediction** via `_verify_no_data_leakage()` (defined in `raw_forecast_engine.py`), which raises an `AssertionError` if any temporal violation is detected:

```python
def _verify_no_data_leakage(train_data, test_date, anchor_date):
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

- Fixed random seeds for development and held-out evaluation (`RANDOM_SEED = 42`, paper hold-out seed = 123)
- Versioned dependencies
- Deterministic preprocessing and model configuration

### Conservative Evaluation

- Performance metrics reflect operational constraints
- No optimistic assumptions
- Realistic data availability simulation

### Statistical Rigor

- Chronological retrospective evaluation
- Quantile/bootstrap uncertainty summaries (configurable)
- Proper hypothesis testing where applicable

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
N_BOOTSTRAP_ITERATIONS = 100
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

## Stability & Sensitivity Validation

The stability study (`paper_stability_study.py`) validates that all per-site tuning decisions are robust and not artifacts of the seed=42 development set.

### Phase 1A: Multi-Seed Noise Floor (5 seeds × per-site ON/OFF)

Performance varies substantially by region:

| Region | Sites | Mean R² | Std | Interpretation |
|--------|-------|---------|-----|----------------|
| Washington | 5 | 0.61 | 0.05 | Stable, meaningful signal |
| Oregon | 5 | −0.03 | 0.25 | Near-random; data scarcity limits all methods |
| Pooled (per-site ON) | 10 | 0.265 | 0.10 | High variance driven by OR sites |
| Pooled (per-site OFF) | 10 | 0.138 | 0.15 | Per-site config adds +0.13 R² pooled |

### Phase 1B: Perturbation Sensitivity (13 experiments at seed=42)

| Perturbation Group | Max |ΔR²| | Finding |
|---|---|---|
| RF hyperparameters (4 configs) | < 0.001 | RF is genuinely robust to hyperparameters |
| Feature subsets (all vs minimal) | 0.006 | Persistence signal dominates; subsets barely matter pooled |
| Prediction clipping (relaxed/off) | 0.030 | Per-site quantiles (0.95–0.98) are well-calibrated |
| Model selection (swap RF↔XGB) | 0.029 | Current winner-take-all assignments are near-optimal |
| Monotonic constraints (off) | < 0.001 | Negligible effect |
| Per-site config (disabled entirely) | 0.103 | Per-site customization is the dominant design decision |

### Go/No-Go Decision

No perturbation exceeds the noise floor (2× seed std = 0.204). Phase 2 systematic tuning (Optuna, grid search) is not warranted. Current configuration is validated.

### Stability Study Env Vars

The stability study uses env vars to propagate perturbations through joblib's loky multiprocessing:

| Variable | Read by | Purpose |
|---|---|---|
| `DATECT_RF_PARAMS_JSON` | `config.py` | Override RF hyperparameters as JSON dict |
| `DATECT_FEATURE_SUBSET_MODE` | `per_site_models.py` | `"all"` or `"minimal"` feature override |
| `DATECT_CLIP_Q_OVERRIDE` | `config.py` + `per_site_models.py` | `"none"` or float (e.g. `"0.99"`) |
| `DATECT_USE_PER_SITE_MODELS` | `config.py` | `"false"` to disable per-site config |
| `DATECT_USE_MONOTONIC_CONSTRAINTS` | `config.py` | `"false"` to disable monotonic constraints |

## Pre-Publication Checklist

Before using results for publication:

- [x] `verify_no_data_leakage()` passes for all predictions (automatic)
- [x] Retrospective evaluation completed across all sites
- [x] Performance metrics documented (R², MAE, Spike F1)
- [x] Feature importance scientifically reasonable
- [x] Confidence intervals properly calibrated
- [x] Reproducibility verified (development and held-out seeds documented)
- [x] Stability study validates per-site tuning robustness across seeds
- [x] Perturbation analysis confirms RF params, features, clipping are insensitive

## Why Trust DATect Results

1. **Strong structural safeguards against leakage**: `_verify_no_data_leakage()` is called for every prediction, and the train/test split, lag features, delays, and persistence recomputation are built around the anchor date
2. **Operational realism**: Data availability delays match real-world constraints
3. **Conservative evaluation**: No optimistic metrics
4. **Structural enforcement**: Safeguards built into pipeline code, not just checked after
5. **Transparent implementation**: Safeguards are documented, auditable, and paired with held-out retrospective evaluation

## References

1. Trainer, V.L., et al. (2007). "Pseudo-nitzschia physiological ecology, phylogeny, toxicity, monitoring and impacts on ecosystem health." *Harmful Algae*.

2. Anderson, C.R., et al. (2021). "Predicting harmful algal blooms: A machine learning approach for early warning systems." *Marine Environmental Research*.

3. Wells, M.L., et al. (2020). "Toxic and harmful algal blooms in temperate coastal waters." *Oceanography*.
