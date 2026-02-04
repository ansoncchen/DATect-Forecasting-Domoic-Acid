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

## Temporal Integrity Test Suite

The system validates temporal integrity through checks built into `DataProcessor`:

### Test 1: Chronological Split Validation

Ensures training data always precedes test data:
- Zero tolerance for future data in training
- Validates every retrospective forecast

### Test 2: Temporal Buffer Enforcement

Validates minimum gap between train/test:
- Default: 1-day buffer
- Prevents same-day information leakage

### Test 3: Future Information Quarantine

Verifies no post-prediction data in features:
- Checks all calculated features
- Validates lag features respect cutoffs

### Test 4: Per-Forecast Category Creation

Confirms categories use training data only:
- No global category boundaries
- Independent categorization per forecast

### Test 5: Satellite Delay Simulation

Enforces realistic satellite data delays:
- 7-day buffer for 8-day composites
- Matches NASA/NOAA operational schedules

### Test 6: Climate Data Lag Validation

Ensures climate index delays:
- 2-month reporting delay
- Applies to PDO, ONI, BEUTI, and monthly anomalies

### Test 7: Cross-Site Consistency

Verifies uniform rules across all sites:
- All 10 monitoring sites follow same constraints
- No site-specific exceptions

## Running Validation

```bash
# Temporal validation runs automatically during:
python run_datect.py       # System startup
python precompute_cache.py # Cache generation
```

**Expected output:**
```
Running Temporal Integrity Validation...
Test 1: Chronological Split - PASSED
Test 2: Temporal Buffer - PASSED
Test 3: Future Info Quarantine - PASSED
Test 4: Per-Forecast Categories - PASSED
Test 5: Satellite Delays - PASSED
Test 6: Climate Delays - PASSED
Test 7: Cross-Site Consistency - PASSED

All temporal integrity tests passed (7/7)
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
- Bootstrap confidence intervals
- Proper hypothesis testing

## Configuration Parameters

```python
# config.py settings

# Forecast horizon
FORECAST_HORIZON_WEEKS = 1
FORECAST_HORIZON_DAYS = 7

# Validation
MIN_TRAINING_SAMPLES = 3
N_RANDOM_ANCHORS = 500

# Bootstrap
N_BOOTSTRAP_ITERATIONS = 20
CONFIDENCE_PERCENTILES = [5, 50, 95]

# Spike detection
SPIKE_THRESHOLD = 20.0  # μg/g

# Random seed
RANDOM_SEED = 42
```

## Validation Failure Actions

If any temporal test fails, the system refuses to start:

```python
if temporal_validation_failed:
    print("CRITICAL: Temporal integrity violation detected")
    print("System is NOT scientifically valid")
    sys.exit(1)  # Prevent startup
```

This ensures invalid results cannot be generated.

## Pre-Publication Checklist

Before using results for publication:

- [ ] All 7 temporal integrity tests pass
- [ ] Retrospective evaluation completed
- [ ] Performance metrics documented
- [ ] Feature importance scientifically reasonable
- [ ] Confidence intervals properly calibrated
- [ ] Reproducibility verified (fixed seeds)

## Why Trust DATect Results

1. **Mathematical impossibility of leakage**: Temporal constraints enforced in code
2. **Operational realism**: Data availability delays match real-world constraints
3. **Conservative evaluation**: No optimistic metrics
4. **Comprehensive validation**: 7 critical tests on every run
5. **Transparent implementation**: All safeguards documented and auditable

## References

1. Trainer, V.L., et al. (2007). "Pseudo-nitzschia physiological ecology, phylogeny, toxicity, monitoring and impacts on ecosystem health." *Harmful Algae*.

2. Anderson, C.R., et al. (2021). "Predicting harmful algal blooms: A machine learning approach for early warning systems." *Marine Environmental Research*.

3. Wells, M.L., et al. (2020). "Toxic and harmful algal blooms in temperate coastal waters." *Oceanography*.
