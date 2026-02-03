# DATect Pipeline Overview

## Complete System Architecture

The DATect forecasting system is a comprehensive machine learning pipeline for predicting domoic acid (DA) concentrations along the Pacific Coast. This document provides a high-level overview of how the entire pipeline works from data ingestion to prediction.

## Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION STAGE                          │
│  (dataset-creation.py - Run once, takes 30-60 minutes)         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  1. Raw Data Sources                   │
        │  • MODIS Satellite (8-day composites)   │
        │  • Climate Indices (PDO, ONI, BEUTI)   │
        │  • Streamflow (USGS Columbia River)     │
        │  • DA Measurements (10 sites)           │
        │  • Pseudo-nitzschia Cell Counts         │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  2. Data Processing                     │
        │  • Download & spatial aggregation        │
        │  • Temporal alignment (weekly grid)      │
        │  • Apply temporal buffers:               │
        │    - 7-day satellite delay              │
        │    - 2-month climate index delay        │
        │  • Biological decay interpolation        │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  3. Output: final_output.parquet        │
        │  Unified weekly time series with:        │
        │  • All environmental features            │
        │  • DA & PN measurements                 │
        │  • Site coordinates                      │
        └─────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    FORECASTING STAGE                            │
│  (forecast_engine.py - Called per forecast request)             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  4. Data Loading & Preparation         │
        │  (data_processor.py)                   │
        │  • Load final_output.parquet            │
        │  • Add temporal features (sin/cos)      │
        │  • Add rolling statistics (optional)    │
        │  • Create lag features (1, 3 weeks)    │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  5. Temporal Validation                 │
        │  (validation.py)                        │
        │  • Validate forecast date                │
        │  • Check temporal buffers                │
        │  • Ensure no future data leakage         │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  6. Feature Engineering                 │
        │  (data_processor.py)                    │
        │  • Calculate anchor date                 │
        │    (forecast_date - FORECAST_HORIZON)   │
        │  • Create lag features with cutoff       │
        │  • Apply temporal cutoffs to all data    │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  7. Train/Test Split                    │
        │  • Training: all data ≤ anchor_date     │
        │  • Test: forecast_date                   │
        │  • Per-forecast DA category creation     │
        │    (prevents target leakage)            │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  8. Model Training                       │
        │  (model_factory.py)                     │
        │  • XGBoost (default) or Linear          │
        │  • Regression or Classification         │
        │  • Sample weighting (optional)           │
        │  • Feature transformation pipeline       │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  9. Prediction                          │
        │  • Generate forecast for target date     │
        │  • Bootstrap confidence intervals        │
        │  • Feature importance extraction         │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  10. Results                            │
        │  • Predicted DA concentration            │
        │  • Risk category (if classification)     │
        │  • Confidence intervals                  │
        │  • Feature importance rankings           │
        └─────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    API & FRONTEND STAGE                         │
│  (backend/api.py + frontend/)                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  11. Web Interface                      │
        │  • Dashboard for single forecasts        │
        │  • Historical analysis                   │
        │  • Retrospective evaluation               │
        │  • Visualizations                        │
        └─────────────────────────────────────────┘
```

## Detailed Stage Breakdown

### Stage 1: Data Ingestion (`dataset-creation.py`)

**Purpose**: Download and integrate all environmental and biological data sources into a unified dataset.

**Key Steps**:
1. **Satellite Data Processing**:
   - Downloads MODIS-Aqua 8-day composite data monthly
   - Variables: Chlorophyll-a, SST, PAR, Fluorescence, K490
   - Monthly anomalies: CHLA-ANOM, SST-ANOM
   - Spatial aggregation: 4km radius around each site
   - Temporal safeguard: 7-day processing buffer

2. **Climate Indices**:
   - PDO (Pacific Decadal Oscillation)
   - ONI (Oceanic Niño Index)
   - BEUTI (Biologically Effective Upwelling Transport Index)
   - Temporal safeguard: 2-month reporting delay

3. **Streamflow Data**:
   - USGS Columbia River discharge (daily)
   - Merged using backward fill with 7-day tolerance

4. **Biological Data**:
   - DA measurements: Weekly aggregation (max per week)
   - PN cell counts: Weekly aggregation (max per week)
   - Biological decay interpolation for gaps

5. **Output**: `data/processed/final_output.parquet`
   - Weekly time series (2002-2023)
   - All features aligned temporally
   - Ready for forecasting

**Runtime**: 30-60 minutes (only run when data changes)

---

### Stage 2: Data Loading (`data_processor.py`)

**Purpose**: Load and prepare data for forecasting with temporal features.

**Key Steps**:
1. Load `final_output.parquet`
2. Add temporal encoding:
   - `sin_day_of_year`, `cos_day_of_year`
   - `sin_month`, `cos_month`
   - `quarter`, `days_since_start`
3. Add rolling statistics (optional):
   - 2-week, 4-week, 8-week rolling means/stds/trends
4. Sort by site and date

---

### Stage 3: Single Forecast Generation (`forecast_engine.py`)

**Purpose**: Generate a single forecast for a specific date and site.

**Key Steps**:

1. **Input Validation**:
   - Validate site exists
   - Validate forecast date is reasonable
   - Check data availability

2. **Anchor Date Calculation**:
   ```python
   anchor_date = forecast_date - FORECAST_HORIZON_DAYS
   ```
   - Default: 7 days (1 week ahead forecast)
   - Ensures realistic data availability

3. **Lag Feature Creation**:
   - Creates lag features (1-week, 3-week) with temporal cutoff
   - Only uses data ≤ anchor_date for lag calculations
   - Prevents future information leakage

4. **Train/Test Split**:
   - **Training**: All data where `date ≤ anchor_date`
   - **Test**: Data at `forecast_date`
   - Strict chronological ordering

5. **DA Category Creation** (for classification):
   - Categories created from training data only
   - Thresholds: Low (0-5), Moderate (5-20), High (20-40), Extreme (>40)
   - Prevents target leakage

6. **Feature Transformation**:
   - Create numeric transformer (unfitted)
   - Fit transformer on training data only
   - Transform both training and test features

7. **Model Training**:
   - Train XGBoost or Linear model
   - Use training data only
   - Optional sample weighting for class imbalance

8. **Prediction**:
   - Generate prediction for forecast date
   - Bootstrap confidence intervals (regression)
   - Feature importance extraction

9. **Output**:
   ```python
   {
       'forecast_date': date,
       'site': site_name,
       'predicted_da': float,  # Regression
       'predicted_category': int,  # Classification
       'bootstrap_quantiles': {'q05': float, 'q50': float, 'q95': float},
       'feature_importance': DataFrame,
       'training_samples': int
   }
   ```

---

### Stage 4: Retrospective Evaluation (`forecast_engine.py`)

**Purpose**: Validate model performance using historical data.

**Key Steps**:
1. Generate random anchor points (500 per site by default)
2. For each anchor:
   - Use data ≤ anchor_date for training
   - Find test sample closest to `anchor_date + FORECAST_HORIZON_DAYS`
   - Generate forecast
   - Compare with actual measurement
3. Aggregate metrics:
   - Regression: R², MAE, F1 (spike detection)
   - Classification: Accuracy, Precision, Recall, F1

**Temporal Safeguards**:
- No future data in training
- Minimum temporal buffer enforced
- Per-forecast category creation
- Satellite/climate delays respected

---

### Stage 5: API & Frontend (`backend/api.py` + `frontend/`)

**Purpose**: Provide web interface for forecasts and analysis.

**Key Endpoints**:
- `POST /api/forecast`: Single forecast
- `POST /api/forecast/enhanced`: Both regression + classification
- `POST /api/retrospective`: Run retrospective evaluation
- `GET /api/visualizations/*`: Various analysis plots
- `GET /api/sites`: Available sites and date ranges

**Frontend Pages**:
- **Dashboard**: Single forecast interface
- **Historical**: Time series analysis
- **About**: System documentation

---

## Key Design Principles

### 1. Temporal Integrity
- **No data leakage**: Training data always before test data
- **Temporal buffers**: Realistic data availability delays
- **Per-forecast categories**: Categories created from training data only
- **Strict chronological ordering**: No random splits

### 2. Scientific Rigor
- **Operational realism**: Incorporates actual data latencies
- **Biological principles**: Respects ecological time scales
- **Reproducibility**: Fixed random seeds, versioned dependencies
- **Validation**: 7 comprehensive temporal integrity tests

### 3. Performance Optimization
- **Caching**: Pre-computed results for common forecasts
- **Parallel processing**: Retrospective evaluation uses joblib
- **Efficient data structures**: Parquet format for fast I/O

---

## Configuration (`config.py`)

Key settings that control pipeline behavior:

```python
# Forecast horizon
FORECAST_HORIZON_WEEKS = 1  # How many weeks ahead to forecast
FORECAST_HORIZON_DAYS = 7   # Derived value

# Model selection
FORECAST_MODEL = "xgboost"   # or "linear"
FORECAST_TASK = "regression" # or "classification"

# Feature engineering
USE_LAG_FEATURES = False     # Enable lag features
LAG_FEATURES = [1, 3]        # Lag periods (weeks)
USE_ROLLING_FEATURES = False # Rolling statistics
USE_ENHANCED_TEMPORAL_FEATURES = True  # Sin/cos encoding

# Model parameters
XGB_REGRESSION_PARAMS = {...}
XGB_CLASSIFICATION_PARAMS = {...}

# Evaluation
N_RANDOM_ANCHORS = 500       # Retrospective evaluation points
N_BOOTSTRAP_ITERATIONS = 20  # Confidence interval samples
```

---

## Data Flow Summary

1. **Raw Data** → `dataset-creation.py` → **Unified Dataset** (`final_output.parquet`)
2. **Unified Dataset** → `data_processor.py` → **Prepared Features**
3. **Prepared Features** → `forecast_engine.py` → **Predictions**
4. **Predictions** → `backend/api.py` → **JSON API Responses**
5. **API Responses** → `frontend/` → **User Interface**

---

## Running the Pipeline

### Initial Setup (One-time)
```bash
# Generate unified dataset (30-60 minutes)
python dataset-creation.py
```

### Daily Operations
```bash
# Start web application
python run_datect.py
# Opens browser at http://localhost:3000
```

### Retrospective Evaluation
```bash
# Run via API endpoint
POST /api/retrospective
# Or programmatically:
from forecasting.forecast_engine import ForecastEngine
engine = ForecastEngine()
results = engine.run_retrospective_evaluation(
    task="regression",
    model_type="xgboost",
    n_anchors=500
)
```

---

## Temporal Safeguards Summary

The pipeline implements 7 critical temporal safeguards:

1. **Chronological Split**: Training always before test
2. **Temporal Buffer**: Minimum gap between train/test
3. **Future Information Quarantine**: No post-anchor data in features
4. **Per-Forecast Category Creation**: No target leakage
5. **Satellite Delay Simulation**: 7-day processing buffer
6. **Climate Data Lag**: 2-month reporting delay
7. **Cross-Site Consistency**: Same temporal rules everywhere

These safeguards ensure that retrospective validation accurately reflects prospective forecasting performance.

---

## Performance Metrics

**Regression Performance**:
- R² ≈ 0.366 (realistic for environmental forecasting)
- MAE ≈ 4.57 μg/g
- F1 (spike detection) ≈ 0.5-0.6

**Classification Performance**:
- Accuracy ≈ 77.6%
- Balanced accuracy accounts for class imbalance
- Per-class metrics available

**Note**: These metrics reflect genuine operational constraints with strict temporal safeguards, not idealized scenarios.

---

## File Structure

```
DATect-Forecasting-Domoic-Acid/
├── dataset-creation.py          # Data ingestion pipeline
├── run_datect.py                # Application launcher
├── config.py                    # Configuration
├── forecasting/
│   ├── forecast_engine.py      # Core forecasting logic
│   ├── data_processor.py        # Data loading & feature engineering
│   ├── model_factory.py         # Model creation & training
│   └── validation.py            # Temporal integrity checks
├── backend/
│   ├── api.py                   # FastAPI endpoints
│   ├── visualizations.py        # Plot generation
│   └── cache_manager.py         # Result caching
├── frontend/                    # React web interface
└── data/
    ├── raw/                     # Raw CSV files
    ├── intermediate/            # Satellite data cache
    └── processed/
        └── final_output.parquet # Unified dataset
```

---

## Conclusion

The DATect pipeline is a scientifically rigorous forecasting system that:

1. **Integrates** multiple environmental data sources
2. **Maintains** strict temporal integrity
3. **Generates** accurate predictions with uncertainty quantification
4. **Provides** accessible web interface for end users
5. **Validates** performance through comprehensive retrospective testing

The system is designed for both research publication and operational deployment, with comprehensive validation at every stage to ensure trustworthy, reproducible domoic acid forecasts.

