# Visualizations Guide

## Overview

DATect provides scientific visualizations for understanding forecasts, validating models, and exploring oceanographic relationships.

## Dashboard Visualizations

### Forecast Results

**Prediction Summary Cards**

| Card | Description |
|------|-------------|
| Regression | Predicted DA concentration (μg/g) with uncertainty |
| Classification | Risk category with confidence percentage |

**Risk Category Colors:**
- **Low (Green)**: 0-5 μg/g - Safe
- **Moderate (Yellow)**: 5-20 μg/g - Caution
- **High (Orange)**: 20-40 μg/g - Avoid
- **Extreme (Red)**: >40 μg/g - Hazard

### Gradient Uncertainty Plot

Visualizes prediction confidence using quantile/bootstrap intervals (configurable):

- **Q50 line**: Median prediction
- **Q05/Q95 markers**: 90% confidence interval
- **Gradient band**: Color intensity shows confidence
- **XGBoost point**: Primary prediction
- **Actual value (X)**: Ground truth when available

**Interpretation:**
- Narrow band = High confidence
- Wide band = High uncertainty
- Point near Q50 = Consistent predictions

### Feature Importance

**XGBoost Gain**
- Tree-based importance (contribution to model)
- Values sum to 1.0
- Higher = more predictive

**Permutation Importance**
- Model-agnostic importance
- Shows performance drop when feature is scrambled
- Higher = more critical

## Historical Analysis

### Correlation Heatmap

Shows Pearson correlations between variables:

| Color | Correlation |
|-------|-------------|
| Red | Strong positive |
| Blue | Strong negative |
| White | Near zero |

**Key relationships:**
- DA vs BEUTI: Positive (upwelling drives blooms)
- DA vs Chlorophyll: Positive (bloom conditions)
- SST vs PDO: Climate influence on temperature

### Time Series Comparison

DA vs Pseudo-nitzschia over time:

**Look for:**
- DA peaks following PN blooms (1-3 week lag)
- Seasonal patterns (spring/summer peaks)
- Correlation between bloom intensity and toxin levels

### Sensitivity Analysis (Sobol Indices)

Quantifies variable contributions:

**First-order indices**: Direct effect of each variable
**Interaction terms**: Combined effects between variables

**Interpretation:**
- Higher index = Stronger influence
- Interaction effects indicate synergistic relationships

### Spectral Analysis

Frequency domain analysis of DA time series:

**Power Spectral Density:**
- Peaks indicate dominant cycles
- 52-week = Annual pattern
- 12-week = Seasonal
- 4-week = Upwelling cycles

## Model Performance

### Scatter Plot (Predicted vs Actual)

- **Diagonal line**: Perfect prediction
- **Point scatter**: Model performance
- **Clustering near diagonal**: Good predictions
- **Outliers**: Check for unusual conditions

### Residual Analysis

Prediction errors vs predicted values:

**Healthy patterns:**
- Horizontal band around zero (no bias)
- Constant spread (homoscedasticity)
- No funnel shape (equal variance)

**Warning signs:**
- Systematic trends (bias)
- Increasing spread (heteroscedasticity)

### Confusion Matrix (Classification)

| | Predicted Low | Predicted High |
|-|---------------|----------------|
| **Actual Low** | True Negative | False Positive |
| **Actual High** | False Negative | True Positive |

- Diagonal = Correct predictions
- Off-diagonal = Misclassifications
- Near-diagonal errors more acceptable than far

### ROC Curves

Per-category discrimination:

| AUC | Interpretation |
|-----|----------------|
| 1.0 | Perfect |
| 0.8-0.9 | Good |
| 0.7-0.8 | Moderate |
| 0.5 | Random |

## Data Quality Indicators

### Missing Data Heatmap

Shows data completeness by site and time:

- **Dark**: Complete data
- **Light**: Missing data
- **Patterns**: Identify seasonal gaps or site issues

### Learning Curves

Training progress visualization:

- **Training score**: Performance on training data
- **Validation score**: Performance on held-out data
- **Gap**: Overfitting indicator
- **Convergence**: When scores stabilize

## API Endpoints

| Endpoint | Visualization |
|----------|---------------|
| `/api/visualizations/correlation/{site}` | Correlation heatmap |
| `/api/visualizations/sensitivity/{site}` | Sobol indices |
| `/api/visualizations/comparison/{site}` | Time series comparison |
| `/api/visualizations/spectral/{site}` | Spectral analysis |
| `/api/visualizations/map` | Site map with risk levels |

## Best Practices

### Do:
- Check uncertainty alongside predictions
- Consider temporal context (seasonality)
- Validate feature importance makes scientific sense
- Monitor data quality
- Compare multiple metrics

### Avoid:
- Over-interpreting single predictions
- Ignoring uncertainty
- Correlations without causation understanding
- Small sample conclusions
- Ignoring seasonal confounds
