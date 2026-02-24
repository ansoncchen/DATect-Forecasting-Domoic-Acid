# Dataset Creation - Scientific Decisions

## Overview

This document explains the scientific rationale behind key decisions in the DATect dataset creation pipeline (`dataset-creation.py`).

## Key Priorities

1. **Temporal Integrity**: Preventing data leakage
2. **Biological Realism**: Reflecting natural toxin dynamics
3. **Scientific Rigor**: Defensible methods
4. **Forecasting Relevance**: Features available in real-world predictions

## Decision 1: Biological Decay Interpolation

### Problem

Original approach used linear interpolation with unlimited forward-fill:
- Created artificial smoothing
- No biological basis
- Potential temporal leakage

### Solution

Exponential decay interpolation based on toxin clearance rates:

```python
# DA Parameters
DA_MAX_GAP_WEEKS = 2   # Conservative gap limit
DA_DECAY_RATE = 0.2    # Per week (3.5-week half-life)

# PN Parameters
PN_MAX_GAP_WEEKS = 4   # More aggressive (sparser data)
PN_DECAY_RATE = 0.3    # Per week (2.3-week half-life)
```

### Justification

- **DA decay rate (0.2/week)**: Within published razor clam depuration rates
- **Exponential decay**: Reflects natural toxin clearance
- **Conservative gap limits**: Based on observed measurement frequency
- **Uses only past values**: No forward-looking information

### Data Support

- Newport DA: 48% zero measurements, gaps averaging 59 days
- Long Beach DA: 0% zeros, 17.8% >20 μg/g
- PN data: Much sparser (131-2000 records)

## Decision 2: Satellite Temporal Buffers

### Problem

Satellite data has variable availability and processing delays.

### Solution

Differential temporal buffers:

| Data Type | Buffer | Rationale |
|-----------|--------|-----------|
| 8-day composites | 7 days | Processing delay |
| Monthly anomalies | 2 months | Same as climate indices |

### Implementation

```python
if is_anomaly_var:
    # 2-month buffer for monthly products
    safe_month = current_month - 2
else:
    # 7-day buffer for 8-day composites
    cutoff_date = target_date - 7 days
```

No fallback to less-delayed data - if unavailable with buffer, leave as NaN.

## Decision 3: Climate Index Delay

### Solution

2-month reporting lag for all climate indices (PDO, ONI, BEUTI):

```python
climate_df['TargetMonth'] = climate_df['date'].to_period('M') - 2
```

### Justification

- Climate indices have 1-2 month reporting delays
- Conservative 2-month lag ensures operational availability
- Reflects real-world forecasting constraints

## Decision 4: BEUTI Gap Filling

### Problem

Original approach filled missing BEUTI with zero, which:
- Created artificial "neutral upwelling" periods
- BEUTI can legitimately be negative (downwelling)

### Solution

Forward-fill with median backup:

```python
# Forward-fill (upwelling patterns persist)
beuti = beuti.fillna(method='ffill')

# Remaining gaps: use median (preserves distribution)
beuti = beuti.fillna(beuti.median())
```

### Justification

- Upwelling patterns persist over multiple weeks
- Median preserves natural BEUTI distribution
- Zero would incorrectly imply neutral conditions

## Decision 5: Weekly Aggregation

### Solution

ISO week aggregation with Monday anchor:

```python
df['Year-Week'] = df['date'].dt.strftime('%G-%V')
df['date'] = pd.to_datetime(df['Year-Week'] + '-1', format='%G-%V-%w')
```

### Justification

- **ISO weeks**: Consistent 52-53 week years
- **Monday anchor**: Aligns with monitoring schedules
- **Coherent alignment**: All data sources on same weekly boundaries

## Decision 6: DA/PN Aggregation Method

### Solution

Weekly MAX aggregation for both DA and PN:

```python
da_weekly = da_df.groupby(['site', 'week'])['da'].max()
pn_weekly = pn_df.groupby(['site', 'week'])['pn'].max()
```

### Justification

- **Safety-first**: Captures worst-case toxin exposure
- **Bloom detection**: Max captures peak bloom intensity
- **Consistent approach**: Same method for both biological signals

## Decision 7: Extended Gap Handling

### DA/PN Extended Gaps

Fill gaps >2-4 weeks with 0:
- Extended sampling gaps likely represent non-detection
- Avoids artificial persistence of old measurements

### Environmental Data

Forward-fill with median backup:
- Environmental conditions persist naturally
- Should not default to artificial zero/neutral states

## Impact on Model Performance

### Positive

1. **Realistic baselines**: Decay prevents artificial naive model improvement
2. **Temporal safety**: Buffers prevent retrospective leakage
3. **Feature relevance**: Only operationally available data used

### Tradeoffs

1. **Reduced density**: Conservative gaps reduce training data
2. **Increased complexity**: Decay interpolation requires parameter tuning
3. **Validation overhead**: Temporal checks slow processing

## References

1. Trainer, V.L., et al. (2007). "Pseudo-nitzschia physiological ecology, phylogeny, toxicity, monitoring and impacts on ecosystem health." *Harmful Algae*, 14, 271-300.

2. Wekell, J.C., et al. (1994). "Occurrence of domoic acid in Washington State razor clam populations." *Journal of Shellfish Research*, 13(1), 197-205.

3. Lefebvre, K.A., et al. (2002). "Detection of domoic acid in northern anchovies and California sea lions." *Natural Toxins*, 6(6), 207-211.
