# Data Pipeline - Technical Details

## Overview

This document describes the technical details of the DATect data ingestion pipeline (`dataset-creation.py`), which transforms raw environmental data into a unified, forecast-ready dataset.

## Source Data

### Biological Targets

**Domoic Acid (DA)**
- Source: State monitoring programs (WA DOH, OR DFW)
- Format: CSV files with varying date formats and headers
- Processing: Parse dates, normalize units, handle "<1" values

**Pseudo-nitzschia (PN)**
- Source: Same monitoring programs
- Format: CSV with cell count data
- Processing: Fuzzy column name matching, weekly aggregation

### Satellite Data (MODIS)

| Variable | Product | Resolution |
|----------|---------|------------|
| Chlorophyll-a | erdMWchla8day | 8-day, 4km |
| SST | erdMWsstd8day | 8-day, 4km |
| PAR | erdMWpar08day | 8-day, 4km |
| Fluorescence | erdMWcflh8day | 8-day, 4km |
| K490 | erdMWk4908day | 8-day, 4km |
| CHL anomaly | osu2ChlaAnom | Monthly |
| SST anomaly | osu2SstAnom | Monthly |

**Processing:**
- Monthly chunking to avoid ERDDAP output caps
- 4km spatial averaging around each site
- 7-day temporal buffer for 8-day composites
- 2-month buffer for monthly anomalies

### Climate Indices

| Index | Source | Buffer |
|-------|--------|--------|
| PDO | NOAA ERDDAP | 2 months |
| ONI | NOAA ERDDAP | 2 months |
| BEUTI | NOAA ERDDAP | 2 months |

### Streamflow

- Source: USGS Columbia River (Site 14246900)
- Format: JSON from NWIS API
- Processing: As-of merge with 7-day backward tolerance

## Pipeline Steps

### 1. CSV to Parquet Conversion

Convert raw CSVs for faster I/O:
```python
# DA files
for site, path in ORIGINAL_DA_FILES.items():
    df = pd.read_csv(path)
    df.to_parquet(output_path)
```

### 2. DA Processing

```python
# Parse heterogeneous date columns
# Handle "CollectDate" or "Harvest Month/Date/Year" formats
# Normalize "<1" values
# Weekly aggregation using MAX (safety-first for toxins)
da_df['Year-Week'] = da_df['date'].dt.strftime('%G-%V')
da_weekly = da_df.groupby(['site', 'Year-Week'])['da'].max()
```

### 3. PN Processing

```python
# Detect PN columns by fuzzy name match
# Weekly MAX aggregation for bloom detection
pn_weekly = pn_df.groupby(['site', 'Year-Week'])['pn'].max()
```

### 4. Climate Index Processing

```python
# Apply 2-month reporting lag
# Monthly data lagged by 2 months before joining
climate_df['TargetMonth'] = climate_df['date'].dt.to_period('M') - 2
```

### 5. BEUTI Processing

```python
# Read gridded NetCDF
# Inverse-distance-weight interpolation to site coordinates
# Daily to weekly alignment
```

### 6. Streamflow Processing

```python
# Parse nested JSON
# Coerce to numeric
# As-of merge with 7-day backward tolerance
```

### 7. Satellite Processing

```python
# Monthly downloads per site/variable
# Stitch monthly files
# Average spatial dimensions
# Apply temporal buffers:
#   - Regular data: 7-day delay
#   - Anomalies: 2-month delay (same as climate)
```

### 8. Weekly Lattice Generation

```python
# Create complete site-week grid
# Include lat/lon coordinates
weeks = pd.date_range(START_DATE, END_DATE, freq='W-MON')
sites = list(SITES.keys())
lattice = pd.MultiIndex.from_product([weeks, sites])
```

### 9. Merge All Sources

```python
# Join climate (lagged), streamflow (as-of), BEUTI, satellite
# All joins use only data available at the time
final_df = lattice.merge(climate, on='YearMonth-2')
final_df = final_df.merge(streamflow, on='date', direction='backward')
final_df = final_df.merge(beuti, on=['site', 'week'])
final_df = final_df.merge(satellite, on=['site', 'week'])
```

### 10. Biological Gap Handling

```python
# Short gaps: Exponential decay interpolation
# DA: up to 2 weeks, decay rate 0.2/week
# PN: up to 4 weeks, decay rate 0.3/week

# Long gaps: Fill with 0 (assumes non-detection)
```

### 11. Column Standardization

Final columns:
- `date`, `site`, `da`, `pn`
- `oni`, `pdo`, `beuti`, `discharge`
- `sat_chlor`, `sat_sst`, `sat_par`, `sat_flh`, `sat_k490`
- `sat_chlor_anom`, `sat_sst_anom`
- `lat`, `lon`

### 12. Export

```python
final_df.to_parquet('data/processed/final_output.parquet')
```

## Temporal Integrity

All merges use only data available strictly before the forecast anchor:

| Data Type | Delay | Rationale |
|-----------|-------|-----------|
| Satellite 8-day | 7 days | Processing time |
| Climate indices | 2 months | Reporting delay |
| Monthly anomalies | 2 months | Same as climate |
| Streamflow | Real-time | As-of merge |

## Data Quality

### Validation Steps

1. **Schema checks**: Required columns, types
2. **Range checks**: Non-negative DA, valid dates
3. **Site coverage**: All 10 sites present
4. **Date range**: Full coverage 2003-2023

### Missing Data Strategy

| Data Type | Short Gaps | Long Gaps |
|-----------|------------|-----------|
| DA | Decay interpolation (2 weeks) | Fill with 0 |
| PN | Decay interpolation (4 weeks) | Fill with 0 |
| Satellite | Leave as NaN | Leave as NaN |
| Climate | Forward fill | Median backup |

## Performance

- **Parquet format**: Fast columnar I/O
- **Monthly chunking**: Avoid ERDDAP caps
- **Intermediate caching**: Skip reprocessing
- **Parallel-friendly**: Per-site/variable loops

## Output Dataset

`data/processed/final_output.parquet`:
- Weekly time series (ISO weeks, Monday anchor)
- 2003-2023 coverage
- 10 sites × ~1090 weeks ≈ 10,900 records
- ~15-20 features per record
