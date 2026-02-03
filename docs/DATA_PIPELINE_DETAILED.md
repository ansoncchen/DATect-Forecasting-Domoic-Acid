# DATect Data Pipeline – Raw Data to Forecast-Ready Dataset

**Purpose**  
This document gives a detailed, end-to-end account of how the DATect project ingested messy, fragmented government data (public records + NOAA/USGS ERDDAP servers) and turned it into a leak-free, forecast-ready weekly dataset (`data/processed/final_output.parquet`). It highlights the engineering decisions, failure handling, and quality controls that map directly to the USAFacts Data Engineering Intern responsibilities.

## 1) Source Landscape and Acquisition Challenges

- **Biological targets (DA, PN) via public records**  
  - Data arrived as mixed-format CSVs from multiple agencies, with inconsistent headers, date formats, units, and site naming.  
  - Required one-off cleaning rules per source and defensive parsing (string dates, "<1" values, missing fields).

- **Environmental drivers via legacy government servers**  
  - **NOAA ERDDAP (MODIS, climate indices, anomalies):** fragmented endpoints, 50k-row practical output caps, occasional timeouts, and separate products for composites vs. anomalies.  
  - **USGS NWIS (Columbia River flow):** JSON payloads with nested structures and multiple series per request.  
  - **BEUTI upwelling index:** gridded NetCDF requiring spatial interpolation to site coordinates.

- **Operational constraints**  
  - ERDDAP enforces modest response limits; large pulls had to be chunked by month.  
  - Satellite anomalies and climate indices published with latency, demanding explicit temporal buffers to avoid leakage.  
  - Mixed coordinate conventions (0–360 vs. −180–180) and varying spatial resolutions.

## 2) Acquisition Strategy (resilient, chunked, reproducible)

- **Monthly chunking for ERDDAP pulls** to stay under output caps and reduce retries; temporary NetCDFs stitched per variable/site.  
- **Progress + cleanup discipline:** every downloaded file recorded, temp dirs always removed, small-size sanity checks before processing.  
- **Adaptive variable detection:** flexible mappings to find the correct data variable in each NetCDF (e.g., `chlorophyll`, `sst`, `k490`, anomaly variants).  
- **Spatial handling:** average within a ~4 km radius of each monitoring site; harmonize longitude conventions.  
- **Temporal buffers baked in:**  
  - Satellite composites: enforce ≥7-day delay before a date can be used.  
  - Climate/anomaly indices: enforce ≥2-month reporting delay.  
- **Fallbacks for spotty servers:** skip-empty-month logic, retry-safe loops, and no-fail final assembly (missing chunks become NaN, never silently forward-filled across future boundaries).

## 3) Ingestion Pipeline (dataset-creation.py)

Below is the concrete sequence that turns raw sources into the unified weekly parquet:

1) **Convert agency CSVs to Parquet** for DA and PN (faster I/O, consistent types).  
2) **DA processing**  
   - Parse heterogeneous date columns (e.g., `CollectDate`, or composed `Harvest Month/Date/Year`).  
   - Normalize “<1” style values; coerce to numeric.  
   - Weekly aggregation by ISO week using **max** (safety-first for toxin exposure).  
3) **PN processing**  
   - Detect PN columns by fuzzy name match; weekly max aggregation for bloom detection.  
4) **Climate indices (PDO, ONI)**  
   - Monthly aggregation; apply a strict 2-month lag before a forecast date can see a value.  
5) **BEUTI upwelling**  
   - Read gridded NetCDF; inverse-distance-weight interpolation by latitude to each site; daily to weekly alignment.  
6) **Streamflow (USGS 14246900)**  
   - Parse nested JSON; coerce to numeric; as-of merge with 7-day backward tolerance (no lookahead).  
7) **Satellite composites (MODIS)**  
   - Monthly downloads per site/variable; stitch; average spatial dims; pivot to `sat_*` columns.  
   - Enforce ≥7-day delay; anomalies (`chla-anom`, `sst-anom`) use the 2-month climate buffer.  
8) **Weekly lattice generation**  
   - Create a complete site-week grid between `START_DATE` and `END_DATE` with lat/lon included.  
9) **Merge all sources**  
   - Join climate (lagged), streamflow (as-of), BEUTI, satellite features onto the site-week grid.  
10) **Biological gap handling**  
    - Short gaps: exponential decay interpolation (DA up to 2 weeks, PN up to 4 weeks) to respect biological decay.  
    - Long gaps: filled with zeros (assumes non-detection), never with future data.  
11) **Column standardization**  
    - Canonical names (`date`, `site`, `da`, `pn`, `oni`, `pdo`, `discharge`, `beuti`, `sat_*`).  
12) **Export**  
    - Save to `data/processed/final_output.parquet`; remove temp files and downloads.

## 4) Temporal Integrity (leak-free by construction)

- **No forward-looking data in features:** all merges use only data available strictly before the forecast anchor.  
- **Delays enforced in code:** 7-day satellite buffer, 2-month climate/anomaly buffer, 1+ day buffer for lags.  
- **Per-forecast category creation:** DA risk categories derived only from training-era data to avoid target leakage.  
- **Chronological splits:** training ≤ anchor_date; test at forecast_date; never random splits.

## 5) Data Quality and Validation

- Schema checks: required columns, non-negative DA, valid dates, site coverage.  
- Range checks: drop impossible values; log anomalies.  
- Temporal coverage checks: ensure ≥1 year coverage before allowing a site into training.  
- Missing-data strategy: decay interpolation for short gaps; explicit NaN for unavailable environmental features; no forward fill past anchor.  
- Auditability: verbose logging, printed counts per stage, and safe fallbacks that keep the pipeline running while surfacing warnings.

## 6) Reliability and Performance Tactics

- Parquet everywhere for speed and columnar access.  
- Chunked ERDDAP requests to avoid server caps and timeouts.  
- Caching intermediate satellite parquet to skip reprocessing when data unchanged.  
- Deterministic seeds and reproducible runs; temp dirs always cleaned to prevent disk bloat.  
- Parallel-friendly structure (per-site/per-variable loops) while keeping external calls polite (no excessive concurrent hits to ERDDAP).

## 7) ML Readiness

- Weekly, aligned time series with explicit lat/lon and harmonized site naming.  
- Optional rolling statistics and temporal encodings (sin/cos day-of-year, month) added downstream.  
- Feature importance compatibility (flat numeric columns).  
- Consistent units and thresholds for DA risk categories.

## 8) How This Maps to the USAFacts Internship Priorities

- **Transform raw government data into reliable, accessible formats:** stitched multiple NOAA ERDDAP products, USGS JSON, and FOIA-style CSVs into one canonical parquet.  
- **Handle fragmented, rate-limited sources:** monthly chunking, retry-safe downloads, and interpolation for gridded products.  
- **Ensure data quality and temporal validity:** rigorous buffers (7-day satellite, 2-month climate), leak-free joins, and schema/range checks.  
- **Document and communicate:** this file plus `PIPELINE_OVERVIEW.md`, `FORECAST_PIPELINE.md`, and in-code logging describe the pipeline clearly.  
- **Support AI/ML downstream:** produced a clean, feature-complete dataset and safeguarded feature engineering for model training.  
- **Performance and reliability mindset:** caching, Parquet optimization, chunked pulls, deterministic runs, and cleanup discipline.  
- **Cross-functional readiness:** the pipeline design anticipates product needs (fast API responses via cached data), scientific constraints, and operational realism.

## 9) Talking Points for Interviews

- Navigated messy public-records CSVs and heterogeneous government APIs; built bespoke parsers and chunked ERDDAP pulls to stay under output limits.  
- Enforced realistic data-latency buffers to eliminate leakage—mirrors production-grade governance of “data availability” vs. “data timestamp.”  
- Implemented decay-based interpolation for biological signals, preserving scientific realism while improving coverage.  
- Delivered a reproducible, fully logged pipeline that emits a single, authoritative parquet for the app and models.  
- Balanced throughput and server friendliness with monthly chunking, temp-file hygiene, and cached intermediates.

