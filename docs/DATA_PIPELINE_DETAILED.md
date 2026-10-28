# Data Pipeline

How `data/processed/final_output.parquet` is built: the full pipeline in `dataset-creation.py`, the offline rebuild scripts that should be used for most changes, the measurement-parsing rules, and the reasoning behind each temporal safeguard.

**Related docs:** [`PIPELINE_DEEP_DIVE.md`](PIPELINE_DEEP_DIVE.md) covers how the forecasting engine consumes this file.

## Output dataset

`data/processed/final_output.parquet` (tracked in git):

- One row per site per Monday, `config.START_DATE` → `config.END_DATE` (2003-01-06 to 2023-12-25): 10 sites × 1,095 weeks = 10,950 rows.
- **33 columns**: `date`, `site`, `lat`, `lon`, `oni`, `pdo`, `discharge`, `da`, `pn`, `beuti`, 7 satellite columns (`chla-anom`, `modis-chla`, `modis-flr`, `modis-k490`, `modis-par`, `modis-sst`, `sst-anom`) and 16 OAD columns (14 score features + 2 cloud fractions, below).
- Column names are lowercase. `date` is stored as a string `"MM/DD/YYYY"`; call `pd.to_datetime(df["date"])` before joins.
- `da` and `pn` are gap-filled weekly series used as training targets/features. Evaluation never scores against them; it uses the raw measurements in `data/raw/da-input/`.

## Rebuilding: use the offline path first

`dataset-creation.py` rebuilds everything from scratch, which re-downloads every MODIS chunk and the PDO / ONI / BEUTI series from NOAA ERDDAP plus USGS streamflow. There is no tracked satellite cache and ERDDAP downloads are slow and unreliable, so **do not re-run it for changes that only touch measurements or OAD features**. Use:

```bash
# da, pn: re-parse data/raw/{da,pn}-input and redo the decay fill
.venv/bin/python scripts/data/rebuild_final_output.py --recompute measurements

# oad_*: recompute the 16 OAD features from data/processed/oad_scores.parquet
#        and data/processed/oad_cloud_fractions.parquet
.venv/bin/python scripts/data/rebuild_final_output.py --recompute oad

# compare only (exit 1 if anything would change); --output PATH writes elsewhere
.venv/bin/python scripts/data/rebuild_final_output.py --recompute all --check
```

The script imports `dataset-creation.py`'s own functions (`process_da`, `process_pn`, `compile_da_pn`, `add_oad_features`) without running `main()`, keeps every environmental and satellite column unchanged, and preserves the existing row order, column order, dtypes and date format. It prints per-column change counts. Recompute only what changed: the `*_30day_trend` OAD columns use `np.polyfit`, whose last bits differ across machines.

**Switching the OAD checkpoint.** DATect reads one tracked copy of a checkpoint's regional scores, `data/processed/oad_scores.parquet` (currently `ae_3d_l32_c4_t4_s42_mae070`). To change it:

```bash
.venv/bin/python scripts/data/export_oad_features.py ae_3d_l32_c4_t4_s42_mae070 \
    [--scores-dir /path/to/oad/outputs/scores]   # default: oad/outputs/scores (generated on Hyak)
.venv/bin/python scripts/data/rebuild_final_output.py --recompute oad
```

The export checks the schema (`date, region, method, aggregation, score, checkpoint`), a single method with `aggregation == "mean"`, every region a DATect site maps to, and unique `(date, region)`. The cloud-fraction input does not depend on the checkpoint; it comes from `compute_region_valid_fractions` in `oad/scripts/07_cloud_confound.py`.

Run the full `dataset-creation.py` only when environmental or satellite sources themselves change (new date range, new variable). Heavy runs belong on Hyak (see [`HYAK_SETUP.md`](HYAK_SETUP.md)).

## Measurement parsing (`forecasting/measurements.py`)

One implementation shared by `dataset-creation.py` (weekly `da`/`pn`), `raw_data_forecaster.load_raw_da_measurements` (evaluation targets) and the API's latest-reading lookup:

| Function | Rule |
|---|---|
| `parse_da_values` | Numeric strings as-is. Below-detection values `"<x"` become **x / 2** (LOD/2; `"<1"` → 0.5), so non-detects are real low observations, training targets and test points. Anything else non-numeric → NaN. |
| `parse_pn_values` | Accepts thousands separators (`"12,345"`, `"1,234,567"`). Ambiguous strings such as `"999,99"` stay NaN. |
| `parse_dates` | Strips surrounding whitespace before parsing; unparseable dates (e.g. typos) → NaT and the row is dropped. |

DA files use one of two schemas: `CollectDate` + `Domoic Result`, or `Harvest Month` / `Harvest Date` / `Harvest Year` + `Domoic Acid`. PN files need a `Date` column and exactly one column whose name contains both "pseudo" and "nitzschia". CSVs are read with pandas (Polars rejected several raw files). Same-day DA samples are kept as separate rows in the CSVs; every consumer aggregates them with **max** (weekly `da` here, weekly training targets and the one-row-per-site-day evaluation targets in `raw_data_forecaster`).

## Full pipeline (`dataset-creation.py` `main()`)

1. **CSV → Parquet** for the DA/PN inputs (temporary files, deleted at the end).
2. **Satellite** (`generate_satellite_parquet`): for each site and variable in `config.SATELLITE_DATA`, download month-sized chunks from ERDDAP (avoids output caps), stitch, and average over a per-site bounding box (0.2° for the seven northern sites, 0.4° for Newport, Coos Bay and Gold Beach). The result is cached at `config.SATELLITE_CACHE_PATH` (`data/intermediate/satellite_data_intermediate.parquet`, git-ignored) and reused unless `FORCE_SATELLITE_REPROCESSING = True` in the script.
   - 8-day composites: `erdMWchla8day`, `erdMWsstd8day`, `erdMWpar08day`, `erdMWcflh8day`, `erdMWk4908day`.
   - Monthly anomalies: `osu2ChlaAnom`, `osu2SstAnom`.
3. **DA / PN** (`process_da`, `process_pn`): parse as above, bucket by ISO week (`%G-%V`, Monday), weekly **MAX** per site.
4. **Environmental series**: USGS discharge (site 14246900, daily), PDO and ONI (monthly means of the ERDDAP series), BEUTI (daily, interpolated to each site's latitude by inverse-distance weighting with power 2).
5. **Weekly lattice** (`generate_compiled_data`): every site × Monday, with `lat`/`lon` from `config.SITES`.
6. **Merges** (`compile_data`, `compile_da_pn`, `add_satellite_data`, then BEUTI), with the lags in the next section.
7. **Gap fill** of `da` and `pn` (below).
8. **OAD features** (`forecasting/oad_features.add_oad_features`), when `data/processed/oad_scores.parquet` exists; the 2 cloud columns stay NaN if `oad_cloud_fractions.parquet` is missing.
9. **Select, rename, write**: `final_core_cols` + sorted `sat_*` columns + OAD columns; date to `"MM/DD/YYYY"`; write `config.FINAL_OUTPUT_PATH`.

**Adding a column:** `final_core_cols` is an allow-list, so a new column is silently dropped unless it is added there (or appended to `final_cols` like the OAD block). Satellite columns are renamed **by position** after sorting the `sat_*` names, so adding a satellite variable shifts the mapping; update `sat_mapping` too.

## Temporal safeguards

Each source is lagged so that a row dated *R* holds only information available by *R*. The forecasting engine then reads environmental values from the row at or before the anchor (target date − 7 days).

| Column(s) | How it is joined to row date *R* | Why |
|---|---|---|
| 8-day satellite (`modis-*`) | Latest composite with timestamp ≤ *R* − 7 days (`merge_asof`, backward, per site) | Processing / availability delay of composites |
| Monthly anomalies (`chla-anom`, `sst-anom`) | Value for month(*R*) − 2; falls back to month(*R*) − 1 when that month is missing | Monthly products are published after the month ends |
| `pdo`, `oni` | Monthly mean for month(*R*) − 2 | Reporting delay of climate indices |
| `discharge` | Latest daily value at or before *R*, within 7 days (`merge_asof`, backward) | Daily gauge data are available in near real time |
| `beuti` | Daily value on *R*; per-site forward fill, then the overall median for anything left | Upwelling regimes persist; 0 would wrongly mean "neutral" (BEUTI can be negative) |
| `oad_*` | Scores up to *R* − 5 days (see below) | MODIS 8-day composites behind the OAD scores are centered on their date |
| `da`, `pn` | Weekly MAX of measurements in the ISO week of *R*, then past-only gap fill | Worst-case exposure; the fill never looks forward |

Satellite, climate and discharge values that cannot be matched stay NaN; the engine median-imputes them from training rows.

### OAD features

`add_oad_features` maps each site to one of four OAD regions (`SITE_TO_REGION`: Olympic Coast, SW Washington / Long Beach, Central Oregon, Southern OR / N CA) plus the overall coastal region, and computes for each (local and overall):

`score` (7-day mean ending *R* − 5), `score_lag1week` (ending *R* − 12), `score_lag2week` (ending *R* − 19), `score_30day_mean` / `_max` / `_trend` (scores in [*R* − 35, *R* − 5], at least 5 days), `score_zscore_doy` (vs. prior years within ±16 days of the same day of year, at least 10 values). Local columns are prefixed `oad_`, overall ones `oad_overall_`. The two cloud controls `oad_valid_frac` / `oad_overall_valid_frac` are 7-day mean valid-pixel fractions ending *R* − 5.

The 5-day shift (`LEAK_SHIFT_DAYS`) exists because a composite dated *D* contains data from about *D* − 4 to *D* + 4: anchoring on *R* − 5 keeps every composite's last day before *R*.

## Gap filling of `da` and `pn`

`compile_da_pn` applies `biological_decay_fill` per site after the weekly merge, using `config.py` constants:

| Series | Decay rate per week | `*_MAX_GAP_WEEKS` |
|---|---|---|
| DA | `DA_DECAY_RATE = 0.2` (half-life ≈ 3.5 weeks) | `DA_MAX_GAP_WEEKS = 2` |
| PN | `PN_DECAY_RATE = 0.3` (half-life ≈ 2.3 weeks) | `PN_MAX_GAP_WEEKS = 4` |

A missing week *k* weeks after the last value *v* gets *v* · exp(−rate · *k*). Only earlier values are used, so the fill is causal. Weeks before a site's first measurement are set to 0.

**Actual behavior of the gap limit:** the loop fills at most `*_MAX_GAP_WEEKS` weeks at a time, but each filled week then counts as the "previous value" for the next stretch, so a gap of any length decays continuously until the next real measurement. The comment in `config.py` ("larger gaps filled with 0") does not describe what the code does; any limit ≥ 1 produces the same output. Any change to this must be followed by `rebuild_final_output.py --recompute measurements` and a re-run of the evaluation.

Why decay rather than linear interpolation or unlimited forward fill: interpolation uses the next measurement (future information), and forward fill makes stale toxin levels persist indefinitely. Exponential decay is past-only and lets a reading fade toward zero. The fill affects only training targets (with `USE_INTERPOLATED_TRAINING`) and the `pn`/`pn_log` features; evaluation targets are always raw measurements.

## Why weekly MAX on ISO weeks

- Monday-anchored ISO weeks (`%G-%V`) give every source the same weekly boundaries.
- MAX keeps the worst-case toxin reading and the bloom peak for PN; a single high sample in a week is what matters for closures.
