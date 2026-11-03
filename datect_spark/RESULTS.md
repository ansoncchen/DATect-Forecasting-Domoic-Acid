# DATect on Spark + Delta Lake + MLflow: results

Open-source Spark, Delta Lake and MLflow on UW's Hyak cluster. This is not Databricks,
and every job ran in Spark local mode on a single node. Each section reports what was
measured, the parity check that ties the Spark output to the existing pipeline, and
the caveats.

## Hardware and versions

| | |
|---|---|
| Nodes | Hyak `ckpt` partition, 1 node per job, 16 cores and 32 GB requested (Slurm) |
| Spark | PySpark 4.0.1, local mode `local[16]`, Java 17 (`coenv/jdk/17.0.8`) |
| Delta Lake | delta-spark 4.0.0 (writes), deltalake 1.6.6 / delta-rs (API reads, no JVM) |
| MLflow | 3.17.2, sqlite tracking and model registry on `/gscratch/stf` |
| Python stack | Python 3.10, numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, xgboost 3.1.3, pyarrow 23.0.0 |
| Storage | Lake and Spark scratch on `/gscratch/scrubbed` (purged after 21 idle days, rebuildable) |

## 1. Distributed re-tune with MLflow tracking (commit e76647bd)

Each of the ensemble's per-site weight and clip settings is post-processing, so one
XGBoost + Random Forest pair per test point serves every configuration. `retune.py
predict` trains those pairs on Spark (one task per real DA measurement, each on its own
pre-anchor data) and `score` applies the 100 weight × clip settings in Spark SQL.

| | |
|---|---|
| Forecasts trained | 1,990 (1,581 tuning points 2015–2021 + 409 holdout points 2022–2023), each an XGBoost + RF pair |
| Wall time, 16 cores | 883 s (675 s + 208 s) |
| Task time | 7,710 s (5,676 s + 2,034 s): 8.7× the wall time |
| MLflow runs | 2,990: 1,990 forecast runs + 1,000 (site, setting) scoring runs |
| Parity | The current configuration rebuilt from the Spark predictions equals the committed chronological predictions exactly (max abs diff 0.0 on 635 + 409 rows) |

Hyak job 42629914. The per-site picks it produced tie a single shared rule on the
holdout, so they were not adopted (`docs/CORRECTED_NUMBERS.md`).

## 2. MODIS imagery into Delta Lake (bronze / silver / gold)

`modis_lake.py` reads the MODIS 8-day composites (chlorophyll-a, Kd490, fluorescence
line height, SST; Aqua, 2003–2024) that are already on Hyak, in place. They are never
re-downloaded from ERDDAP. It builds three layers and checks the result against
the existing OAD pipeline (`oad/scripts/02_build_cube.py` →
`07_cloud_confound.compute_region_valid_fractions` →
`data/processed/oad_cloud_fractions.parquet`, the input of two DATect forecast features).

**Input.** 2,563 NetCDF files totalling 13,904,484,780 bytes (13.90 GB): chla 4.78 GB,
k490 2.57 GB, nflh 2.63 GB, sst 3.92 GB. That is 264 monthly files per channel plus
1,507 per-day files, mostly chlorophyll 2003–2006, that repeat composites already in
the monthly files. Every file shares one 321 × 409 grid (0.025°, 41–49° N,
131–120.8° W); the job checks this from a hash of the coordinates. 29 macOS
AppleDouble stubs (`._chla_*.nc`, 163 B each) sit beside the data. The reference's
`chla_*.nc` glob never matched them, so they are skipped too.

| Layer | Table | Rows | Delta size | Files |
|---|---|---:|---:|---:|
| bronze | `satellite_files` (path, rank in the reference's sort order, bytes, sha256, grid hash) | 2,563 | 0.2 MB | 1 |
| bronze | `satellite_times` (every timestamp of every file, fully cloudy ones included) | 26,417 | 5.1 MB | 2,563 |
| bronze | `satellite_pixels` (non-NaN pixels; partitioned by channel and year) | 1,956,017,408 | 5.85 GB | 2,584 |
| silver | `satellite_times` (one source per channel and time: the first file in sorted order) | 24,668 | | |
| silver | `cube_times` (times present in all four channels) | 4,692 | | |
| silver | `satellite_cube` (one row per time and pixel with any channel; four values + `valid`; by year) | 373,593,583 | 4.06 GB | 407 |
| gold | `oad_region_valid_frac` (valid pixels / pixels in each region's box, 5 regions) | 23,460 | 0.3 MB | 1 |

Deduplication drops 1,749 repeated (channel, time) rows. The four channels have
7,778 / 4,708 / 4,814 / 7,368 unique times, of which 4,692 are present in all four.
299,461,004 cube rows have all four channels valid. After VACUUM the lake takes
12.4 GB on disk (bronze 8.0 GB after compaction, silver 4.4 GB). It peaked at
21 GB before VACUUM.

**Parity** (job 42764314, all exact):
- the gold table equals `oad_cloud_fractions.parquet`: `assert_frame_equal(check_exact=True)`, 23,460 of 23,460 rows, max abs diff 0.0;
- the silver time axis equals `cube.zarr`'s, all 4,692 timestamps;
- five dates (2003-01-05, 2015-03-22, 2018-06-09, 2021-09-13, 2024-12-31) were re-standardized from silver with the cube's stored mean and std. Their mask and all four channels are **bitwise identical** to `cube.zarr` (65k–86k valid pixels per date);
- the reference itself reproduces: re-running `02_build_cube.py` gives the committed parquet, the same time axis and the same mask (job 42763175).

**Timings** (one 16-core node each; the Spark job ran in `local[16]`):

| Step | Wall time |
|---|---:|
| bronze: file metadata + sha256 of all 13.9 GB | 110 s |
| bronze: decode NetCDFs, write 1.96 B pixel rows | 292 s |
| silver: dedup, time axis, pivot (one range shuffle on time) | 699 s |
| gold: region aggregation | 20 s |
| **Spark, raw → gold** | **1,163 s (19.4 min)** |
| Reference: `02_build_cube.py` (dense standardized cube, 5.4 GB zarr, 13.7 GB peak RSS) | 553 s |
| Reference: `compute_region_valid_fractions` | 1.9 s |
| **Reference, raw → valid fractions** | **555 s (9.3 min)** |

On one node the Spark pipeline is about 2× slower than the xarray reference.
The two jobs also do different work: Spark hashes every input byte and writes three
queryable layers, about 2 billion rows in all, while the reference writes one dense
array. Spark's gain here is the lake (SQL over pixels, data skipping, time travel,
lineage from file hash to gold row), not speed. Multi-node runs were not measured.

**OPTIMIZE / Z-ORDER** (target file size 64 MB, 440 s in all):

| | Before | After |
|---|---:|---:|
| bronze pixels: compaction (263 s) | 2,584 files, 5.85 GB | 130 files, 7.92 GB |
| silver cube: `ZORDER BY (lon_idx, lat_idx)` (141 s) | 407 files, 4.06 GB | 59 files, 4.38 GB |
| silver: files (bytes) a region-box query must read, Overall | 407 (100%) | 42 (72%) |
| silver: same, SW Washington / Long Beach box | 407 (100%) | 19 (31%) |
| silver: same, other three boxes | 407 (100%) | 36–41 (63–70%) |
| gold region query, wall (cold, warm) | 9.0 s, 6.1 s | 6.5 s, 6.6 s |

Before Z-ordering, every silver file spans the whole grid because it holds a run of
dates, so no file can be skipped. Z-ordering clusters files spatially, and Delta's
min/max statistics then skip 28–69% of the bytes for a region filter (counted from the
transaction log's per-file statistics). The gold query is still no faster on this
node. It scans all 22 years of the 300 M valid rows, and where its time goes was not
profiled. Compaction cut bronze's file count 20× but made it 35% larger. The
rewritten files compress worse than the originals, each of which held one NetCDF's
rows in time and grid order; the encoding difference was not investigated further.

## 3. Serving precomputed forecasts from Delta Lake

A dashboard forecast (`POST /api/forecast/enhanced`) makes two
`generate_single_forecast` calls, regression and classification. Each retrains the
evaluated models on data up to the anchor date, which takes seconds.
`forecast_serving.py precompute` runs those calls on Spark (one task per site and
date, one thread each) and stores each output unchanged, as lossless JSON, in
`gold/forecasts`. Each row also carries the MLflow model version and a config hash.
The API (`backend/precomputed_forecasts.py`) loads the table with `deltalake`
(delta-rs, no JVM) into an in-memory dict at startup. It answers from the dict when
the key is present and the row's config hash matches the running code, config, data
and library versions; otherwise it computes on demand as before. Responses gain
`source` and `model_version`. Job 42764310 ran all four steps below on one node
(n3244, 16 cores).

**What is precomputed.** All 10,950 (site, week) rows of `final_output.parquet`
(10 sites × 1,095 Mondays, 2003-01-06 to 2023-12-25), model `ensemble` (the
dashboard default), both tasks: 21,900 entries, 0 errors. 220 of them are early
dates with too little training data, stored as `None` exactly as the engine
returns it. The Delta table holds 21,900 rows in 4 files, 16.9 MB.

| Precompute (Spark, `local[16]`) | |
|---|---:|
| Wall time | 7,887 s (2.19 h) |
| Task time | 93,623 s (26.0 h): 11.9× the wall time |
| Mean task time per (site, date), both forecasts, 1 thread | 8.55 s |
| Daily grid instead (76,590 site-days), estimated | 11.4 node-hours |

The daily grid would cost more than the 2-node-hour budget, so the table is weekly.

**MLflow Model Registry.** `datect-forecaster` version 1 is a pyfunc wrapping the
forecast procedure. The model is retrained per request on data up to the anchor, so
it runs inside a DATECT checkout. Its tags record the config hash, git SHA, grid,
Delta table path and version, entry count and wall time. Every table row carries
`model_version = datect-forecaster/1`. It lives in the same sqlite store as the
re-tune's 2,990 runs.

**Parity** (all exact):
- 50 random table entries were compared with `generate_single_forecast` run fresh in
  a separate process with the API's threading (no one-thread pin): 50 of 50 equal in
  every value and type (floats bitwise, `np.float32` feature importances, timestamps);
- the registered pyfunc, loaded from `models:/datect-forecaster/1`, reproduces 3 of 3
  entries;
- end to end, all 20 timed responses are identical to `main`'s on-demand responses
  (sha256 of the body without `source`/`model_version`);
- earlier, Spark workers' single-threaded XGBoost matched the API's multithreaded
  XGBoost on 8 forecasts on the laptop. The 50-entry check above covers this on Hyak.

**Latency of `POST /api/forecast/enhanced`.** The same 20 (site, date) requests
(seed 0, drawn from the weekly rows), one at a time, after one untimed warm-up
request. API on the same node, uvicorn, 16 cores.

| | Before: `main` e665a53c, computed on demand | After: 4a99460c, served from Delta |
|---|---:|---:|
| p50 | 3.283 s | **0.122 s** (27× lower) |
| p95 | 10.628 s | **0.127 s** (84× lower) |
| mean | 4.920 s | 0.122 s |
| max | 16.625 s | 0.128 s |
| Sources | 20 computed | 20 precomputed |

The lookup and JSON decode take about 25 µs. Most of the remaining ~0.12 s is
building the response, mainly the Plotly uncertainty figure (about 53 ms on a laptop;
not profiled on Hyak). API startup was 49 s before and 2.3 s after, but the "before"
process started with cold file caches on GPFS, so that difference is not a
measurement of this change.

**Caveats.**
- Only weekly dates (the Mondays in the data) and the `ensemble` model are served
  from the table. Any other date, or `naive`/`linear`, falls back to computing on
  demand at the old latency. The dashboard's default date (dataset end minus 30 days,
  2023-11-25, a Saturday) is a miss, so a uniformly random calendar date hits about
  one time in seven.
- `linear` cannot be precomputed with exact parity: its regression interval is a
  bootstrap (100 XGBoost + Ridge fits) drawn from the process-global NumPy RNG, so
  its on-demand output itself changes from call to call.
- Any change to `forecasting/`, `config.py`, the tuned JSON, the data or the library
  versions changes the config hash. The table then goes stale and serves nothing
  until `precompute` is re-run. The conformal error band in progress will do this.
- The Cloud Run deployment does not ship the table yet. With
  `DATECT_FORECAST_TABLE` unset the API behaves exactly as before.

## Environment notes (Hyak)

- `/gscratch/scrubbed` (98% full) and `/gscratch/stf` (99%) refused writes
  intermittently with EDQUOT while this ran, and the first jobs died on it. The jobs
  now run from a `git archive` on the node's local disk (`/tmp`, 358 GB, no quota),
  read the MODIS files in place, and copy only their result JSONs out
  (`datect_spark/hyak_stage.sh`). The lake is rebuilt on each run.
- Compute nodes set `http_proxy`, and its `no_proxy` lacks `127.0.0.1`. The latency
  harness therefore talks to the local API with proxy settings ignored.
- No ERDDAP data was downloaded. The tabular features (M4) were not rebuilt on Spark,
  so the "13.9 GB" is the satellite imagery alone.

## Reproduce

```bash
# from any Hyak checkout of this branch; logs outside the full filesystems
L="-o $HOME/datect-spark-logs/%x-%j.out -e $HOME/datect-spark-logs/%x-%j.err"
sbatch $L datect_spark/modis_lake.sbatch                                  # MODIS -> Delta + parity
sbatch $L --export=ALL,STEPS=reference datect_spark/modis_lake.sbatch     # timed reference
sbatch $L --export=ALL,MAIN_SHA=$(git rev-parse origin/main) datect_spark/serving.sbatch
```

Result JSONs: `results/spark_lake/` (MODIS) and `results/serving/` (precompute,
parity, latency before/after).

## Resume bullets (numbers from the runs above)

- Built a PySpark + Delta Lake pipeline (bronze/silver/gold) over 13.9 GB of MODIS
  satellite imagery (2,563 NetCDF files, 2003–2024, 1.96 billion pixel rows) that
  reproduces the existing ocean-feature pipeline exactly, with OPTIMIZE / Z-ORDER
  cutting the data a regional query must read by 28–69%.
- Trained 1,990 leak-free XGBoost + Random Forest forecasting model pairs in parallel
  on Spark (8.7× over serial on 16 cores), tracking 2,990 runs in MLflow, with
  predictions matching the existing evaluation exactly.
- Cut forecast API latency from 3.3 s to 0.12 s at the median (10.6 s to 0.13 s at
  p95) by precomputing 21,900 forecasts on Spark (11.9× parallel speedup) into Delta
  Lake, served through delta-rs with an MLflow-registered model version and
  bit-identical responses.

All three ran on open-source Spark on a university HPC cluster (UW Hyak), one node
per job, not on Databricks.
