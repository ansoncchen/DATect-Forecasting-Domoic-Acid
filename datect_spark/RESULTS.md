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
| joblibspark | 0.6.0: Spark as joblib's backend, so the engine's own parallel calls run on Spark |
| Python stack | Python 3.10, numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, xgboost 3.1.3, pyarrow 23.0.0 |
| Storage | MODIS read in place from `/gscratch/stf`; lake, Spark scratch and API runs on the node's local disk (`/tmp`, see the environment notes); MLflow on `/gscratch/stf` |

## 1. Distributed re-tune with MLflow tracking (commit e76647bd)

Each of the ensemble's per-site weight and clip settings is post-processing, so one
XGBoost + Random Forest pair per test point serves every configuration. `retune.py
predict` trains those pairs on Spark (one task per real DA measurement, each on its own
pre-anchor data) and `score` applies the 100 weight × clip settings in Spark SQL.

| | |
|---|---|
| Forecasts trained | 1,990 (1,581 tuning points 2015–2021 + 409 holdout points 2022–2023), each an XGBoost + RF pair |
| Wall time, 16 cores | 883 s (675 s + 208 s) |
| Task time | 7,710 s (5,676 s + 2,034 s): 8.7× the wall time (summed task time ÷ wall, not a separate serial run) |
| MLflow runs | 2,990: 1,990 forecast runs + 1,000 (site, setting) scoring runs |
| Parity | The current configuration rebuilt from the Spark predictions equals the committed chronological predictions exactly (max abs diff 0.0 on 635 + 409 rows) |

Hyak job 42629914. The per-site picks it produced tie a single shared rule on the
holdout, so they were not adopted (`docs/CORRECTED_NUMBERS.md`). The clip has since
been removed (#18), and `retune.py` now tunes weights only; that run is in section 4.

## 2. MODIS imagery into Delta Lake (bronze / silver / gold)

`modis_lake.py` reads the MODIS 8-day composites (chlorophyll-a, Kd490, fluorescence
line height, SST; Aqua, 2003–2024) that are already on Hyak, in place. They are never
re-downloaded from ERDDAP. It builds three layers and checks the result against
the existing OAD pipeline (`oad/scripts/02_build_cube.py` → a region count over
`cube.zarr`, then in `07_cloud_confound.py`, now `modis_lake.reference_valid_fractions` →
`data/processed/oad_cloud_fractions.parquet`, the input of two DATect forecast
features). Since these runs, the lake is that file's producer (`export` step).

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
about 22 GB (21 GiB) before VACUUM.

**Parity** (job 42764314, all exact):
- the gold table equals `oad_cloud_fractions.parquet`: `assert_frame_equal(check_exact=True)`, 23,460 of 23,460 rows, max abs diff 0.0;
- the silver time axis equals `cube.zarr`'s, all 4,692 timestamps;
- five dates (2003-01-05, 2015-03-22, 2018-06-09, 2021-09-13, 2024-12-31) were re-standardized from silver with the cube's stored mean and std. Their mask and all four channels are **bitwise identical** to `cube.zarr` (65k–86k valid pixels per date);
- the reference itself reproduces: re-running `02_build_cube.py` gives the committed parquet, the same time axis and the same mask (job 42763175).

**Timings** (one 16-core node each; the Spark job ran in `local[16]`):

| Step | Wall time |
|---|---:|
| bronze (438 s): file metadata + sha256 of all 13.9 GB | 110 s |
| bronze: decode NetCDFs, write 1.96 B pixel rows | 292 s |
| bronze: Spark start-up, grid and table statistics | 36 s |
| silver: dedup, time axis, pivot (one range shuffle on time) | 699 s |
| gold: region aggregation and write | 26 s |
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
| silver: files and bytes a region-box query must read, Overall box | 407 files, 4.06 GB | 42 files, 3.16 GB |
| silver: same, SW Washington / Long Beach box | 407 files, 4.06 GB | 19 files, 1.35 GB |
| silver: same, other three boxes | 407 files, 4.06 GB | 36–41 files, 2.74–3.07 GB |
| gold region query, wall (cold, warm) | 9.0 s, 6.1 s | 6.5 s, 6.6 s |

Before Z-ordering, every silver file spans the whole grid because it holds a run of
dates, so no file can be skipped. Z-ordering clusters files spatially, and Delta's
min/max statistics let a region filter skip files. Against the 4.06 GB a query
had to read before, a region box now reads 1.35–3.16 GB (22–67% less). These figures
are counted from the transaction log's per-file statistics, not from measured reads. The gold query is still no faster on this
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
The API (`backend/precomputed_forecasts.py` in these runs; now `backend/precomputed.py`,
see section 4) loads the table with `deltalake`
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
| Task time | 93,623 s (26.0 h): 11.9× the wall time (summed task time ÷ wall) |
| Mean task time per (site, date), both forecasts, 1 thread | 8.55 s |
| Daily grid instead (76,590 site-days), estimated | 11.4 node-hours at a perfect 16×; about 15.3 at the measured 11.9× |

A daily grid would cost far more than the roughly 2-node-hour budget, so the table
is weekly. The weekly run itself took 2.19 hours on one node.

**MLflow Model Registry.** `datect-forecaster` version 1 is a pyfunc wrapping the
forecast procedure. The model is retrained per request on data up to the anchor, so
it runs inside a DATECT checkout. Its tags record the config hash, git SHA, grid,
Delta table version, entry count and wall time. The `delta_table` tag holds the job's
node-local path, which was deleted at job end; the table was copied to
`/gscratch/scrubbed/ac283/datect-lake/lake/gold/forecasts-42764310`. Every table row carries
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
- an ad hoc laptop check (not committed) found Spark workers' single-threaded
  XGBoost identical to the API's multithreaded XGBoost on 8 forecasts. The 50-entry
  check above covers this on Hyak;
- none of the 21,900 stored entries carries a bootstrap-fallback interval. That
  fallback uses the global RNG, and the precompute now refuses to store it.

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

In ad hoc laptop timings (not committed), the lookup and JSON decode took about
25–30 µs and the response's Plotly uncertainty figure about 53 ms. The remaining
~0.12 s on Hyak was not profiled. API startup was 49 s before and 2.3 s after, but the "before"
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
- Any change to `forecasting/`, `config.py`, the tuned JSON, the data bytes or the
  library versions changes the config hash. The table then goes stale and serves
  nothing until `precompute` is re-run; the conformal error band in progress will do
  this. The hash is conservative: setting a `DATECT_*` variable to its default value
  also counts as a change. Stale rows are logged at WARNING and reported on
  `/health`.
- These runs set `DATECT_FORECAST_TABLE` to the table. It is now
  `DATECT_PRECOMPUTED_DIR`, the store directory (section 4), and the image ships the
  store. Unset, every forecast is computed as before. Responses still gain
  `source: "computed"` and `model_version: null`, and `/health` gains
  `precomputed: null`.

## 4. One pipeline per job

After the measurements above, each job was given a single implementation on the
Spark/Delta path. The evaluation itself (chronological, golden, multi-seed) stays on
the joblib engine; Spark is a second executor of the same engine.

**The lake makes the cloud fractions.** `modis_lake.py export` writes
`data/processed/oad_cloud_fractions.parquet` from the gold table. With the committed
writer versions (pandas 2.3.3, pyarrow 23.0.0, the Hyak env) the file is
byte-identical to the committed one (sha256 `7ad73afb…`; job 42774330, where parity
passed again). `oad/scripts/07_cloud_confound.py` reads that file instead of counting
`cube.zarr`'s mask itself. The original counting code lives on unchanged as
`modis_lake.reference_valid_fractions`, used only to re-time the old pipeline.
`02_build_cube.py` stays, because the autoencoder trains on `cube.zarr`.

**One weight tuner.** `retune.py` now tunes only the per-site XGBoost/Random Forest
weights (nothing is clipped since #18), scored on yearly folds 2015–2021 (job
42774505, commit 23ab0058):

| | |
|---|---|
| Forecasts trained | 1,990 (1,581 tuning + 409 holdout points), each an XGBoost + RF pair |
| Wall time, 16 cores | 816 s (624 s + 192 s), against 7,362 s of task time |
| MLflow runs | 1,990 forecast runs + 50 (site, weight) scoring runs |
| Parity | The current weights applied to its raw predictions equal the committed chronological predictions exactly (max abs diff 0.0 on 635 + 409 rows) |
| Holdout 2022–2023 | Current weights R² 0.516 (the headline), MAE 7.44; proposed R² 0.507, MAE 7.39 |

Nothing was adopted: the proposal (`results/tuning_provenance/retune_weights/`)
does worse on the holdout R². The old tuning stack is removed: the clip grid search,
`tune_all.py` (which would have written the clip back into the tuned JSON), its
sweeps, and `validate_tuned_on_holdout.py`. `results/tuning_provenance/` keeps their
outputs. The current weights came from the removed grid search (promoted in
`cdc2a32c`). Its folds differ from these, which is why a re-run can propose other
weights.

**One precompute store.** The file cache (`precompute_cache.py` → `cache/`, read by
`cache_manager.py`, with an optional Redis layer) is replaced by one directory of
Delta tables, `DATECT_PRECOMPUTED_DIR`, read with deltalake (`backend/precomputed.py`):

| Table | Rows | Built by |
|---|---:|---|
| `forecasts` | 21,900 (section 3) | `forecast_serving.py precompute` |
| `retrospective` | 8 slices × 1,054 records | `precompute_store.py retrospective`: `run_retrospective_evaluation` unchanged, with joblib's backend set to Spark (joblibspark), so Spark executes the engine's own per-point forecasts |
| `spectral` | 11 (10 sites + all) | `precompute_store.py visuals`, one Spark task per chart |
| `correlation` | 10 | same |

Every table carries the config hash, which now also covers
`backend/visualizations.py`, the contents of `*_JSON` override files and the CPU
architecture. It no longer covers `DATECT_PARALLEL_BACKEND` or
`DATECT_TEMPORAL_HOLDOUT_CUTOFF`, which never change a stored row.

The gate passed (job 42775261, commit 2401ecac). All **29 of 29** precomputed
endpoint bodies served from the store are byte-identical to the old file cache's,
built by `precompute_cache.py` at the PR #21 head (job 42774205): 8 retrospective
slices, 11 spectral analyses and 10 correlation heatmaps, compared by sha256 without
the fields that say where a body was served from. Build time on one 16-core node:

| | Wall time |
|---|---:|
| Old: `precompute_cache.py`, joblib/loky, everything | 1,800 s |
| New: retrospective on Spark (8 slices, 257–335 s each) | 2,567 s |
| New: spectral + correlation on Spark | 4 s |

Spark as joblib's executor is about 1.4× slower on one node, from per-batch Spark jobs
and pickling the engine with its feature frame. The gain is one store and one build
path, with the same bodies, not speed. The retrospective, spectral and correlation
tables take 1.2 MB.

**Serve live from Delta.**
- The image copies `store/` (synced from `~/datect-store/<commit>/` on Hyak) and sets
  `DATECT_PRECOMPUTED_DIR`.
- It installs the model libraries under `constraints.txt`, the Hyak Spark env's
  versions, so a store built there at the deployed commit matches the image's hash.
- CI's docker job prints the image's hash, builds a two-entry forecasts table inside
  the image and checks that the container serves it and computes another date.
- The dashboard's default date is now the latest weekly row, which is always
  precomputed. Before, it was a Saturday, and `new Date()` shifted it a day in US time
  zones.
- The production table is not rebuilt here. `main`'s config hash changed when #19
  added `forecasting/conformal.py`, and the forecasting rework that follows will change
  it again. The store is rebuilt once after that lands. Until then a stale table only
  falls back to computing on demand.

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
sbatch $L datect_spark/retune.sbatch                                      # weight tuner (proposes only)
sbatch $L --export=ALL,STEPS="forecasts parity save" datect_spark/store.sbatch
sbatch $L --export=ALL,STEPS="retrospective save visuals save" datect_spark/store.sbatch
```

Result JSONs: `results/spark_lake/` (MODIS, export), `results/serving/` (precompute,
parity, latency before/after), `results/tuning_provenance/retune_weights/` (tuner) and
`results/store/` (store build, endpoint-body gate).

## Resume bullets (numbers from the runs above)

- Built a PySpark + Delta Lake pipeline (bronze/silver/gold) over 13.9 GB of MODIS
  satellite imagery (2,563 NetCDF files, 2003–2024, 1.96 billion pixel rows). Its
  cloud-cover features match the existing pipeline exactly, and OPTIMIZE / Z-ORDER
  cuts the bytes a regional query must read by 22–67%.
- Trained 1,990 leak-free XGBoost + Random Forest forecasting model pairs in parallel
  on Spark (8.7× task-time-to-wall-time on 16 cores), tracking 2,990 runs in MLflow,
  with predictions matching the existing evaluation exactly.
- Cut forecast API latency on precomputed (weekly) dates from 3.3 s to 0.12 s at the
  median (10.6 s to 0.13 s at p95) by precomputing 21,900 forecasts on Spark (11.9×
  task-time-to-wall-time) into Delta Lake, served through delta-rs with an
  MLflow-registered model version and responses identical to on-demand output.

All three ran on open-source Spark on a university HPC cluster (UW Hyak), one node
per job, not on Databricks.
