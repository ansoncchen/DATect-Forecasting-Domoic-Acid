# Ocean Anomaly Detection (OAD): Hyak-first workflow

Unsupervised convolutional autoencoder over 4-channel MODIS Aqua composites
(chl-a, Kd490, nFLH, SST) producing a per-region ocean-state anomaly index
for the U.S. Pacific Northwest. Compared against single-channel z-score,
multivariate climatological z-score, and PCA reconstruction at matched bottleneck.

Two model variants live side by side:
- **2D ConvAE**: single-frame snapshot anomalies (`src/model.py`)
- **3D ConvAE3D**: short temporal context (T=4 frames lookback, `src/model3d.py`)

See [`OAD_STORYLINE.md`](OAD_STORYLINE.md) for the design narrative and
[`RESULTS.md`](RESULTS.md) for validated numbers. How the scores feed DATect is in
[`../docs/OAD_INTEGRATION_RESULTS.md`](../docs/OAD_INTEGRATION_RESULTS.md).

## Defaults

- **Resolution**: stride-2 (0.025°); full resolution is too slow to download
- **Cadence**: all daily rolling 8-day composites (~364/yr/channel) for dense training signal
- **Anchor-only mode**: opt-in via `--anchor-only` for ~46 frames/yr instead

## Where things live on Hyak

| What | Path |
|---|---|
| OAD code + data + models + outputs | `/gscratch/stf/ac283/oad_current`, a symlink to `/gscratch/stf/ac283/DATect-Forecasting-Domoic-Acid/ocean anomaly detection` |
| Python (torch, xarray, zarr, CUDA) | conda env `/gscratch/stf/ac283/envs/datect_scratch` |

Every sbatch file in `hyak/` defaults `OAD_ROOT` to `/gscratch/stf/ac283/oad_current`,
`cd`s there, and runs `/gscratch/stf/ac283/envs/datect_scratch/bin/python`. Jobs run
`scripts/*.py` relative to `OAD_ROOT`, and `config.py` resolves `data/`, `models/`
and `outputs/` relative to itself, so `OAD_ROOT` must hold the code and the data
together. Override it per job with `sbatch --export=ALL,OAD_ROOT=/path/to/oad ...`.

The symlink targets the pre-rename directory name. If that Hyak checkout is updated
past the commit that renamed `ocean anomaly detection/` to `oad/`, git moves the
tracked code to `oad/` but leaves the untracked `data/`, `models/` and `outputs/`
behind; move those into `oad/` and repoint the symlink before submitting jobs again.

## Hyak workflow

Everything heavy runs on Hyak; a laptop is for editing code and viewing figures.

### Setup

```bash
ssh klone-login
ls /gscratch/stf/ac283/oad_current/{scripts,src,data,models}   # code and data are already there
```

On a fresh install: clone this repo under `/gscratch/stf/<netid>/`, create a conda
env with the parent `requirements.txt` plus `oad/requirements.txt`, pass
`OAD_ROOT=<clone>/oad` to each job, and edit the hardcoded `PYBIN` line in the
sbatch files to point at your env.

### Submit jobs

Run from `OAD_ROOT` (the sbatch files live in its `hyak/` directory):

```bash
cd /gscratch/stf/ac283/oad_current

# End to end: download → cube → train (2D + 3D sweeps) → infer → evaluate
sbatch hyak/full_pipeline.sbatch
sbatch --export=ALL,SKIP_DOWNLOAD=1 hyak/full_pipeline.sbatch          # data already present
sbatch --export=ALL,START=2003-01-01,END=2010-12-31 hyak/full_pipeline.sbatch

# Or in pieces:
sbatch hyak/download.sbatch                                  # 1. ERDDAP download, then cube build
                                                             #    (SKIP_CUBE=1 for parallel per-channel jobs)
sbatch --dependency=afterok:<id> hyak/verify_files.sbatch    #    gate: every channel complete
sbatch hyak/build_cube.sbatch                                # 2. standalone data/cube.zarr build
sbatch --export=ALL,MODE=temporal_sweep hyak/train.sbatch    # 3. training (MODE: see below)
sbatch --export=ALL,MASK_RATIO=0.70 hyak/mae_train.sbatch    #    or masked-AE training at one ratio
sbatch hyak/infer.sbatch                                     # 4. scores for all methods
sbatch hyak/sanity_checks.sbatch                             # 5. cloud confound + multi-step lead sweep
```

### Pull figures back

```bash
# From a laptop (generated figures are not tracked; the papers keep copies of the ones they use)
scp -r klone-login:/gscratch/stf/ac283/oad_current/outputs/figures oad/outputs/figures
scp klone-login:/gscratch/stf/ac283/oad_current/outputs/figures/RESULTS_summary.txt oad/outputs/
```

### Feeding DATect

DATect reads one checkpoint's regional scores from the tracked
`data/processed/oad_scores.parquet`. From the repo root:

```bash
python scripts/data/export_oad_features.py ae_3d_l32_c4_t4_s42_mae070 --scores-dir <OAD_ROOT>/outputs/scores
python scripts/data/rebuild_final_output.py --recompute oad
```

## Storage

The 22-year stride-2 dataset measured in October 2026: raw NetCDFs (`data/raw/`)
about 14 GB, `data/cube.zarr` about 5 GB on disk. The cube is Zarr v2 with one-year
chunks; training reads the full array into RAM on a node with ≥ 64 GB.

## Local dev (code only)

Needs an environment with the parent `requirements.txt` and `oad/requirements.txt`
(the project's `.venv` has no torch or zarr). Run from `oad/`; outputs go to
`oad/models/` and `oad/outputs/`.

```bash
python scripts/make_synthetic_cube.py --out data/cube_synth.zarr --n-times 120
python scripts/03_train_ae.py --cube data/cube_synth.zarr --latent 32 --epochs 5 --debug
python scripts/04_run_inference.py --cube data/cube_synth.zarr --all-ae --baselines-only --baselines-3d
python scripts/05_evaluate.py
python scripts/06_summarize_results.py
```

On a Mac the 3D ConvAE falls back to CPU (MPS lacks ConvTranspose3D): slow but functional.

## File map

```
oad/
├── README.md                       # this file
├── RESULTS.md                      # validated numbers
├── OAD_STORYLINE.md                # narrative walkthrough
├── AGENTS.md                       # pointer for coding agents
├── config.py                       # hyperparameters, paths, channels
├── requirements.txt                # torch, zarr, shapely, matplotlib, dask
├── hyak/
│   ├── full_pipeline.sbatch        # download → cube → train → infer → evaluate
│   ├── download.sbatch             # ERDDAP download
│   ├── verify_files.sbatch         # gate: each channel covers enough months
│   ├── build_cube.sbatch           # standalone cube build
│   ├── train.sbatch                # training (MODE override)
│   ├── mae_train.sbatch            # masked-AE training (2D + 3D) at one MASK_RATIO
│   ├── infer.sbatch                # inference for all methods
│   └── sanity_checks.sbatch        # cloud confound + multi-step forecastability
├── src/
│   ├── regions.py                  # 5 region polygons (Overall + 4 subregions)
│   ├── model.py                    # 2D ConvAE + masked_mse
│   ├── model3d.py                  # 3D ConvAE3D + masked_mse_3d
│   ├── dataset_temporal.py         # temporal patch sampler
│   ├── train.py                    # train() + train_temporal()
│   ├── infer.py                    # 2D tiled inference
│   ├── infer3d.py                  # 3D tiled inference (anchor-frame error)
│   ├── baselines.py                # B1, B2, B3, B3T
│   └── evaluate.py                 # E1, E2, E4, E5 + bootstrap CI
├── scripts/
│   ├── 01_download.py              # ERDDAP downloader (daily default, --anchor-only opt-in)
│   ├── 02_build_cube.py            # standardize → cube.zarr
│   ├── 03_train_ae.py              # --temporal toggles 2D/3D; --mask-ratio for masked AE
│   ├── 04_run_inference.py         # dispatches 2D/3D from the checkpoint
│   ├── 05_evaluate.py              # evaluation figures
│   ├── 06_summarize_results.py     # numeric summary
│   ├── 07_cloud_confound.py        # corr(score, valid-pixel fraction)
│   ├── 08_multistep_forecast.py    # forecastability at lead ∈ {1, 7, 14, 28, 56} days
│   ├── plot_annual_cycle.py        # score annual cycle stacked by year
│   ├── make_paper_figures_mae070.py  # regenerates the mae070 paper figures
│   ├── make_poster_assets.py       # poster schematics
│   └── make_synthetic_cube.py      # synthetic cube for unit testing
├── data/                           # (not tracked) raw NetCDFs + cube.zarr
├── models/                         # (not tracked) AE checkpoints
└── outputs/                        # scores + figures (not tracked); poster_assets/ is tracked
```

## Agent / workspace notes

Conventions and gotchas from prior runs; read before modifying the pipeline.

### Conventions

- **Run everything heavy on Hyak**; the laptop is for editing code and viewing figures.
- **Keep the defaults**: stride-2 (0.025°), all daily rolling 8-day composites.
  `--anchor-only` is opt-in.
- **5 regions** in `src/regions.py`: `OVERALL_COAST_REGION` (bbox envelope, listed first)
  + 4 `SUBREGIONS` (Olympic Coast WA, SW Washington/Long Beach, Central Oregon, Southern OR/N CA).
  `build_region_masks` and `aggregate_to_regions` consume `REGIONS` in that order.
- **Run order**: `01_download.py` → `02_build_cube.py` → `03_train_ae.py` →
  `04_run_inference.py` → `05_evaluate.py`. Re-run inference after region or
  checkpoint changes (`--all-ae` picks up every checkpoint).

### Gotchas

- **`scripts/01_download.py --per-year` actually does per-MONTH chunks**, not
  per-year. A full-year request times out server-side at ERDDAP. The flag name
  is kept for backward compatibility. Pattern: `{chan}_{YYYY}_{MM}.nc`.
- **Cube builder glob picks up BOTH** per-day (`{chan}_{YYYYMMDD}.nc`) and
  per-month (`{chan}_{YYYY}_{MM}.nc`) filename patterns; safe to mix.
- **HTTP 429 (rate limit) retry** is built into the downloader with exponential
  backoff up to 5 min. ERDDAP responsiveness is bursty, especially during
  US business hours. 8 workers is the safe default.
- **MODIS 8-day composites are CENTERED on the labeled date** (`long_name =
  "Centered Time"`). A score at date *t* contains ~3-4 days of "future" data
  relative to *t*. Any downstream consumer must lag by ≥ 4 days beyond its own
  anchor convention. DATect lags by 5 days (`test_date − 12` at the latest);
  see `../docs/OAD_INTEGRATION_RESULTS.md`.
- **MPS doesn't support `ConvTranspose3D`**, so on Mac the 3D ConvAE falls back
  to CPU automatically (`_select_device_for_3d`). CUDA on Hyak handles it fine.
- **Coastal patch bias**: `config.TRAIN_COASTAL_PATCH_MIN_OVERLAP=0.5` makes
  training patches preferentially come from the coastal bbox. `--full-domain-patches`
  disables it for ablation. Inference always uses full-grid tiling.
- **Phase C MAE convention** (`src/train.py`): `mask_ratio > 0` enables random
  per-pixel hiding (broadcasts over channels). Loss = `masked_mse(pred, target,
  valid_mask * hidden_mask)`. Val loss uses the same protocol so train/val are
  directly comparable. Inference is unchanged.
- **Checkpoint filename suffix** encodes mask ratio: `ae_{2d|3d}_l{L}_c{C}_..._mae{NNN}.pt`
  where NNN = `round(mask_ratio * 100)`. The inference script auto-includes
  `_maeNNN` in the method name.

### Headline result (validated, 22-yr cube)

`AE_3d_l32_t4_mae070` (3D ConvAE3D with Phase C MAE training at 70% pixel hiding)
is the best AE method in SW Washington / Long Beach: **R²=+0.87, CIΔ vs matched-k
PCA = [+0.92, +1.05]**, an entirely positive 95% bootstrap CI. Across regions,
the optimal mask ratio is 40–70% for 3D and 50% for 2D. See `RESULTS.md`.

**Strongest defensive finding** (multi-step decay): at lead=7 days,
`AE_3d_l32_t4_mae070` is the **only method** with positive R² in every PNW
region (0.10–0.26). Climatology baselines B1/B2 collapse to ≤ 0 (B2 goes
*negative*), and matched-k PCA was already at 0. The 1-day-ahead
headline R²=0.87 is partly inflated by 8-day composite overlap.

**Cloud-cover confound**: r ≈ +0.4–0.5 with in-region valid-pixel fraction
(~24% of variance). The AE is less confounded than PCA (+0.75) but more than
climatology. `mae070` still wins overall because the cleaner-cloud `mae050`
variant has substantially worse multi-step skill (R²=−0.17 vs +0.19 at lead=7
in Olympic Coast). This is why DATect uses `mae070`.

See "SANITY-CHECK CAVEATS" at the top of `RESULTS.md`.

### Hyak sbatch templates

| File | Partition | Purpose |
|---|---|---|
| `download.sbatch` | compute | ERDDAP download. `START`, `END`, `CHANNELS`, `PER_YEAR=1`, `ANCHOR_ONLY=1`, `SKIP_CUBE=1` env knobs. |
| `verify_files.sbatch` | ckpt | Gate: exits 0 iff all 4 channels cover ≥ `MIN_MONTHS` (default 258 of 264) distinct months. Use as an `afterok` dependency. |
| `build_cube.sbatch` | ckpt | Standalone cube build; `CUBE_ARGS` passes flags to `02_build_cube.py`. |
| `train.sbatch` | gpu-2080ti | Training. `MODE` ∈ {snapshot, temporal, snapshot_sweep, temporal_sweep, snapshot_ablate, temporal_ablate, smoke}. |
| `mae_train.sbatch` | ckpt (rtx6k) | **Phase C MAE training only.** Trains 2D + 3D at the given `MASK_RATIO`. Chain several with `--dependency=afterany:PREV`. |
| `infer.sbatch` | gpu-2080ti | Inference. `FLAGS=` overrides the default `--all-ae --baselines-only --baselines-3d`. |
| `full_pipeline.sbatch` | gpu-2080ti | End to end (download → cube → train → infer → eval). |
| `sanity_checks.sbatch` | ckpt | `07_cloud_confound.py` + `08_multistep_forecast.py`. |
