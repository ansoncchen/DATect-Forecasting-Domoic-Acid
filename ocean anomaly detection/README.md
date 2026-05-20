# Ocean anomaly detection — Hyak-first workflow

Unsupervised convolutional autoencoder over 4-channel MODIS Aqua composites
(chl-a, Kd490, nflh, SST) producing a per-region ocean-state anomaly index
for the U.S. Pacific Northwest. Compared against single-channel z-score,
multivariate climatological z-score, and PCA reconstruction at matched bottleneck.

Two model variants live side-by-side:
- **2D ConvAE** — single-frame snapshot anomalies (`src/model.py`)
- **3D ConvAE3D** — short temporal context (T=4 frames lookback, `src/model3d.py`)

See `IMPLEMENTATION_PLAN.md` for design and `RESULTS.md` for validated numbers.

## Defaults (matching original ocean-anomaly branch)

- **Resolution**: stride-2 (0.025°) — full resolution is **too slow**; stride-2 gets ~30× faster downloads
- **Cadence**: all daily rolling 8-day composites (~364/yr/channel) for dense training signal
- **Anchor-only mode**: opt-in via `--anchor-only` if you want ~46 frames/yr instead

## Hyak-first workflow

Everything runs on Hyak. Local laptops are only for editing code and viewing figures.

### One-time setup on Hyak

```bash
ssh klone.hyak.uw.edu
cd /gscratch/stf/ac283
git clone https://github.com/ansoncchen/DATect-Forecasting-Domoic-Acid.git oad-repo
cd oad-repo
git checkout ocean-anomaly-v2
cp -r "ocean anomaly detection" /gscratch/stf/ac283/oad
cd /gscratch/stf/ac283/oad
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir -p logs data models outputs/scores outputs/figures
```

### Submit jobs

```bash
# Full pipeline in one shot — download → cube → train (sweeps) → infer → evaluate
sbatch hyak/full_pipeline.sbatch

# Or pieces:
sbatch hyak/download.sbatch                                  # 1. Download + build cube
sbatch --export=ALL,MODE=snapshot_sweep hyak/train.sbatch    # 2a. 2D sweep
sbatch --export=ALL,MODE=temporal_sweep hyak/train.sbatch    # 2b. 3D sweep
sbatch hyak/infer.sbatch                                     # 3. Inference (all methods)

# Custom date range (downloader + full pipeline both accept):
sbatch --export=ALL,START=2003-01-01,END=2010-12-31 hyak/full_pipeline.sbatch

# Skip download if data already there:
sbatch --export=ALL,SKIP_DOWNLOAD=1 hyak/full_pipeline.sbatch
```

### Pull figures + numeric summary back

```bash
# From local laptop
scp -r klone:/gscratch/stf/ac283/oad/outputs/figures \
       "ocean anomaly detection/outputs/figures_hyak"
scp klone:/gscratch/stf/ac283/oad/outputs/figures/RESULTS_summary.txt \
       "ocean anomaly detection/outputs/"
```

## Storage estimates (stride-2, 4 channels, daily cadence)

| Scale | Files | Cube size |
|---|---|---|
| 1 year   | ~364 × 4 = 1.5k | ~3 GB |
| 5 years  | ~7.3k | ~15 GB |
| 22 years | ~32k  | ~67 GB |

All fits on `/gscratch/stf/`. Cube is stored as Zarr v2 with one-year chunks; training reads the full array into RAM on a node with ≥64GB (standard on Hyak GPU partitions).

## Local dev (laptop) — code only

Only needed for editing the code or testing on the synthetic cube:

```bash
# Synthetic cube for unit testing (no downloads)
python scripts/make_synthetic_cube.py --out data/cube_synth.zarr --n-times 120
python scripts/03_train_ae.py --cube data/cube_synth.zarr --latent 32 --epochs 5 --debug
python scripts/04_run_inference.py --cube data/cube_synth.zarr --all-ae --baselines-only --baselines-3d
python scripts/05_evaluate.py
python scripts/06_summarize_results.py
```

Note: on Mac, the 3D ConvAE falls back to CPU (MPS lacks ConvTranspose3D). Slow but functional for debugging.

## File map

```
ocean anomaly detection/
├── README.md                       # this file
├── IMPLEMENTATION_PLAN.md          # design + status
├── RESULTS.md                      # validation numbers
├── config.py                       # all hyperparams, paths, channels
├── requirements.txt                # torch, zarr, shapely, matplotlib
├── hyak/
│   ├── download.sbatch             # Download all channels + build cube
│   ├── full_pipeline.sbatch        # End-to-end pipeline (recommended entry point)
│   ├── train.sbatch                # Standalone training (MODE override)
│   └── infer.sbatch                # Standalone inference
├── src/
│   ├── regions.py                  # 5 region polygons (Overall + 4 subregions)
│   ├── infer.py                    # 2D tiled inference
│   ├── infer3d.py                  # 3D tiled inference (anchor-frame error)
│   ├── model.py                    # 2D ConvAE + masked_mse
│   ├── model3d.py                  # 3D ConvAE3D + masked_mse_3d
│   ├── dataset_temporal.py         # Temporal patch sampler with fallback counter
│   ├── train.py                    # train() + train_temporal()
│   ├── baselines.py                # B1, B2, B3, B3T
│   └── evaluate.py                 # E1, E2, E4, E5 + bootstrap CI
├── scripts/
│   ├── 01_download.py              # ERDDAP downloader (daily default, --anchor-only opt-in)
│   ├── 02_build_cube.py            # Standardize → cube.zarr
│   ├── 03_train_ae.py              # --temporal flag toggles 2D/3D
│   ├── 04_run_inference.py         # Auto-dispatches 2D/3D from ckpt
│   ├── 05_evaluate.py              # All evaluation figures
│   ├── 06_summarize_results.py     # Numeric summary
│   └── make_synthetic_cube.py      # Synthetic cube for unit testing
├── data/                           # (gitignored) raw NetCDFs + cube.zarr
├── models/                         # (gitignored) AE checkpoints
└── outputs/                        # (mostly gitignored) scores + figures
```

## Agent / workspace notes

Conventions and gotchas that have accumulated from prior runs — read before
modifying the pipeline.

### Conventions

- **Use `python3`** (avoid typo `pytho`).
- **Run everything on Hyak** (`/gscratch/stf/ac283/...`); local laptop is for editing
  code and viewing figures only.
- **Match the original ocean-anomaly branch defaults**: stride-2 (0.025°), keep
  ALL daily rolling 8-day composites (not just 8-day anchors) for denser training
  signal. `--anchor-only` is an opt-in flag, not the default.
- **5 regions** in `src/regions.py`: `OVERALL_COAST_REGION` (bbox envelope, listed first)
  + 4 `SUBREGIONS` (Olympic Coast WA, SW Washington/Long Beach, Central Oregon, Southern OR/N CA).
  `build_region_masks` and `aggregate_to_regions` consume `REGIONS` in that order.
- **Run order**: `01_download.py` → `02_build_cube.py` → `03_train_ae.py` →
  `04_run_inference.py` → `05_evaluate.py`. Re-run inference after region or
  checkpoint changes (it auto-picks up everything via `--all-ae`).

### Gotchas

- **`scripts/01_download.py --per-year` actually does per-MONTH chunks**, not
  per-year. A full-year request times out server-side at ERDDAP. The flag name
  is kept for backward compatibility. Pattern: `{chan}_{YYYY}_{MM}.nc`.
- **Cube builder glob picks up BOTH** per-day (`{chan}_{YYYYMMDD}.nc`) and
  per-month (`{chan}_{YYYY}_{MM}.nc`) filename patterns — safe to mix.
- **HTTP 429 (rate limit) retry** is built into the downloader with exponential
  backoff up to 5 min. Real ERDDAP responsiveness is bursty, especially during
  US business hours.
- **MODIS 8-day composites are CENTERED on the labeled date** (`long_name =
  "Centered Time"`). A score at date *t* contains ~3-4 days of "future" data
  relative to *t*. Any downstream consumer (e.g. integrating scores into
  DATect) must lag by ≥4 days beyond their own anchor convention to avoid leakage.
  See `RESULTS.md` for the integration pattern (`test_date − 12`).
- **MPS doesn't support `ConvTranspose3D`**, so on Mac the 3D ConvAE falls back
  to CPU automatically (`_select_device_for_3d`). Slow but functional. CUDA on
  Hyak handles it fine.
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
PCA = [+0.92, +1.05]** — entirely positive 95% bootstrap CI. Across regions,
the optimal mask ratio is 40–70% for 3D and 50% for 2D. See `RESULTS.md`.

### Hyak sbatch templates

| File | Purpose |
|---|---|
| `download.sbatch` | ERDDAP download. `PER_YEAR=1`, `ANCHOR_ONLY=1`, `SKIP_CUBE=1` env knobs. |
| `build_cube.sbatch` | Standalone cube build. |
| `train.sbatch` | Vanilla training. `MODE` ∈ {snapshot, temporal, snapshot_sweep, temporal_sweep, snapshot_ablate, temporal_ablate, smoke}. |
| `mae_train.sbatch` | **Phase C MAE training only.** Trains BOTH 2D + 3D at given `MASK_RATIO`. Chain multiple with `--dependency=afterany:PREV`. |
| `infer.sbatch` | Inference. `FLAGS=` env var; pass `--pca-k 4 16 32 64 128 --all-ae --baselines-only --baselines-3d`. |
| `full_pipeline.sbatch` | End-to-end (download → cube → train → infer → eval). |
| `verify_files.sbatch` | Gate: exits 0 iff all 4 channels have ≥260 distinct months covered. Use as `afterok` dependency. |
