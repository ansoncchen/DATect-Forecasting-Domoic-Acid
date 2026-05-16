# Ocean anomaly detection — workflow

Unsupervised convolutional autoencoder over 4-channel MODIS Aqua composites
(chl-a, Kd490, nflh, SST) producing a per-region ocean-state anomaly index
for the U.S. Pacific Northwest. Compared against single-channel z-score,
multivariate climatological z-score, and PCA reconstruction at matched bottleneck.

Two model variants live side-by-side:
- **2D ConvAE** — single-frame snapshot anomalies (`src/model.py`)
- **3D ConvAE3D** — short temporal context (T=4 frames lookback, `src/model3d.py`)

See `IMPLEMENTATION_PLAN.md` for design decisions, current status, and locked-in
architectural choices.

## End-to-end workflow

```bash
# ---------- LOCAL (downloads only) ----------
# Smoke test (single year, single channel)
python scripts/01_download.py --channels chla --start 2010-01-01 --end 2010-12-31

# Full download (all 4 channels, full date range, ~5 hr at stride=2 / 0.025°)
python scripts/01_download.py

# Build standardized cube (deduplicates ERDDAP's daily duplicates)
python scripts/02_build_cube.py

# ---------- HYAK (training) ----------
# scp the cube to Hyak first:
#   scp -r data/cube.zarr klone:/gscratch/stf/ac283/oad/data/

# Smoke-test 2D + 3D on Hyak GPU
python scripts/03_train_ae.py --latent 32 --epochs 5 --debug
python scripts/03_train_ae.py --temporal --latent 32 --epochs 5 --debug

# Full sweeps
python scripts/03_train_ae.py --sweep                  # 2D bottleneck sweep
python scripts/03_train_ae.py --temporal --sweep       # 3D bottleneck sweep
python scripts/03_train_ae.py --ablate-channels        # 2D channel ablations
python scripts/03_train_ae.py --temporal --ablate-channels  # 3D channel ablations

# Or submit via sbatch (uses hyak/train.sbatch — edit accordingly)
sbatch hyak/train.sbatch

# ---------- HYAK or LOCAL (inference) ----------
# Run baselines (B1, B2, B3) + temporal baselines (B3T) + all AE checkpoints
python scripts/04_run_inference.py --all-ae --baselines-only --baselines-3d

# ---------- LOCAL (evaluation) ----------
# scp models + outputs/scores back, then:
python scripts/05_evaluate.py
# → outputs/figures/E1, E2, E4, E5, sanity_drift_*.png
```

## Hyak setup

1. **Install env**:
   ```bash
   ssh klone
   cd /gscratch/stf/ac283/
   mkdir -p oad && cd oad
   # copy this folder (without data/ models/ outputs/) here
   module load cuda/12.4   # or whichever CUDA your node has
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **scp cube**:
   ```bash
   # from local laptop
   scp -r "ocean anomaly detection/data/cube.zarr" \
          klone:/gscratch/stf/ac283/oad/data/
   ```
3. **Train**: see `hyak/train.sbatch`.
4. **scp results back**:
   ```bash
   # from local laptop
   scp -r klone:/gscratch/stf/ac283/oad/models ./models
   scp -r klone:/gscratch/stf/ac283/oad/outputs/scores ./outputs/scores
   ```

## Verification ladder (do these in order)

1. **Imports + model forward** (≤1 min, no data needed):
   ```bash
   python -c "from src.model import ConvAE; from src.model3d import ConvAE3D; \
              import torch; \
              print(ConvAE()(torch.randn(2,4,64,64)).shape); \
              print(ConvAE3D()(torch.randn(2,4,4,64,64)).shape)"
   ```
2. **One-year cube** (`scripts/02_build_cube.py --year 2010`): opens with `xr.open_zarr`, shape `(time, 4, H, W)`.
3. **Debug train** (`scripts/03_train_ae.py --debug --epochs 5`): loss decreases monotonically over 5 epochs.
4. **Debug infer** (`scripts/04_run_inference.py --checkpoint models/ae_*.pt`): produces parquet with one row per (date, region).
5. **Evaluate** (`scripts/05_evaluate.py`): all E1/E2/E4/E5 figures land in `outputs/figures/`.

## File map

```
ocean anomaly detection/
├── README.md                       # this file
├── IMPLEMENTATION_PLAN.md          # design decisions + status
├── config.py                       # all hyperparams, paths, channels
├── requirements.txt                # torch, zarr, shapely, matplotlib
├── hyak/
│   └── train.sbatch                # SLURM batch script template
├── src/
│   ├── regions.py                  # 5 region polygons (Overall + 4 subregions)
│   ├── infer.py                    # 2D tiled inference
│   ├── infer3d.py                  # 3D tiled inference (anchor-frame error)
│   ├── model.py                    # 2D ConvAE + masked_mse
│   ├── model3d.py                  # 3D ConvAE3D + masked_mse_3d
│   ├── dataset_temporal.py         # Temporal patch sampler with fallback counter
│   ├── train.py                    # train() + train_temporal()
│   ├── baselines.py                # B1, B2, B3, B3T (Temporal PCA)
│   └── evaluate.py                 # E1, E2, E4, E5 + bootstrap CI
├── scripts/
│   ├── 01_download.py              # ERDDAP downloader with unique-date dedup
│   ├── 02_build_cube.py            # Standardize → cube.zarr
│   ├── 03_train_ae.py              # Single run / --sweep / --ablate-channels
│   ├── 04_run_inference.py         # Auto-dispatches 2D/3D from checkpoint variant
│   └── 05_evaluate.py              # All evaluation figures
├── data/                           # (gitignored) raw NetCDFs + cube.zarr
├── models/                         # (gitignored) AE checkpoints
└── outputs/
    ├── scores/                     # (gitignored) one parquet per method
    └── figures/                    # generated by 05_evaluate.py
```
