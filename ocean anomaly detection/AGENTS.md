## Learned User Preferences

- Use `python3` (avoid typo `pytho`).
- Run everything on Hyak; local laptop is for editing and viewing figures only. SSH ControlMaster `klone-login` is set up but expires periodically — user re-auths from their terminal when needed.
- Match the original ocean-anomaly branch defaults: stride-2 (0.025°), keep ALL daily rolling 8-day composites (not just 8-day anchors) for denser training signal.

## Learned Workspace Facts

- `src/regions.py`: Four named `SUBREGIONS` bands; `OVERALL_COAST_REGION` is the lat/lon bbox envelope; `REGIONS = [OVERALL_COAST_REGION] + SUBREGIONS` with overall first. `build_region_masks` and `aggregate_to_regions` consume `REGIONS`. `overall_coastal_mask()` defines geography for AE and PCA patch sampling.
- `config.TRAIN_COASTAL_PATCH_MIN_OVERLAP`: Coastal patches biased to the overall coastal bbox via `train(coastal_patch_min_overlap=...)` and PCA `fit(..., coastal_patch_min_overlap)`. `scripts/03_train_ae.py --full-domain-patches` sets overlap to 0. Inference uses full-grid tiling.
- Run order: `scripts/01_download.py` → `02_build_cube.py` → `03_train_ae.py` → `04_run_inference.py` → `05_evaluate.py`. Re-run inference after region changes or new checkpoints.
- `scripts/01_download.py`: supports per-frame default and `--per-year` (which actually does per-MONTH chunks via `_erddap_url_range`, because a full-year request times out server-side at ERDDAP). Also `--anchor-only` to subsample to native 8-day anchor cadence. Has HTTP 429 retry-with-backoff up to 5 min.
- Raw `data/raw/{chan}/`: per-month files `{chan}_{YYYY}_{MM}.nc` from `--per-year` mode, or per-day `{chan}_{YYYYMMDD}.nc` from default mode. The cube builder's glob picks up both.
- **Phase C MAE training (`src/train.py`)**: `train()` and `train_temporal()` accept `mask_ratio` param. When > 0: during each training step, random per-pixel hidden mask (broadcasts over channels) is generated, input is zeroed at hidden positions, loss is computed only on pixels that were originally valid AND hidden. Val loss uses the same protocol so train/val are directly comparable. Inference is unchanged — at inference time the model gets the full observed input. Checkpoint stores `mask_ratio` so `04_run_inference.py` can auto-name the method as `AE_2d_l32_mae030` etc.
- **MODIS 8-day composites on ERDDAP are CENTERED on the labeled date** (`long_name = "Centered Time"`). A score at date *t* contains ~3-4 days of "future" data relative to *t*. Any consumer that wants leak-free use must lag the score by ≥4 more days beyond their own anchor convention.
- **Headline result** (validated on full 22-year cube, 100-epoch training): `AE_3d_l32_t4_mae030` (Phase C MAE @ 0.30 on the 3D ConvAE) is the best AE method in every region. In SW Washington / Long Beach it statistically beats matched-k temporal PCA at 95% (bootstrap CIΔ entirely positive [+0.758, +0.885]). In most other regions climatology baselines (B1, B2) still win pure forecastability because 22-year daily climatology is hard to beat. See `RESULTS.md` for full per-region tables.
- **Hyak sbatch templates** (`hyak/`):
  - `download.sbatch` — ERDDAP download with `PER_YEAR=1`, `ANCHOR_ONLY=1`, `SKIP_CUBE=1` env knobs
  - `build_cube.sbatch` — standalone cube build
  - `train.sbatch` — vanilla training (MODE = snapshot/temporal/snapshot_sweep/temporal_sweep/snapshot_ablate/temporal_ablate/smoke)
  - `mae_train.sbatch` — Phase C MAE training only (training, no inference). Trains BOTH 2D + 3D ConvAE at given `MASK_RATIO`. Chain multiple together with `--dependency=afterany:PREV`.
  - `infer.sbatch` — inference (`FLAGS=` env var; pass `--pca-k 4 16 32 64 128 --all-ae --baselines-only --baselines-3d`)
  - `full_pipeline.sbatch` — end-to-end (download → cube → train → infer → eval)
  - `verify_files.sbatch` — gate job that exits 0 only if all 4 channels have ≥260 distinct months covered. Use as `afterok` dependency.
- **Trained checkpoint exports for downstream use**: best model is at `models/ae_3d_l32_c4_t4_s42_mae030.pt`. Per-date per-region scores in `outputs/scores/ae_3d_l32_c4_t4_s42_mae030.parquet`. Region mapping for integration into DATect's 10-site forecast is documented in `RESULTS.md`.
