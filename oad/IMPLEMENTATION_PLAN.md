# Implementation plan: mask-aware spatiotemporal anomaly detection

Status: **Phases A + B + C all implemented and validated on Hyak with the full
22-year 4-channel cube.** Phase D (DDPM) deliberately out of scope.

## Goals

1. **Validity masks first**: training and scoring never treat invalid pixels (clouds, land, gaps) like real observations.
2. **Short temporal context**: judge anomalies on how fields evolve, not single snapshots only (Phase A + B).
3. **MAE-style augmentation** (Phase C) — random pixel hiding during training, loss on hidden-valid only. Implemented and was the headline win in SW Washington/Long Beach.
4. **DDPM dropped** from scope: adds full U-Net + noise schedule for a result that doesn't strengthen the AE-vs-PCA core claim.

## Current state (✅ done, 🔄 in progress, ⬜ todo)

| Component | Status | Notes |
|---|---|---|
| Cube + mask + ERDDAP downloader | ✅ | Per-month chunking (`--per-year`), DOY anchor sub-sampling (`--anchor-only`), 429 backoff; 22-year cube on Hyak at `data/cube.zarr` (4692 frames × 4 channels × 321 × 409, ~9.86 GB) |
| 2D ConvAE + masked_mse | ✅ | `src/model.py`; ~500K params at l=32 |
| 3D ConvAE3D + masked_mse_3d | ✅ | `src/model3d.py`; T=4 → 2 → 1 in 2 stride-2 time-downs; bottleneck flat dim = 8192 (same as 2D); 1.35M params |
| 2D tiled inference | ✅ | `src/infer.py`; identity reconstructor verified zero error |
| 3D tiled inference | ✅ | `src/infer3d.py`; tiles spatially only, preserves T whole, scores anchor frame only |
| Baselines B1/B2/B3 | ✅ | `src/baselines.py` |
| Temporal PCA baseline (B3T) | ✅ | Matched-bottleneck PCA on flattened (C·T·P·P) vectors |
| 2D training loop | ✅ | `src/train.py::train` with coastal patch bias + fallback counter |
| 3D training loop | ✅ | `src/train.py::train_temporal` reusing shared loop; checkpoint name includes `t{T}` |
| **Phase C MAE training** | ✅ | `--mask-ratio` flag on `scripts/03_train_ae.py`; checkpoint name includes `_maeNNN`; loss computed only on hidden-AND-valid pixels |
| Inference auto-dispatch (2D/3D/MAE) | ✅ | `scripts/04_run_inference.py` reads `variant` + `mask_ratio` from checkpoint; method names include `_maeNNN` |
| Evaluation E1/E2/E4/E5 + bootstrap CI | ✅ | `src/evaluate.py`; CI compares AE vs *matched* PCA (2D AE → B3, 3D AE → B3T) |
| Sanity drift + cloud confound | ✅ | `plot_yearly_drift`, `plot_cloud_confound`, `plot_aggregation_comparison` |
| Annual-cycle stacked-years plot | ✅ | `scripts/plot_annual_cycle.py` — shows seasonal cycle repeating across all 22 years |
| ⬜ Spatial coherence (E3 — Moran's I) | ⬜ | `morans_i` function exists in `src/evaluate.py`; not wired into the main eval driver because it needs per-pixel error maps saved during inference |
| ⬜ Integration into DATect main forecast | ⬜ | Plan: site → region mapping, lag = `test_date − 12`, 7 derived features. See `RESULTS.md` and the integration prompt in chat history |

## Architecture decisions (locked in)

1. **Lookback temporal window**: anchor at last index, stack `[t-T+1, …, t]`. Means at inference we only need past observations — matches the operational forecasting use case and the parent DATect repo convention.
2. **Stride-2 in time** along with spatial strides. With `T=4`: 4 → 2 → 1, bottleneck stays comparable to 2D.
3. **`T=4` hardcoded** in `ConvAE3D` (raises `NotImplementedError` for other windows). Larger T would require adjusting the stride tuples in the encoder.
4. **Anchor-frame error**: 3D AE scores the anchor frame only. Same scalar shape as 2D AE — downstream evaluation pipeline consumes both uniformly.
5. **Coastal patch bias** (`TRAIN_COASTAL_PATCH_MIN_OVERLAP=0.5`): used for both 2D and 3D training and for both `B3` and `B3T` PCA fits. `--full-domain-patches` flag disables for ablation.
6. **Separate files** `src/model.py` vs `src/model3d.py` — keeps the working 2D baseline untouched; no `--temporal` switching inside one class.
7. **Phase C MAE convention**: hidden mask is per-pixel (broadcasts over channels) so the model can't cheat by reading another channel at the same pixel. Loss = `masked_mse(pred, target, valid_mask * hidden_mask)`. Val loss uses the same protocol so train/val are directly comparable.
8. **MODIS 8-day composites are CENTERED on the labeled date** (`long_name="Centered Time"`). Any downstream consumer (e.g. integrating scores into DATect) must lag by ≥4 days beyond their normal anchor convention to avoid leakage.

## Evaluation and validation

- `scripts/05_evaluate.py` produces E1/E2/E4/E5 + yearly drift sanity figures across all 5 regions.
- `scripts/06_summarize_results.py` produces `RESULTS_summary_full.txt` with cross-method correlation matrices, top-3 anomalous dates, and full E4 bootstrap CIs per region.
- **E4 framing**: 3D AE's score at date *t* inherits info from *t-1 … t-T+1* by construction, so it's *expected* to be more forecastable than 2D AE. The honest comparison is **3D AE vs B3T** at matched bottleneck — this is what the bootstrap CI reports.
- **Sanity**: per-epoch fallback rate from `PatchDataset` / `TemporalPatchDataset`. Warns at >5% rate. Catches winter cloud-cover starvation.

## File touch list (final state)

| Area | Files |
|------|--------|
| Config | `config.py` (+`TEMPORAL_WINDOW`, +`TRAIN_COASTAL_PATCH_MIN_OVERLAP`) |
| Data | `src/dataset_temporal.py` (new) |
| Model / loss | `src/model.py` (2D), `src/model3d.py` (3D, new) |
| Train | `src/train.py` (`train()` + `train_temporal()` with `mask_ratio` param), `scripts/03_train_ae.py` (with `--temporal`, `--mask-ratio`) |
| Infer | `src/infer.py` (2D), `src/infer3d.py` (3D, new), `scripts/04_run_inference.py` (auto-dispatch + `_maeNNN` method naming) |
| Baselines | `src/baselines.py` (B1/B2/B3 + `TemporalPCAReconstruction`) |
| Eval | `src/evaluate.py`, `scripts/05_evaluate.py`, `scripts/06_summarize_results.py`, `scripts/plot_annual_cycle.py` |
| Hyak | `hyak/download.sbatch`, `hyak/build_cube.sbatch`, `hyak/train.sbatch`, `hyak/mae_train.sbatch`, `hyak/infer.sbatch`, `hyak/full_pipeline.sbatch`, `hyak/verify_files.sbatch` |

## Order of work (forward-looking)

1. ✅ Phase A + B implemented; full 22-year cube downloaded; 100-epoch sweeps + ablations completed; results documented in `RESULTS.md`.
2. ✅ Phase C MAE implemented and tested at ratios {0.15, 0.30, 0.40, 0.50, 0.70} for both 2D + 3D variants.
3. ⬜ **Integrate OAD into DATect main forecast** — add 7 `oad_*` features per (site, date) row using `test_date − 12` lag, re-run `scripts/eval/eval_paper_metrics.py` baseline vs +OAD, compare R² / MAE / spike recall.
4. ⬜ Phase D (DDPM) — explicitly out of scope (see goals §4).

## Implemented safeguards already in code

- **Fallback counter in `PatchDataset` and `TemporalPatchDataset`** (`src/train.py`, `src/dataset_temporal.py`): per-epoch logging of `fallback/total` patches; warns at >5% rate even on non-reporting epochs.
- **`OVERALL_COAST_REGION` + `overall_coastal_mask()`** (`src/regions.py`): envelope of all subregions as primary rollup metric; also used as coastal patch sampling footprint during training.
- **`TRAIN_COASTAL_PATCH_MIN_OVERLAP=0.5`** (`config.py`): patches must have ≥50% overlap with the coastal envelope. `--full-domain-patches` flag disables this for ablation.
- **ERDDAP time-axis deduplication** (`02_build_cube.py`): the 8-day product is served indexed by daily timestamps with duplicate composites. The cube builder deduplicates on the `time` coordinate.
- **HTTP 429 backoff** (`scripts/01_download.py`): rate-limit retries with exponential backoff up to 5 min.
- **`set -euo pipefail`** in all sbatch scripts so a failed stage doesn't cascade into garbage results.
- **`verify_files.sbatch`**: gate that checks all 4 channels have ≥260 distinct months covered before letting the GPU training chain start (uses `--dependency=afterok:verify-job`).

## Out of scope (firm)

- Full ViT-MAE at native grid resolution.
- DDPM baseline.
- Multi-GPU / mixed-precision (add only if training is too slow after baseline works).
- Changing ERDDAP download semantics beyond unique-date discovery + 429 backoff.
