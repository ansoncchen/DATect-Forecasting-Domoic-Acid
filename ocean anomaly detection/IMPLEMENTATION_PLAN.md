# Implementation plan: mask-aware spatiotemporal anomaly detection

This document tracks what's implemented and what remains. Phase A + B (temporal
3D ConvAE) is now wired up alongside the 2D snapshot baseline.

## Goals

1. **Validity masks first**: training and scoring never treat invalid pixels (clouds, land, gaps) like real observations.
2. **Short temporal context**: judge anomalies on how fields evolve, not single snapshots only (Phase A + B).
3. Optional **MAE-style augmentation** later (Phase C) — only after Phase B's E2 shows clean event signals.
4. **DDPM dropped** from scope: adds full U-Net + noise schedule for a result that doesn't strengthen the AE-vs-PCA core claim.

## Current state (✅ done, 🔄 in progress, ⬜ todo)

| Component | Status | Notes |
|---|---|---|
| Cube + mask + ERDDAP downloader | ✅ | `01_download.py` dedups time axis; `02_build_cube.py` deduplicates duplicates in concatenation |
| 2D ConvAE + masked_mse | ✅ | `src/model.py`; ~500K params at l=32 |
| 3D ConvAE3D + masked_mse_3d | ✅ | `src/model3d.py`; T=4 → 2 → 1 in 2 stride-2 time-downs; bottleneck flat dim = 8192 (same as 2D); 1.35M params |
| 2D tiled inference | ✅ | `src/infer.py`; identity reconstructor verified zero error |
| 3D tiled inference | ✅ | `src/infer3d.py`; tiles spatially only, preserves T whole, scores anchor frame only |
| Baselines B1/B2/B3 | ✅ | `src/baselines.py` |
| **Temporal PCA baseline (B3T)** | ✅ | Required for fair Phase B E4 comparison; fits PCA on flattened (C·T·P·P) vectors |
| 2D training loop | ✅ | `src/train.py` `train()` with coastal patch bias + fallback counter |
| 3D training loop | ✅ | `src/train.py` `train_temporal()` reusing shared loop; checkpoint name includes `t{T}` |
| Inference auto-dispatch (2D/3D) | ✅ | `scripts/04_run_inference.py` reads `variant` field from checkpoint |
| Evaluation E1/E2/E4/E5 + bootstrap CI | ✅ | `src/evaluate.py`; CI compares AE vs *matched* PCA (2D AE → B3, 3D AE → B3T) |
| ⬜ Phase C — MAE-style augmentation | ⬜ | Random masking during training; loss on masked-out subset only |
| ⬜ Spatial coherence (E3 — Moran's I) | ⬜ | Numpy implementation; needs per-pixel error maps saved to disk during inference |
| ⬜ Cloud-coverage sanity scatter | ⬜ | Score vs valid-pixel fraction per frame |

## Architecture decisions (locked in)

1. **Lookback temporal window**: anchor at last index, stack `[t-T+1, …, t]`. Means at inference we only need past observations — matches the operational forecasting use case and the parent DATect repo convention.
2. **Stride-2 in time** along with spatial strides. With `T=4`: 4 → 2 → 1, bottleneck stays comparable to 2D.
3. **`T=4` hardcoded** in `ConvAE3D` (raises `NotImplementedError` for other windows). Larger T would require adjusting the stride tuples in the encoder.
4. **Anchor-frame error**: 3D AE scores the anchor frame only. Same scalar shape as 2D AE — downstream evaluation pipeline consumes both uniformly.
5. **Coastal patch bias** (`TRAIN_COASTAL_PATCH_MIN_OVERLAP=0.5`): used for both 2D and 3D training and for both `B3` and `B3T` PCA fits. `--full-domain-patches` flag disables for ablation.
6. **Separate files** `src/model.py` vs `src/model3d.py` — keeps the working 2D baseline untouched; no `--temporal` switching inside one class.

## Evaluation and validation

- `05_evaluate.py` produces E1/E2/E4/E5 + yearly drift sanity figures.
- **E4 framing**: 3D AE's score at date *t* inherits info from *t-1 … t-T+1* by construction, so it's *expected* to be more forecastable than 2D AE. The honest comparison is **3D AE vs B3T** at matched bottleneck — this is what the bootstrap CI now reports.
- **Sanity**: per-epoch fallback rate from `PatchDataset`. Warns at >5% rate. Catches winter cloud-cover starvation.
- **Planned ablations**: 2D vs 3D AE; same `latent_dim`; channel ablations across both variants.

## File touch list (final state)

| Area | Files |
|------|--------|
| Config | `config.py` (+`TEMPORAL_WINDOW`, +`TRAIN_COASTAL_PATCH_MIN_OVERLAP`) |
| Data | `src/dataset_temporal.py` (new) |
| Model / loss | `src/model.py` (2D), `src/model3d.py` (3D, new) |
| Train | `src/train.py` (both `train()` and `train_temporal()`), `scripts/03_train_ae.py` (with `--temporal`) |
| Infer | `src/infer.py` (2D), `src/infer3d.py` (3D, new), `scripts/04_run_inference.py` (auto-dispatch) |
| Baselines | `src/baselines.py` (B1/B2/B3 + new `TemporalPCAReconstruction`) |
| Eval | `src/evaluate.py`, `scripts/05_evaluate.py` |

## Order of work (forward-looking)

1. ✅ Phase A + B implemented in this rebuild.
2. ⬜ Run end-to-end on Hyak: download remaining 3 channels (k490, nflh, sst), build cube, train, infer, evaluate.
3. ⬜ Phase C (optional) — only if Phase B's E2 plot shows clean event signals.

## Implemented safeguards already in code

- **Fallback counter in `PatchDataset` and `TemporalPatchDataset`** (`src/train.py`, `src/dataset_temporal.py`): per-epoch logging of `fallback/total` patches; warns at >5% rate even on non-reporting epochs.
- **`OVERALL_COAST_REGION` + `overall_coastal_mask()`** (`src/regions.py`): envelope of all subregions as primary rollup metric; also used as coastal patch sampling footprint during training.
- **`TRAIN_COASTAL_PATCH_MIN_OVERLAP=0.5`** (`config.py`): patches must have ≥50% overlap with the coastal envelope. `--full-domain-patches` flag disables this for ablation.
- **ERDDAP time-axis deduplication** (`02_build_cube.py`): the 8-day product is served indexed by daily timestamps with duplicate composites. The cube builder deduplicates on the `time` coordinate.

## Out of scope (firm)

- Full ViT-MAE at native grid resolution.
- DDPM baseline.
- Multi-GPU / mixed-precision (add only if training is too slow after baseline works).
- Changing ERDDAP download semantics beyond unique-date discovery.
