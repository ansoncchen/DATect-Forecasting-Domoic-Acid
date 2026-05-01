# Implementation plan: mask-aware and spatiotemporal anomaly detection

This document turns the agreed direction into concrete phases aligned with the current repo (`data/cube.zarr`, `mask`, `src/model.py` `ConvAE`, `masked_mse`, scripts `01`–`05`).

## Goals

1. Keep **validity masks first**: training and scoring must not treat invalid pixels (clouds, land, gaps) like real observations.
2. Add **short temporal context** so anomalies are judged on **how fields evolve**, not a single snapshot only.
3. Optionally add **held-out pixel prediction** (MAE-style) so the model explicitly practices “fill in missing regions.”
4. Leave **lightweight DDPM** as a later comparison, not a blocker for the main pipeline.

## Current baseline (reference)

- Cube: `(time, channel, lat, lon)` + `mask(time, lat, lon)` from `scripts/02_build_cube.py`.
- Model: 2D `ConvAE`, 64×64 patches, `masked_mse` in `src/model.py`.
- Training / inference / eval: `scripts/03_train_ae.py`, `04_run_inference.py`, `05_evaluate.py`.

## Phase A — Data API for temporal stacks

**Objective:** Produce training samples as `(x, mask)` with shape `(B, C, T, H, W)` and `(B, 1, T, H, W)` (or `(B, 1, H, W)` if you collapse time in the loss later).

**Tasks**

1. **Config** (`config.py`): add e.g. `TEMPORAL_WINDOW` (default `4`), `TEMPORAL_STRIDE` for sliding windows over `time`, minimum valid fraction per patch (reuse or extend coastal patch overlap rules).
2. **Dataset module** (new file e.g. `src/dataset_temporal.py` or extend existing loader in `src/train.py`):
   - For each anchor time index `t`, stack channels for times `t, t-1, …` (or `t … t+T-1` — pick one convention and stick to it).
   - Align with the cube’s **shared time axis**; skip windows where alignment is impossible near the start of the series.
   - Emit patches already standardized (same as today) plus a **boolean/float mask** per pixel per time step (intersect channel validity if needed).
3. **Patch eligibility:** Reuse coastal bias from `TRAIN_COASTAL_PATCH_MIN_OVERLAP`; add rule such as “discard windows where valid pixel count in patch < X%” to avoid empty training steps.

**Acceptance:** Single batch loads on GPU with stable shapes; NaNs absent from tensors (masked out or filled with 0 where mask=0, consistent with current 2D path).

## Phase B — 3D spatiotemporal ConvAE (primary model)

**Objective:** Same reconstruction loss as today, extended over time; anomaly score = reconstruction error on **valid** locations only.

**Tasks**

1. **Model** (`src/model.py` or `src/model3d.py`): `Conv3d` encoder–decoder mirroring depths of current 2D AE (or 2D backbone + temporal fusion module v1 — choose one for simplicity).
   - Input: `(B, C*T or C, T, H, W)` — if you use `C*T` folded into channels, document clearly; prefer **`(B, C, T, H, W)` with true 3D convs** for clarity.
   - Output shape matches input.
2. **Loss:** Generalize `masked_mse` to 5D tensors (mask broadcast over channels; sum over valid voxels only).
3. **Training script:** `scripts/03_train_ae.py` — add flag `--temporal` / config branch: dataset from Phase A, checkpoint name includes `t{T}`.
4. **Inference** (`src/infer.py`, `04_run_inference.py`): tiled windows over space **and** sliding window over time; write scores back to a compatible format with current regional aggregation (may need to map per-time scores to a scalar per date for comparison with PCA baselines — decide and document).

**Acceptance:** Trains to convergence on a 1-year slice; inference produces a score map or regional series without shape errors; eval script runs with a new method label (e.g. `AE3D_l32_t4`).

**Hardware:** Target **2080 Ti (11 GB)** with modest `batch_size` and gradient accumulation if needed; **3060 Ti** for longer sweeps with smaller batch.

## Phase C — MAE-style augmentation (optional but high value)

**Objective:** During training, randomly mask **additional** valid pixels (or whole patch tokens); loss computed on **masked-out** subset only (or joint with full reconstruction — pick one objective to avoid double-counting).

**Tasks**

1. In training step only: build `mask_train = valid_mask * random_mask`.
2. Either:
   - **Inpainting head:** model sees corrupted input (zeros or mean-filled where random mask), predicts full field; loss on random-masked **and** still-valid region, or
   - **Decoder-only target:** standard AE but MSE only on pixels removed by random mask.
3. Keep inference **unchanged** (full valid observation, no random mask) unless you explicitly want “score = hole-filling error.”

**Acceptance:** Ablate one run with vs without random mask; table in eval shows whether regional metrics or DA alignment improve.

## Phase D — DDPM comparison (defer)

**Objective:** One small U-Net diffusion baseline, **same patches and masks** as Phase B.

**Tasks:** Only after Phases A–B are stable: timestep schedule, masked loss on noise prediction or on x0-reconstruction, short training schedule suitable for 8 h Hyak slots + long home runs.

**Acceptance:** Single row in evaluation parquet; narrative: “heavier generative baseline, not default.”

## Evaluation and validation

- Reuse `05_evaluate.py` pathways; add method string for 3D AE.
- **Sanity checks:** cloud-heavy days (low valid %) should not dominate loss; log mean valid fraction per batch during training.
- **Planned ablations:** 2D vs 3D; with vs without random-mask training; same `latent_dim` for fair comparison.

## File touch list (expected)

| Area | Files |
|------|--------|
| Config | `config.py` |
| Data | new `src/dataset_temporal.py` (or extend `src/train.py`) |
| Model / loss | `src/model.py` or `src/model3d.py` |
| Train | `src/train.py`, `scripts/03_train_ae.py` |
| Infer | `src/infer.py`, `scripts/04_run_inference.py` |
| Eval | `src/evaluate.py`, `scripts/05_evaluate.py` (if new method labels need wiring) |

## Order of work

1. Phase A (temporal batches + masks)  
2. Phase B (3D ConvAE + train/infer/eval)  
3. Phase C (MAE-style augmentation)  
4. Phase D (DDPM) if needed for write-up

## Out of scope (for now)

- Full ViT-MAE at native grid resolution  
- Multi-GPU / mixed-precision (add only if training is too slow after baseline works)  
- Changing ERDDAP download semantics (`01_download.py`) unless cube definition changes
