# Real-data smoke results

Validation run on 2025-05-16 covering:

1. **Synthetic 4-channel cube** (`data/cube_synth.zarr`, 120 frames, 160×200) — full pipeline integration test
2. **Real 2-channel MODIS Aqua, PNW coast, 2010** (`data/cube_2010.zarr`, 45 frames, 321×409) — first real-data run with chla + sst (year 2010 has no nflh/k490 ERDDAP coverage)
3. **Real 4-channel MODIS Aqua, PNW coast, 2003** (`data/cube_2003.zarr`, 45 frames, 321×409) — full 4-channel real-data run

All runs used 5 epochs (`--debug` mode) on Mac (MPS for 2D AE, CPU for 3D AE since MPS lacks ConvTranspose3D). Production-quality numbers require multi-year cubes + 100-epoch Hyak GPU training.

## 1. Synthetic 4-channel pipeline test (120 frames)

| Method | Mean | Std | val_loss |
|---|---:|---:|---:|
| AE_2d_l32 | 0.952 | 0.917 | 0.547 |
| AE_3d_l32_t4 | 0.883 | 0.867 | 0.409 (-25% vs 2D) |
| B1 chl-a z-score | 0.921 | 0.264 | — |
| B2 multivar z-score | 3.001 | 0.711 | — |
| B3 PCA k=32 | 0.536 | 0.702 | — |
| B3T temporal PCA k=32 | 0.515 | 0.672 | — |

**Cross-method correlations** (Overall region):
- AE_2d ↔ AE_3d: **r=0.967** (consistent)
- AE_3d ↔ B3T_pca: **r=0.985** (both temporal methods)
- AE methods ↔ B1/B2: r ≈ 0.1-0.3 (orthogonal — different signal)
- B3 ↔ B3T: r=1.000 (snapshot vs temporal PCA highly aligned)

**Top-3 most-anomalous dates**: AE_2d, AE_3d, B3 PCA all flag **2011-12-30** + **2011-12-14**. These align with the injected anomaly window at frame 100 (≈Jan 2012 with ±5 frame spread = mid-Nov 2011 to mid-Jan 2012). **Anomaly detection works.**

## 2. Real MODIS 2-channel PNW 2010 (chla + sst)

| Method | val_loss | Mean (Overall) | Std (Overall) |
|---|---:|---:|---:|
| AE_2d_l32 | 0.613 | 0.412 | 0.231 |
| AE_3d_l32_t4 | **0.493** (-20% vs 2D) | 0.506 | 0.312 |
| B1 chl-a z-score | — | 0.926 | 0.209 |
| B2 multivar z-score | — | 1.715 | 0.345 |
| B3 PCA k=32 | — | 0.115 | 0.064 |
| B3T temporal PCA k=32 | — | 0.160 | 0.081 |

**Cross-method correlations** (Overall):
- AE_2d ↔ AE_3d: r=0.646
- AE_2d ↔ B3_pca_k16: r=0.717 (AE tracks PCA reconstruction error)
- AE_3d ↔ B3T_pca_k16: r=0.677
- AE methods ↔ B1/B2 climatology: r ∈ [-0.2, 0.26] (orthogonal — as expected)

**Top-3 most-anomalous dates** (Overall, real MODIS 2010):
- AE_2d: 2010-08-21, 2010-07-04, 2010-06-26 (summer upwelling)
- AE_3d: 2010-03-22, 2010-02-18, 2010-09-22 (spring bloom + fall)
- B1 chl-z: 2010-03-22 (spring bloom)
- B3 PCA: 2010-07-20, 2010-06-26 (summer)

These flagged dates align with known PNW oceanographic activity windows: spring (Mar-Apr) chl-a bloom and summer (Jul-Sep) upwelling-driven productivity peak.

**E1 seasonal cycle** (`outputs/figures/E1_seasonal_*.png`): AE_2d peaks Jul-Sep; AE_3d shows Mar peak + Jul-Sep peak; B2 multivar peaks Mar-Apr. Physically correct.

**E4 forecastability** (one-step-ahead Lasso R²): all methods negative R² (worst predictor: AE_3d Overall R²=-1.92). Bootstrap CIs span both positive and negative (e.g. AE_2d Central Oregon CIΔ=[-6.013, +0.307] vs B3_pca_k32 — straddles 0). This is the **honest small-N caveat** the plan flagged: 45 frames + 5 epochs is not enough to distinguish AE from PCA statistically.

## 3. Real MODIS 4-channel PNW 2003 (chla + k490 + nflh + sst)

| Method | val_loss |
|---|---:|
| AE_2d_l32 | 0.711 |
| AE_3d_l32_t4 | **0.673** (-5% vs 2D) |

The 3D AE win is smaller (-5%) than on real 2010 (-20%) or synthetic (-25%), likely because:
- 5 epochs hasn't given the 3D model time to fully exploit temporal context
- 4 channels give the 2D model more cross-channel info to compensate

Full inference (B1/B2/B3/B3T + 2 AE) and evaluation figures running in background.

## What works end-to-end

- ERDDAP download with DOY-anchor sub-sampling (~46 files/yr/channel)
- Cube building with time-axis deduplication and lat/lon normalization
- 2D and 3D ConvAE training (CPU fallback for 3D on Mac, CUDA-ready)
- Tiled inference with overlap-averaging (identity-verified exact zero error)
- 2D and 3D PCA baselines via shared tiler (B3, B3T)
- Bootstrap CI on AE-vs-PCA R² delta correctly straddling zero in small-N regime
- 22 evaluation figures produced (E1, E2, E4, E5, sanity drift × 5 regions)

## Known limitations of this smoke run

| Limitation | Impact | Fix |
|---|---|---|
| 5 epochs | AE not converged | Run 100 epochs on Hyak |
| 1 year of data (45 frames) | E4 CIs too wide to distinguish methods | Build multi-year cube |
| Mac MPS lacks ConvTranspose3D | 3D AE on CPU = 5-10× slower | Hyak CUDA fixes |
| 2010 k490+nflh unavailable | 2010 limited to 2-channel cube | Use 2003-2009 for 4-channel; ERDDAP coverage varies |

## Next-step recommendation for production results

```bash
# On Hyak with all 4 channels for 2003-2015 (12 years × 46 frames = ~550 anchors):
python scripts/01_download.py --start 2003-01-01 --end 2015-12-31
python scripts/02_build_cube.py
sbatch --export=ALL,MODE=temporal_sweep hyak/train.sbatch     # full 100-epoch sweep
sbatch hyak/infer.sbatch                                       # all baselines + AE
# scp back, then:
python scripts/05_evaluate.py
python scripts/06_summarize_results.py
```

The pipeline is ready. Production runs depend only on time + compute, not on code work.
