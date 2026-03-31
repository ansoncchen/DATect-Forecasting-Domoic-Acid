# Satellite Computer Vision for Bloom Regime Detection

## Motivation

DATect currently extracts **single point values** from MODIS satellite imagery — spatially averaging all pixels within a small bounding box (~0.05-0.1 degrees) around each monitoring site. This discards rich spatial information:

- **Bloom plume geometry**: Chlorophyll plumes have shape, orientation, and extent that indicate bloom stage and transport direction
- **Upwelling fronts**: SST gradients reveal coastal upwelling patterns that drive nutrient delivery
- **Spatial bloom progression**: Blooms propagate along-coast; a CV model could detect an approaching bloom before it reaches the monitoring site
- **Cross-site spatial context**: Adjacent sites share oceanographic forcing that point extraction treats independently

A computer vision approach on raw satellite tiles could capture these patterns and provide a **regime state** (dormant / bloom-primed / active bloom) as an input feature to the existing forecasting pipeline.

---

## Current Point Extraction Limitations

```
Current pipeline (dataset-creation.py):
  MODIS 8-day composite → NetCDF tile (~4km resolution)
    → spatial mean over small bounding box
      → single scalar per variable per timestep

Information lost:
  - Spatial gradients (SST fronts, chlorophyll plumes)
  - Bloom extent and shape
  - Along-coast propagation patterns
  - Offshore vs nearshore bloom positioning
```

---

## Proposed Architecture

### Overview

```
Raw MODIS tiles (multi-band)
  → Pretrained encoder (self-supervised on unlabeled tiles)
    → Spatial embedding vector per timestep
      → Regime classifier OR direct feature for XGBoost/RF
```

### Phase 1: Data Collection and Preparation

**Satellite Products (multi-band input):**
| Band | Product | Resolution | What It Captures |
|------|---------|-----------|-----------------|
| 1 | Chlorophyll-a (modis-chla) | ~4km, 8-day | Bloom intensity and extent |
| 2 | SST (modis-sst) | ~4km, 8-day | Upwelling fronts, thermal structure |
| 3 | FLH (fluorescence line height) | ~4km, 8-day | Active photosynthesis (better bloom indicator than chla) |
| 4 | PAR | ~4km, 8-day | Light availability for growth |
| 5 | K490 (diffuse attenuation) | ~4km, 8-day | Water clarity / sediment load |

**Tile Size:**
- **Option A: Site-local (small)** — ~50x50 km region around each site (~12x12 pixels at 4km). Captures local bloom state. Lower data requirements.
- **Option B: Regional (large)** — ~600x200 km covering the full WA-OR coast (~150x50 pixels). Captures along-coast propagation. Much richer but needs more compute and data.
- **Recommended: Start with Option A**, scale to Option B if results are promising.

**Temporal Coverage:**
- MODIS-Aqua: July 2002 – present (~2,800 8-day composites over 22 years)
- Each composite is one "image" — so ~2,800 labeled samples per site (Option A) or ~2,800 total (Option B)
- This is actually a reasonable dataset size for a small CNN

**Cloud Masking Challenge:**
- Pacific Northwest cloud cover is ~70% during peak bloom season (April-October)
- 8-day composites already mitigate this (pixel-level temporal averaging)
- Remaining missing pixels: use NaN masking or inpainting
- **Critical**: Cloud gaps are non-random (more clouds = more upwelling = more blooms). The model must learn that missing data is informative, not just noise.

**Data Source:**
- Same ERDDAP endpoints already used in `dataset-creation.py`, but request **full spatial grids** instead of averaging
- Modify the download to retain the lat/lon dimensions instead of calling `.mean(dim=spatial_dims)`
- Alternatively, use NASA OceanColor L3 mapped products (global 4km, 8-day) via OPeNDAP

### Phase 2: Self-Supervised Pretraining (Recommended)

**Why pretrain?**
- Only ~900 DA measurement dates with co-located satellite imagery (sparse labels)
- ~2,800 8-day composites available as unlabeled images (3x more)
- Self-supervised learning extracts useful representations without labels

**Approaches (ranked by feasibility):**

1. **Masked Autoencoder (MAE)** — Mask 50-75% of patches, train encoder-decoder to reconstruct. Works well with small datasets and naturally handles cloud-masked pixels (clouds = additional masking). This is the recommended approach.

2. **Contrastive Learning (SimCLR/MoCo)** — Learn representations where temporally close tiles are similar and distant tiles are different. Good for capturing seasonal/regime structure. More complex to implement.

3. **Variational Autoencoder (VAE)** — Learn compressed latent space. Latent dimensions might naturally separate into bloom regimes. Provides uncertainty estimates.

**Architecture:**
- **Encoder**: Small ResNet-18 or Vision Transformer (ViT-Tiny) with 5 input channels
- **Patch size**: 4x4 pixels (for ViT) or standard conv layers (for ResNet)
- **Output**: 64-128 dimensional embedding per tile
- **Training**: ~2,800 tiles × 10 sites = ~28,000 samples (Option A) — sufficient for a small model

### Phase 3: Regime Classification / Feature Extraction

**Option A: Explicit Regime Labels**
- Define regimes from DA measurements: Dormant (DA < 5, 4+ weeks), Primed (rising environmental indicators, DA < 20), Active (DA >= 20)
- Train classifier head on pretrained encoder
- Output: `regime_prob_active`, `regime_prob_primed` as features for XGBoost/RF
- **Pro**: Interpretable. **Con**: Requires defining regimes from the sparse labels you're trying to predict.

**Option B: Direct Feature Extraction (Recommended)**
- Use pretrained encoder to produce embedding vector per tile
- Apply PCA to reduce to 3-5 dimensions
- Feed PCA components directly as features into existing XGBoost/RF pipeline
- **Pro**: No regime definition needed; model learns what's useful. **Con**: Less interpretable.
- The XGBoost/RF models are good at selecting useful features from noisy inputs — let them decide.

**Option C: Anomaly-Based Regime Detection**
- Train autoencoder on "normal" (low-DA) tiles only
- Reconstruction error on new tiles = "abnormality score"
- High reconstruction error = bloom-like conditions
- **Pro**: Doesn't need bloom labels for training. **Con**: Assumes blooms are anomalous (mostly true, given 87% are low-DA).

### Phase 4: Integration with DATect

```python
# In feature_utils.py or new module

def extract_satellite_cv_features(site, date, encoder, pca):
    """Extract CV-derived features from raw satellite tile."""
    tile = load_satellite_tile(site, date)  # 5-band, HxW
    if tile is None or tile_cloud_fraction(tile) > 0.8:
        return None  # Too cloudy — fall back to point features

    embedding = encoder(tile)  # 64-128 dim vector
    pca_features = pca.transform(embedding)  # 3-5 dim

    return {
        f'sat_cv_pc{i}': v for i, v in enumerate(pca_features)
    }
```

**Temporal integrity**: Encoder and PCA must be fit only on data before anchor_date. For practical purposes, fit once on pre-2018 data (before the test period) and freeze.

---

## Data Requirements Assessment

| Component | Data Available | Data Needed | Feasible? |
|-----------|---------------|-------------|-----------|
| Unlabeled satellite tiles | ~2,800 per site (8-day, 22 years) | ~1,000+ for pretraining | Yes |
| DA-labeled tiles | ~700 per WA site, ~60-140 per OR site | ~200+ for supervised fine-tuning | Marginal (WA yes, OR no) |
| Multi-band imagery | 5 MODIS products already sourced | Same products, full spatial grid | Yes (pipeline modification) |
| Compute for training | GPU needed | 1-2 hours on single GPU (small model) | Yes |

**Verdict**: Feasible for WA sites and regional model. OR sites lack sufficient labels for site-specific fine-tuning but could benefit from a regional model trained on WA data.

---

## Key Challenges

### 1. Cloud Cover Correlation with Blooms
Cloud cover peaks during upwelling season (bloom season). Missing satellite data is systematically biased toward the most interesting periods. Solutions:
- Use cloud fraction as an explicit feature (informative missingness)
- Multi-temporal compositing (16-day or monthly reduces gaps but loses temporal resolution)
- Sentinel-3 OLCI as backup (different orbit, partially decorrelated cloud gaps)

### 2. MODIS End-of-Life
MODIS-Aqua is aging (launched 2002). Transition to **PACE** (launched Feb 2024) is imminent.
- PACE has hyperspectral ocean color (350-885nm) — could directly identify Pseudo-nitzschia spectral signatures
- Model would need retraining on PACE data
- Consider building the pipeline to be sensor-agnostic (normalize to standard bands)

### 3. Label Sparsity for Oregon Sites
OR sites have 61-144 total DA measurements. Even with transfer learning from WA sites, fine-tuning on this few labels is challenging.
- **Mitigation**: Regional model (Option B tile size) pools spatial information across all sites
- **Mitigation**: Use WA-pretrained encoder as frozen feature extractor for OR sites

### 4. Spatial Resolution vs. Coastal Dynamics
4km MODIS pixels are coarse for nearshore processes. Key bloom dynamics (upwelling plumes, river plumes from Columbia River) occur at sub-km scales.
- Sentinel-2 MSI (10-60m) or Landsat (30m) offer much higher resolution but lack ocean color bands
- PACE will improve spectral resolution but not spatial (~1km)

---

## Implementation Roadmap

### Stage 1: Data Pipeline Extension (1-2 weeks)
- Modify `dataset-creation.py` to retain full spatial grids instead of averaging
- Download and store tiles as numpy arrays or zarr (compressed)
- Build cloud masking pipeline (MODIS quality flags)
- Estimated storage: ~50-100 GB for full archive

### Stage 2: Self-Supervised Pretraining (2-3 weeks)
- Implement Masked Autoencoder on multi-band tiles
- Train on all available tiles (unlabeled)
- Validate: do learned representations cluster by season/regime?
- Tools: PyTorch, timm (for ViT backbone)

### Stage 3: Feature Integration (1-2 weeks)
- Extract embeddings for all tiles
- PCA to 3-5 components
- Add as features to existing XGBoost/RF pipeline
- Evaluate impact on R2, MAE, spike recall
- **Critical**: temporal split — encoder trained on pre-test data only

### Stage 4: Evaluation and Iteration (2-3 weeks)
- Compare point-only vs point+CV features
- Per-site breakdown (expect biggest gains at WA sites with dense labels)
- Ablation: which PCA components matter?
- Visualization: what do the learned representations look like?

**Total estimated timeline: 6-10 weeks**

---

## Alternative: Simpler Spatial Features (Quick Win)

Before committing to full CV, consider extracting **hand-crafted spatial features** from the same tiles:

```python
# Simple spatial statistics from raw tiles (no deep learning needed)
spatial_features = {
    'chla_spatial_std': tile_chla.std(),        # Bloom patchiness
    'chla_spatial_max': tile_chla.max(),        # Peak intensity anywhere in region
    'sst_gradient_mag': np.gradient(tile_sst),  # Upwelling front strength
    'chla_offshore_ratio': offshore_mean / nearshore_mean,  # Bloom position
    'bloom_pixel_fraction': (tile_chla > threshold).mean(),  # Bloom extent
}
```

These could be computed from the same ERDDAP data with minimal pipeline changes and tested within 1-2 days. If spatial statistics improve skill, that validates the full CV approach.

---

## Expected Impact

| Approach | Expected R2 Impact | Confidence | Timeline |
|----------|-------------------|------------|----------|
| Simple spatial statistics | +0.01-0.03 | Medium | 1-2 days |
| CV embeddings (site-local) | +0.02-0.05 | Medium-Low | 6-10 weeks |
| CV embeddings (regional) | +0.03-0.08 | Low | 8-12 weeks |
| PACE hyperspectral CV | +0.05-0.15 | Medium (long-term) | 6+ months |

The honest expectation: CV-derived features will be **additive but not transformative** for DATect in its current form. The binding constraint remains DA measurement sparsity at Oregon sites, which no amount of satellite imagery can substitute for. However, spatial features could meaningfully improve **spike timing** by detecting approaching blooms before they reach monitoring sites.

---

## References

- He et al. (2022). Masked Autoencoders Are Scalable Vision Learners. CVPR.
- Caron et al. (2021). Emerging Properties in Self-Supervised Vision Transformers. ICCV.
- Brunson et al. (2024). DA biosynthesis gene coexpression analysis for DA prediction.
- PACE Mission: https://pace.gsfc.nasa.gov/ (launched Feb 8, 2024)
- MODIS Ocean Color: https://oceancolor.gsfc.nasa.gov/
