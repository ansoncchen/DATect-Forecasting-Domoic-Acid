## Learned User Preferences

- Use `python3` (avoid typo `pytho`).

## Learned Workspace Facts

- `src/regions.py`: Four named `SUBREGIONS` bands; `OVERALL_COAST_REGION` is the lat/lon bbox envelope; `REGIONS = [OVERALL_COAST_REGION] + SUBREGIONS` with overall first. `build_region_masks` and `aggregate_to_regions` consume `REGIONS`. `overall_coastal_mask()` defines geography for AE and PCA patch sampling.
- `config.TRAIN_COASTAL_PATCH_MIN_OVERLAP`: Coastal patches biased to the overall coastal bbox via `train(coastal_patch_min_overlap=...)` and PCA `fit(..., coastal_patch_min_overlap)`. `scripts/03_train_ae.py --full-domain-patches` sets overlap to 0. Inference uses full-grid tiling.
- Run order: `scripts/01_download.py` → `02_build_cube.py` → `03_train_ae.py` → `04_run_inference.py` → `05_evaluate.py`. Re-run inference after region changes.
- `scripts/01_download.py`: IDs like `erdMWchla8day_LonPM180` refer to MODIS 8-day science product naming, not sparse ERDDAP timesteps; CoastWatch ERDDAP often exposes roughly daily timestamps (~364/year per calendar year). The script follows ERDDAP’s native `time` axis with one `.nc` per listed date; docstrings implying one file per 8-day composite can misrepresent actual ERDDAP cadence.
- Raw `data/raw/chla/chla_*.nc`: singleton `time` per file with `chlorophyll` (mg m⁻³); valid ocean pixels are often a minority of the grid and vary sharply by date (clouds, land). Pipelines should use validity masks / masked losses rather than treating full frames as dense images.