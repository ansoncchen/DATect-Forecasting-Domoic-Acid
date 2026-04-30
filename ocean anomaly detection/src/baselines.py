"""
Three baselines for comparison with the ConvAE anomaly metric.

B1  — single-channel chl-a z-score (climatological DOY baseline)
B2  — multivariate climatological z-score (all 4 channels)
B3  — linear PCA reconstruction error (matched-k sweep)

All baselines expose a fit(cube) → self and score(cube) → DataFrame interface.

The B3 PCA tiled reconstructor calls src/infer.py so the tiler is exercised
on a trusted method before the AE exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import IncrementalPCA

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.regions import build_region_masks, aggregate_to_regions
from src.infer import reconstruct_frame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doy_climatology(
    da: xr.DataArray,
    smooth_days: int = 14,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-pixel DOY climatological mean and std from a (time, lat, lon) DataArray.
    smooth_days: ±window in days for rolling smoothing across DOYs.
    Returns (mean_doy, std_doy) arrays of shape (366, H, W).
    """
    vals = da.values  # (T, H, W)
    times = pd.DatetimeIndex(da.time.values)
    doys = times.dayofyear  # 1–366

    H, W = vals.shape[1], vals.shape[2]
    clim_mean = np.full((366, H, W), np.nan, dtype=np.float32)
    clim_std = np.full((366, H, W), np.nan, dtype=np.float32)

    window = smooth_days // 8  # convert days to 8-day steps (minimum 1)
    window = max(window, 1)

    for d in range(1, 367):
        # Gather all years' frames within ±window 8-day steps of this DOY
        idx = np.where(np.abs(doys - d) <= window * 8)[0]
        if len(idx) == 0:
            continue
        slab = vals[idx]  # (k, H, W)
        clim_mean[d - 1] = np.nanmean(slab, axis=0)
        clim_std[d - 1] = np.nanstd(slab, axis=0)

    return clim_mean, clim_std


def _zscore_frame(
    frame: np.ndarray,  # (H, W)
    doy: int,           # 1-indexed
    clim_mean: np.ndarray,
    clim_std: np.ndarray,
) -> np.ndarray:
    mu = clim_mean[doy - 1]
    sigma = clim_std[doy - 1]
    return (frame - mu) / np.where(sigma > 1e-8, sigma, np.nan)


# ---------------------------------------------------------------------------
# B1: Single-channel chl-a z-score
# ---------------------------------------------------------------------------

class ChlaZScore:
    """B1 baseline."""

    def fit(self, cube: xr.Dataset) -> "ChlaZScore":
        ch_idx = list(cube.attrs["channels"]).index("chla")
        da = cube["data"].isel(channel=ch_idx)
        print("B1: computing chl-a DOY climatology…")
        self._mean, self._std = _doy_climatology(da)
        return self

    def score(
        self,
        cube: xr.Dataset,
        aggregation: str = "mean",
    ) -> pd.DataFrame:
        ch_idx = list(cube.attrs["channels"]).index("chla")
        data = cube["data"].values       # (T, C, H, W)
        mask = cube["mask"].values       # (T, H, W)
        times = pd.DatetimeIndex(cube["data"].time.values)
        lat = cube["data"].lat.values
        lon = cube["data"].lon.values
        region_masks = build_region_masks(lat, lon)

        rows = []
        for t_idx, ts in enumerate(times):
            frame = data[t_idx, ch_idx]  # (H, W)
            vm = mask[t_idx]
            doy = ts.dayofyear
            zscore = _zscore_frame(frame, doy, self._mean, self._std)
            sq_err = zscore ** 2
            sq_err = np.where(vm, sq_err, np.nan)
            scores = aggregate_to_regions(sq_err, vm, region_masks, aggregation)
            for region, val in scores.items():
                rows.append({"date": ts.date(), "region": region, "method": "B1_chla_zscore",
                             "aggregation": aggregation, "score": val})
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# B2: Multivariate climatological z-score
# ---------------------------------------------------------------------------

class MultivarZScore:
    """B2 baseline — independent z-score per channel, sum-of-squares per pixel."""

    def fit(self, cube: xr.Dataset) -> "MultivarZScore":
        channels = list(cube.attrs["channels"])
        self._clim = {}
        for i, ch in enumerate(channels):
            da = cube["data"].isel(channel=i)
            print(f"B2: computing DOY climatology for {ch}…")
            mu, sigma = _doy_climatology(da)
            self._clim[ch] = (mu, sigma)
        self._channels = channels
        return self

    def score(
        self,
        cube: xr.Dataset,
        aggregation: str = "mean",
    ) -> pd.DataFrame:
        channels = self._channels
        data = cube["data"].values
        mask = cube["mask"].values
        times = pd.DatetimeIndex(cube["data"].time.values)
        lat = cube["data"].lat.values
        lon = cube["data"].lon.values
        region_masks = build_region_masks(lat, lon)

        rows = []
        for t_idx, ts in enumerate(times):
            vm = mask[t_idx]
            doy = ts.dayofyear
            sum_sq = np.zeros(vm.shape, dtype=np.float32)
            for i, ch in enumerate(channels):
                frame = data[t_idx, i]
                z = _zscore_frame(frame, doy, *self._clim[ch])
                sum_sq += np.nan_to_num(z ** 2, nan=0.0)
            sum_sq = np.where(vm, sum_sq, np.nan)
            scores = aggregate_to_regions(sum_sq, vm, region_masks, aggregation)
            for region, val in scores.items():
                rows.append({"date": ts.date(), "region": region, "method": "B2_multivar_zscore",
                             "aggregation": aggregation, "score": val})
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# B3: PCA reconstruction error (tiled, matched-k)
# ---------------------------------------------------------------------------

class PCATiler:
    """Wraps a fitted sklearn PCA as a Reconstructor for src/infer.reconstruct_frame."""

    def __init__(self, pca: IncrementalPCA, patch_size: int):
        self._pca = pca
        self._patch_size = patch_size

    def __call__(self, patches: np.ndarray) -> np.ndarray:
        """patches: (B, C, P, P) → (B, C, P, P) reconstructed."""
        B, C, P, _ = patches.shape
        flat = patches.reshape(B, -1)  # (B, C*P*P)
        proj = self._pca.transform(flat)
        recon_flat = self._pca.inverse_transform(proj)
        return recon_flat.reshape(B, C, P, P).astype(np.float32)


class PCAReconstruction:
    """B3 baseline for one value of k."""

    def __init__(self, k: int):
        self.k = k
        self._method_name = f"B3_pca_k{k}"

    def fit(self, cube: xr.Dataset, n_samples: int = 20_000) -> "PCAReconstruction":
        """
        Fit IncrementalPCA on random patches from the cube.
        n_samples: number of patches to use for fitting.
        """
        data = cube["data"].values   # (T, C, H, W)
        mask = cube["mask"].values   # (T, H, W)
        T, C, H, W = data.shape
        P = config.PATCH_SIZE

        rng = np.random.default_rng(config.SEED)
        flat_patches = []
        attempts = 0
        while len(flat_patches) < n_samples and attempts < n_samples * 10:
            t = rng.integers(0, T)
            r = rng.integers(0, max(H - P, 1))
            c_off = rng.integers(0, max(W - P, 1))
            vm_patch = mask[t, r:r + P, c_off:c_off + P]
            if vm_patch.mean() < config.MIN_VALID_FRACTION:
                attempts += 1
                continue
            patch = data[t, :, r:r + P, c_off:c_off + P].reshape(-1)  # (C*P*P)
            patch = np.nan_to_num(patch, nan=0.0)
            flat_patches.append(patch)
            attempts += 1

        flat = np.stack(flat_patches)
        print(f"B3 k={self.k}: fitting PCA on {len(flat)} patches…")
        self._pca = IncrementalPCA(n_components=self.k)
        self._pca.fit(flat)
        print(f"  explained variance ratio sum: {self._pca.explained_variance_ratio_.sum():.3f}")
        return self

    def score(
        self,
        cube: xr.Dataset,
        aggregation: str = "mean",
    ) -> pd.DataFrame:
        data = cube["data"].values
        mask = cube["mask"].values
        times = pd.DatetimeIndex(cube["data"].time.values)
        lat = cube["data"].lat.values
        lon = cube["data"].lon.values
        region_masks = build_region_masks(lat, lon)

        reconstructor = PCATiler(self._pca, config.PATCH_SIZE)
        rows = []
        for t_idx, ts in enumerate(times):
            frame = data[t_idx]   # (C, H, W)
            vm = mask[t_idx]      # (H, W)
            error_map = reconstruct_frame(
                frame, vm, reconstructor,
                patch_size=config.PATCH_SIZE,
                stride=config.PATCH_STRIDE,
                min_valid_fraction=config.MIN_VALID_FRACTION,
            )
            scores = aggregate_to_regions(error_map, vm, region_masks, aggregation)
            for region, val in scores.items():
                rows.append({"date": ts.date(), "region": region, "method": self._method_name,
                             "aggregation": aggregation, "score": val})
        return pd.DataFrame(rows)
