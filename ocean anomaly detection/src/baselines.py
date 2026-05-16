"""
Baselines for comparison with the ConvAE anomaly metric.

B1 — single-channel chl-a z-score (climatological DOY)
B2 — multivariate climatological z-score (all 4 channels)
B3 — linear PCA reconstruction error (matched-k sweep, snapshot patches)
B3T — TEMPORAL PCA reconstruction error (matched-k sweep on (C*T)*P*P vectors)
      Required for a fair Phase B E4 comparison against the 3D ConvAE.

All baselines expose .fit(cube) → self and .score(cube) → DataFrame.
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
from src.regions import build_region_masks, aggregate_to_regions, overall_coastal_mask
from src.infer import reconstruct_frame
from src.infer3d import reconstruct_temporal_frame


# ---------------------------------------------------------------------------
# DOY climatology helpers
# ---------------------------------------------------------------------------

def _doy_climatology(
    da: xr.DataArray,
    smooth_steps: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-pixel DOY climatology (mean, std) for an 8-day composite product.
    smooth_steps: number of adjacent 8-day composites included on each side
                  (±2 → ±16 days window).
    """
    vals = da.values
    times = pd.DatetimeIndex(da.time.values)
    doys = times.dayofyear

    H, W = vals.shape[1], vals.shape[2]
    clim_mean = np.full((366, H, W), np.nan, dtype=np.float32)
    clim_std = np.full((366, H, W), np.nan, dtype=np.float32)

    for d in range(1, 367):
        diff = np.abs(doys - d)
        idx = np.where(diff <= smooth_steps * 8)[0]
        if len(idx) == 0:
            continue
        slab = vals[idx]
        clim_mean[d - 1] = np.nanmean(slab, axis=0)
        clim_std[d - 1] = np.nanstd(slab, axis=0)
    return clim_mean, clim_std


def _zscore_frame(frame, doy, clim_mean, clim_std):
    mu = clim_mean[doy - 1]
    sigma = clim_std[doy - 1]
    return (frame - mu) / np.where(sigma > 1e-8, sigma, np.nan)


# ---------------------------------------------------------------------------
# B1: Single-channel chl-a z-score
# ---------------------------------------------------------------------------

class ChlaZScore:
    def fit(self, cube: xr.Dataset) -> "ChlaZScore":
        ch_idx = list(cube.attrs["channels"]).index("chla")
        da = cube["data"].isel(channel=ch_idx)
        print("B1: computing chl-a DOY climatology…")
        self._mean, self._std = _doy_climatology(da)
        return self

    def score(self, cube: xr.Dataset, aggregation: str = "mean") -> pd.DataFrame:
        ch_idx = list(cube.attrs["channels"]).index("chla")
        data = cube["data"].values
        mask = cube["mask"].values
        times = pd.DatetimeIndex(cube["data"].time.values)
        region_masks = build_region_masks(cube["data"].lat.values, cube["data"].lon.values)

        rows = []
        for t_idx, ts in enumerate(times):
            frame = data[t_idx, ch_idx]
            vm = mask[t_idx]
            z = _zscore_frame(frame, ts.dayofyear, self._mean, self._std)
            err = np.where(vm, z ** 2, np.nan)
            for region, val in aggregate_to_regions(err, vm, region_masks, aggregation).items():
                rows.append({"date": ts.date(), "region": region, "method": "B1_chla_zscore",
                             "aggregation": aggregation, "score": val})
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# B2: Multivariate climatological z-score
# ---------------------------------------------------------------------------

class MultivarZScore:
    def fit(self, cube: xr.Dataset) -> "MultivarZScore":
        channels = list(cube.attrs["channels"])
        self._clim = {}
        for i, ch in enumerate(channels):
            print(f"B2: computing DOY climatology for {ch}…")
            self._clim[ch] = _doy_climatology(cube["data"].isel(channel=i))
        self._channels = channels
        return self

    def score(self, cube: xr.Dataset, aggregation: str = "mean") -> pd.DataFrame:
        data = cube["data"].values
        mask = cube["mask"].values
        times = pd.DatetimeIndex(cube["data"].time.values)
        region_masks = build_region_masks(cube["data"].lat.values, cube["data"].lon.values)

        rows = []
        for t_idx, ts in enumerate(times):
            vm = mask[t_idx]
            sum_sq = np.zeros(vm.shape, dtype=np.float32)
            for i, ch in enumerate(self._channels):
                z = _zscore_frame(data[t_idx, i], ts.dayofyear, *self._clim[ch])
                sum_sq += np.nan_to_num(z ** 2, nan=0.0)
            err = np.where(vm, sum_sq, np.nan)
            for region, val in aggregate_to_regions(err, vm, region_masks, aggregation).items():
                rows.append({"date": ts.date(), "region": region, "method": "B2_multivar_zscore",
                             "aggregation": aggregation, "score": val})
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# B3: Snapshot PCA reconstruction error (matched-k for 2D AE comparison)
# ---------------------------------------------------------------------------

class PCATiler:
    def __init__(self, pca: IncrementalPCA): self._pca = pca

    def __call__(self, patches: np.ndarray) -> np.ndarray:
        B, C, P, _ = patches.shape
        flat = patches.reshape(B, -1)
        proj = self._pca.transform(flat)
        recon = self._pca.inverse_transform(proj)
        return recon.reshape(B, C, P, P).astype(np.float32)


class PCAReconstruction:
    """B3 baseline for one value of k."""

    def __init__(self, k: int):
        self.k = k
        self._method_name = f"B3_pca_k{k}"

    def fit(self, cube: xr.Dataset, n_samples: int = 20_000,
            coastal_patch_min_overlap: float | None = None) -> "PCAReconstruction":
        data = cube["data"].values
        mask = cube["mask"].values
        T, C, H, W = data.shape
        P = config.PATCH_SIZE
        overlap = (coastal_patch_min_overlap if coastal_patch_min_overlap is not None
                   else config.TRAIN_COASTAL_PATCH_MIN_OVERLAP)
        coastal_mask = (overall_coastal_mask(cube["data"].lat.values, cube["data"].lon.values)
                        if overlap > 0 else None)

        rng = np.random.default_rng(config.SEED)
        flat_patches = []
        attempts = 0
        while len(flat_patches) < n_samples and attempts < n_samples * 20:
            t = rng.integers(0, T)
            r = rng.integers(0, max(H - P, 1))
            c = rng.integers(0, max(W - P, 1))
            attempts += 1
            if coastal_mask is not None and overlap > 0:
                if coastal_mask[r:r+P, c:c+P].mean() < overlap:
                    continue
            vm = mask[t, r:r+P, c:c+P]
            if vm.mean() < config.MIN_VALID_FRACTION:
                continue
            patch = data[t, :, r:r+P, c:c+P].reshape(-1)
            flat_patches.append(np.nan_to_num(patch, nan=0.0))

        flat = np.stack(flat_patches)
        print(f"B3 k={self.k}: fitting PCA on {len(flat)} patches…")
        self._pca = IncrementalPCA(n_components=self.k)
        self._pca.fit(flat)
        print(f"  explained variance ratio sum: {self._pca.explained_variance_ratio_.sum():.3f}")
        return self

    def score(self, cube: xr.Dataset, aggregation: str = "mean") -> pd.DataFrame:
        data = cube["data"].values
        mask = cube["mask"].values
        times = pd.DatetimeIndex(cube["data"].time.values)
        region_masks = build_region_masks(cube["data"].lat.values, cube["data"].lon.values)
        reconstructor = PCATiler(self._pca)

        rows = []
        for t_idx, ts in enumerate(times):
            err = reconstruct_frame(
                data[t_idx], mask[t_idx], reconstructor,
                patch_size=config.PATCH_SIZE, stride=config.PATCH_STRIDE,
                min_valid_fraction=config.MIN_VALID_FRACTION,
            )
            for region, val in aggregate_to_regions(err, mask[t_idx], region_masks, aggregation).items():
                rows.append({"date": ts.date(), "region": region, "method": self._method_name,
                             "aggregation": aggregation, "score": val})
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# B3T: Temporal PCA reconstruction error (Phase B — for fair 3D-AE comparison)
# ---------------------------------------------------------------------------

class TemporalPCATiler:
    def __init__(self, pca: IncrementalPCA): self._pca = pca

    def __call__(self, patches: np.ndarray) -> np.ndarray:
        """(B, C, T, P, P) → (B, C, T, P, P)"""
        B, C, T, P, _ = patches.shape
        flat = patches.reshape(B, -1)
        proj = self._pca.transform(flat)
        recon = self._pca.inverse_transform(proj)
        return recon.reshape(B, C, T, P, P).astype(np.float32)


class TemporalPCAReconstruction:
    """
    PCA fit on (C, T, P, P) → flat (C*T*P*P) vectors.

    Anchor convention matches src/infer3d.py: score at date t comes from
    reconstructing the window [t-T+1, ..., t] and reporting error on anchor.
    """

    def __init__(self, k: int, temporal_window: int = config.TEMPORAL_WINDOW):
        self.k = k
        self.T_window = temporal_window
        self._method_name = f"B3T_pca_k{k}_t{temporal_window}"

    def fit(self, cube: xr.Dataset, n_samples: int = 10_000,
            coastal_patch_min_overlap: float | None = None) -> "TemporalPCAReconstruction":
        data = cube["data"].values
        mask = cube["mask"].values
        T_total, C, H, W = data.shape
        P = config.PATCH_SIZE
        T = self.T_window
        overlap = (coastal_patch_min_overlap if coastal_patch_min_overlap is not None
                   else config.TRAIN_COASTAL_PATCH_MIN_OVERLAP)
        coastal_mask = (overall_coastal_mask(cube["data"].lat.values, cube["data"].lon.values)
                        if overlap > 0 else None)

        rng = np.random.default_rng(config.SEED)
        flat_patches = []
        attempts = 0
        max_attempts = n_samples * 30
        while len(flat_patches) < n_samples and attempts < max_attempts:
            attempts += 1
            anchor = rng.integers(T - 1, T_total)
            t_start = anchor - T + 1
            r = rng.integers(0, max(H - P, 1))
            c = rng.integers(0, max(W - P, 1))
            if coastal_mask is not None and overlap > 0:
                if coastal_mask[r:r+P, c:c+P].mean() < overlap:
                    continue
            vm = mask[t_start:t_start+T, r:r+P, c:c+P]
            if vm.reshape(T, -1).mean(axis=1).min() < config.MIN_VALID_FRACTION:
                continue
            window = data[t_start:t_start+T, :, r:r+P, c:c+P]   # (T, C, P, P)
            flat = np.transpose(window, (1, 0, 2, 3)).reshape(-1)  # (C*T*P*P)
            flat_patches.append(np.nan_to_num(flat, nan=0.0))

        flat = np.stack(flat_patches)
        print(f"B3T k={self.k} T={self.T_window}: fitting PCA on {len(flat)} windows…")
        self._pca = IncrementalPCA(n_components=self.k)
        self._pca.fit(flat)
        print(f"  explained variance ratio sum: {self._pca.explained_variance_ratio_.sum():.3f}")
        return self

    def score(self, cube: xr.Dataset, aggregation: str = "mean") -> pd.DataFrame:
        data = cube["data"].values
        mask = cube["mask"].values
        times = pd.DatetimeIndex(cube["data"].time.values)
        region_masks = build_region_masks(cube["data"].lat.values, cube["data"].lon.values)
        reconstructor = TemporalPCATiler(self._pca)
        T = self.T_window

        rows = []
        for anchor in range(T - 1, data.shape[0]):
            ts = times[anchor]
            t_start = anchor - T + 1
            window = np.transpose(data[t_start:t_start+T], (1, 0, 2, 3))  # (C, T, H, W)
            window_mask = mask[t_start:t_start+T]                          # (T, H, W)
            anchor_mask = mask[anchor]                                     # (H, W)
            err = reconstruct_temporal_frame(
                window, anchor_mask, window_mask, reconstructor,
                patch_size=config.PATCH_SIZE, stride=config.PATCH_STRIDE,
                min_valid_fraction=config.MIN_VALID_FRACTION,
            )
            for region, val in aggregate_to_regions(err, anchor_mask, region_masks, aggregation).items():
                rows.append({"date": ts.date(), "region": region, "method": self._method_name,
                             "aggregation": aggregation, "score": val})
        return pd.DataFrame(rows)
