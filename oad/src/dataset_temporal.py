"""
Temporal patch dataset for the 3D ConvAE (Phase A).

Produces samples shaped (C, T, P, P) with corresponding masks (1, T, P, P).

Convention: **lookback**. The anchor frame is the last index along T;
samples stack frames `[t-T+1, t-T+2, ..., t-1, t]` so that at inference we
only need past observations (matches the operational use case).

Patches are rejected if:
  - any frame in the window has insufficient valid pixels, OR
  - the window cannot fit (anchor t < T-1).

A coastal-overlap criterion (same as 2D path) is also enforced.

Fallback counter is included so we can detect winter cloud-cover starvation
the same way the 2D PatchDataset does (see src/train.py).
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class TemporalPatchDataset(Dataset):
    """
    Randomly samples (C, T, 64, 64) windows from a (T_total, C, H, W) cube.

    Anchors are restricted to frame_indices ∩ {t : t >= T-1} so the window
    fits in the past direction. Window = [t-T+1 ... t].
    """

    def __init__(
        self,
        data: np.ndarray,           # (T_total, C, H, W) standardized
        mask: np.ndarray,           # (T_total, H, W) bool
        anchor_indices: list[int],  # candidate anchor times (must be >= T-1)
        temporal_window: int = config.TEMPORAL_WINDOW,
        patch_size: int = config.PATCH_SIZE,
        min_valid_fraction: float = config.MIN_VALID_FRACTION,
        patches_per_epoch: int = config.PATCHES_PER_EPOCH,
        seed: int = config.SEED,
        coastal_mask: np.ndarray | None = None,
        coastal_min_overlap: float = 0.0,
        max_sampling_tries: int = 800,
    ):
        self.data = data
        self.mask = mask
        self.T_window = temporal_window
        # Keep only anchors that have full lookback
        self.anchors = [a for a in anchor_indices if a >= self.T_window - 1]
        if not self.anchors:
            raise ValueError(
                f"No valid anchors: need anchor_indices with t >= {self.T_window - 1}"
            )
        self.patch_size = patch_size
        self.min_valid_fraction = min_valid_fraction
        self.patches_per_epoch = patches_per_epoch
        self.rng = np.random.default_rng(seed)
        self.T_total, self.C, self.H, self.W = data.shape
        self.coastal_mask = coastal_mask
        self.coastal_min_overlap = coastal_min_overlap
        self.max_sampling_tries = max_sampling_tries

        self._fallback_count = 0
        self._total_requested = 0
        self._lock = threading.Lock()

    def reset_fallback_counter(self) -> None:
        with self._lock:
            self._fallback_count = 0
            self._total_requested = 0

    def fallback_rate(self) -> tuple[int, int]:
        with self._lock:
            return self._fallback_count, self._total_requested

    def __len__(self) -> int:
        return self.patches_per_epoch

    def __getitem__(self, _idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        P = self.patch_size
        T = self.T_window
        max_r = self.H - P
        max_c = self.W - P

        with self._lock:
            self._total_requested += 1

        for _ in range(self.max_sampling_tries):
            anchor = int(self.rng.choice(self.anchors))
            t_start = anchor - T + 1
            r = int(self.rng.integers(0, max(max_r, 1)))
            c = int(self.rng.integers(0, max(max_c, 1)))

            # Coastal overlap check (single mask, time-invariant)
            if (
                self.coastal_mask is not None
                and self.coastal_min_overlap > 0
                and self.coastal_mask[r:r + P, c:c + P].mean() < self.coastal_min_overlap
            ):
                continue

            window_mask = self.mask[t_start:t_start + T, r:r + P, c:c + P]  # (T, P, P)
            # Reject if ANY frame in the window is mostly invalid
            per_frame_valid = window_mask.reshape(T, -1).mean(axis=1)
            if per_frame_valid.min() < self.min_valid_fraction:
                continue

            window_data = self.data[t_start:t_start + T, :, r:r + P, c:c + P]  # (T, C, P, P)
            # Reshape to (C, T, P, P)
            patch = np.transpose(window_data, (1, 0, 2, 3)).astype(np.float32)
            patch = np.nan_to_num(patch, nan=0.0)
            mask_patch = window_mask.astype(np.float32)[np.newaxis]  # (1, T, P, P)
            return torch.from_numpy(patch), torch.from_numpy(mask_patch)

        # Fallback: all-zero patch (rare unless coast is fully cloud-occluded
        # across the entire T-window for every candidate anchor).
        with self._lock:
            self._fallback_count += 1
        return torch.zeros(self.C, T, P, P), torch.zeros(1, T, P, P)
