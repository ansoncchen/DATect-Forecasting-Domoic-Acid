"""
3-D tiled inference engine for the temporal ConvAE3D.

Scoring convention:
  For each *anchor* date t (with full T-frame lookback available), build a
  (C, T, H, W) tensor for the window [t-T+1, ..., t], tile it spatially
  (the temporal axis is kept whole at T=4), reconstruct each (C, T, P, P)
  patch, accumulate spatially with overlap-averaging into a full
  (C, T, H, W) reconstruction, then report the *per-pixel squared error
  at the anchor frame only* (last index along T), summed across channels.

This makes the 3D anomaly score directly comparable to the 2D score
(one scalar per (date, region)) while letting the temporal context inform
the reconstruction.

The 3D PCA baseline uses the same tiler with a PCA reconstructor that
operates on flattened (C*T*P*P) vectors.
"""
from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from src.infer import _reflect_pad  # reuse the 2D padding helper


class TemporalReconstructor(Protocol):
    """(B, C, T, P, P) ndarray → (B, C, T, P, P) ndarray."""
    def __call__(self, patches: np.ndarray) -> np.ndarray: ...


def tile_temporal_frame(
    window: np.ndarray,
    patch_size: int = 64,
    stride: int = 32,
) -> tuple[list[tuple[int, int]], np.ndarray, int, int]:
    """
    Tile a (C, T, H, W) window into (N, C, T, P, P) spatial patches.

    Only the spatial axes are tiled. The temporal axis is preserved whole.
    """
    window = window.astype(np.float32)
    C, T, H, W = window.shape

    # Treat (C*T) as the "channel" axis for the existing reflect-pad helper
    flat = window.reshape(C * T, H, W)
    padded_flat, pad_h, pad_w = _reflect_pad(flat, stride, stride)
    pH, pW = padded_flat.shape[1], padded_flat.shape[2]
    padded = padded_flat.reshape(C, T, pH, pW)

    offsets, patch_list = [], []
    for r in range(0, pH - patch_size + 1, stride):
        for c in range(0, pW - patch_size + 1, stride):
            offsets.append((r, c))
            patch_list.append(padded[:, :, r:r + patch_size, c:c + patch_size])
    patches = np.stack(patch_list, axis=0)  # (N, C, T, P, P)
    return offsets, patches, pad_h, pad_w


def untile_temporal_frame(
    offsets: list[tuple[int, int]],
    recon_patches: np.ndarray,    # (N, C, T, P, P)
    original_H: int,
    original_W: int,
    pad_h: int,
    pad_w: int,
    patch_size: int = 64,
    stride: int = 32,
) -> np.ndarray:
    """Overlap-average reconstructed (C, T, H, W) frame."""
    C, T = recon_patches.shape[1], recon_patches.shape[2]
    pH = original_H + pad_h
    pW = original_W + pad_w
    accum = np.zeros((C, T, pH, pW), dtype=np.float64)
    count = np.zeros((pH, pW), dtype=np.float64)
    for (r, c), patch in zip(offsets, recon_patches):
        accum[:, :, r:r + patch_size, c:c + patch_size] += patch
        count[r:r + patch_size, c:c + patch_size] += 1.0
    count = np.maximum(count, 1.0)
    averaged = (accum / count[np.newaxis, np.newaxis]).astype(np.float32)
    return averaged[:, :, :original_H, :original_W]


def reconstruct_temporal_frame(
    window: np.ndarray,            # (C, T, H, W)
    anchor_valid_mask: np.ndarray, # (H, W)  — mask of the anchor frame
    window_valid_mask: np.ndarray, # (T, H, W) — full per-frame masks
    reconstructor: TemporalReconstructor,
    patch_size: int = 64,
    stride: int = 32,
    batch_size: int = 32,
    min_valid_fraction: float = 0.5,
) -> np.ndarray:
    """
    Run tiled 3D reconstruction and return per-pixel squared error on the
    *anchor frame* (last index along T), summed across channels.

    Returns (H, W) float32 array; NaN outside anchor_valid_mask.
    """
    C, T, H, W = window.shape
    window_filled = np.nan_to_num(window, nan=0.0)
    offsets, patches, pad_h, pad_w = tile_temporal_frame(window_filled, patch_size, stride)
    N = len(offsets)

    # Per-patch validity: require minimum coverage in the anchor frame
    anchor_mask = anchor_valid_mask.astype(np.float32)
    padded_anchor = np.pad(anchor_mask, ((0, pad_h), (0, pad_w)), mode="constant")
    patch_valid_fracs = np.array([
        padded_anchor[r:r + patch_size, c:c + patch_size].mean()
        for r, c in offsets
    ])

    recon_list = []
    for start in range(0, N, batch_size):
        recon_list.append(reconstructor(patches[start:start + batch_size]))
    recon_patches = np.concatenate(recon_list, axis=0)

    low_valid = patch_valid_fracs < min_valid_fraction
    recon_patches[low_valid] = patches[low_valid]  # zero error contribution

    recon_window = untile_temporal_frame(
        offsets, recon_patches, H, W, pad_h, pad_w, patch_size, stride
    )

    # Squared error on the anchor frame only (last index along T)
    anchor_recon = recon_window[:, -1, :, :]    # (C, H, W)
    anchor_input = window_filled[:, -1, :, :]   # (C, H, W)
    sq = ((anchor_recon - anchor_input) ** 2).sum(axis=0)  # (H, W)

    return np.where(anchor_valid_mask, sq, np.nan).astype(np.float32)
