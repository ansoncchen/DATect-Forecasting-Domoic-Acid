"""
2-D tiled inference engine — shared by the PCA baseline (B3) and the 2D ConvAE.

Core operation: take a full-frame (C, H, W) tensor, tile it into 64×64 patches
with 50% overlap, run each patch through a callable reconstructor, accumulate
with overlap-averaging, return the per-pixel reconstruction error map (H, W).

The 3D analog lives in src/infer3d.py and reuses the same overlap-average
discipline along the spatial axes.
"""
from __future__ import annotations

from typing import Callable, Protocol

import numpy as np


class Reconstructor(Protocol):
    """Anything that maps (B, C, H, W) ndarray → (B, C, H, W) ndarray."""
    def __call__(self, patches: np.ndarray) -> np.ndarray: ...


def _reflect_pad(arr: np.ndarray, ph: int, pw: int) -> tuple[np.ndarray, int, int]:
    """Reflect-pad (C, H, W) so H, W are divisible by ph, pw. Returns (padded, pad_h, pad_w)."""
    C, H, W = arr.shape
    pad_h = (ph - H % ph) % ph if H % ph != 0 else 0
    pad_w = (pw - W % pw) % pw if W % pw != 0 else 0
    if pad_h == 0 and pad_w == 0:
        return arr, 0, 0
    padded = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    return padded, pad_h, pad_w


def tile_frame(
    frame: np.ndarray,
    patch_size: int = 64,
    stride: int = 32,
) -> tuple[list[tuple[int, int]], np.ndarray, int, int]:
    """Tile a (C, H, W) frame into overlapping (N, C, P, P) patches."""
    frame = frame.astype(np.float32)
    C, H, W = frame.shape
    padded, pad_h, pad_w = _reflect_pad(frame, stride, stride)
    _, pH, pW = padded.shape

    offsets, patch_list = [], []
    for r in range(0, pH - patch_size + 1, stride):
        for c in range(0, pW - patch_size + 1, stride):
            offsets.append((r, c))
            patch_list.append(padded[:, r:r + patch_size, c:c + patch_size])
    patches = np.stack(patch_list, axis=0)
    return offsets, patches, pad_h, pad_w


def untile_frame(
    offsets: list[tuple[int, int]],
    recon_patches: np.ndarray,
    original_H: int,
    original_W: int,
    pad_h: int,
    pad_w: int,
    patch_size: int = 64,
    stride: int = 32,
) -> np.ndarray:
    """Overlap-average reconstructed patches back into a (C, H, W) frame."""
    C = recon_patches.shape[1]
    pH = original_H + pad_h
    pW = original_W + pad_w
    accum = np.zeros((C, pH, pW), dtype=np.float64)
    count = np.zeros((pH, pW), dtype=np.float64)
    for (r, c), patch in zip(offsets, recon_patches):
        accum[:, r:r + patch_size, c:c + patch_size] += patch
        count[r:r + patch_size, c:c + patch_size] += 1.0
    count = np.maximum(count, 1.0)
    averaged = (accum / count).astype(np.float32)
    return averaged[:, :original_H, :original_W]


def reconstruct_frame(
    frame: np.ndarray,
    valid_mask: np.ndarray,
    reconstructor: Reconstructor,
    patch_size: int = 64,
    stride: int = 32,
    batch_size: int = 64,
    min_valid_fraction: float = 0.5,
) -> np.ndarray:
    """
    Run tiled reconstruction on a single (C, H, W) frame.

    Returns per-pixel summed-squared-error (H, W); NaN outside valid_mask.
    """
    C, H, W = frame.shape
    frame_filled = np.nan_to_num(frame, nan=0.0)
    offsets, patches, pad_h, pad_w = tile_frame(frame_filled, patch_size, stride)
    N = len(offsets)

    padded_mask = np.pad(valid_mask.astype(np.float32), ((0, pad_h), (0, pad_w)), mode="constant")
    patch_valid_fracs = np.array([
        padded_mask[r:r + patch_size, c:c + patch_size].mean()
        for r, c in offsets
    ])

    recon_list = []
    for start in range(0, N, batch_size):
        recon_list.append(reconstructor(patches[start:start + batch_size]))
    recon_patches = np.concatenate(recon_list, axis=0)

    low_valid = patch_valid_fracs < min_valid_fraction
    recon_patches[low_valid] = patches[low_valid]  # zero error contribution

    recon_frame = untile_frame(offsets, recon_patches, H, W, pad_h, pad_w, patch_size, stride)
    sq_err = ((recon_frame - frame_filled) ** 2).sum(axis=0)
    return np.where(valid_mask, sq_err, np.nan).astype(np.float32)
