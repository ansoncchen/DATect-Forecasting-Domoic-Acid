"""
Tiled inference engine — shared by the PCA baseline (B3) and the ConvAE.

Core operation: take a full-frame (C, H, W) tensor, tile it into 64×64 patches
with 50% overlap, run each patch through a callable reconstructor, accumulate
with overlap-averaging, return the per-pixel reconstruction error map (H, W).

Design notes
------------
- The linear bottleneck in the ConvAE fixes the expected patch size to 64×64.
  This function reflect-pads the frame to the next multiple of patch_stride so
  all patches are exactly 64×64. Padded pixels are discarded after accumulation.
- Patches with <50% valid pixels are still passed through the reconstructor
  (masked MSE handles them), but their squared errors are zeroed in the
  final error map using the valid mask.
- Overlap-average is implemented via a float32 weight accumulation buffer;
  dividing element-wise at the end is numerically equivalent to Hann-windowed
  overlap-add but without per-patch weighting (flat window, uniform weight 1.0).
"""
import numpy as np
from typing import Callable, Protocol


class Reconstructor(Protocol):
    """Anything that maps (B, C, H, W) ndarray → (B, C, H, W) ndarray."""
    def __call__(self, patches: np.ndarray) -> np.ndarray: ...


def _reflect_pad(arr: np.ndarray, ph: int, pw: int) -> tuple[np.ndarray, int, int]:
    """
    Reflect-pad (C, H, W) array so H and W are at least ph×pw tile-able.
    Returns (padded_array, pad_h, pad_w).
    """
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
    """
    Extract overlapping patches from a (C, H, W) frame.

    Returns
    -------
    offsets  : list of (row_start, col_start) for each patch (in padded coords)
    patches  : (N, C, patch_size, patch_size) float32 array
    pad_h    : number of rows added at bottom during reflect-padding
    pad_w    : number of cols added at right
    """
    frame = frame.astype(np.float32)
    C, H, W = frame.shape
    padded, pad_h, pad_w = _reflect_pad(frame, stride, stride)
    _, pH, pW = padded.shape

    offsets = []
    patch_list = []
    for r in range(0, pH - patch_size + 1, stride):
        for c in range(0, pW - patch_size + 1, stride):
            offsets.append((r, c))
            patch_list.append(padded[:, r:r + patch_size, c:c + patch_size])

    patches = np.stack(patch_list, axis=0)  # (N, C, P, P)
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
    """
    Overlap-average reconstructed patches back into a (C, H, W) frame.

    Parameters
    ----------
    offsets       : (row, col) offsets returned by tile_frame (padded coords)
    recon_patches : (N, C, patch_size, patch_size)
    original_H/W  : spatial dims BEFORE padding
    pad_h / pad_w : padding applied by tile_frame
    """
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

    Parameters
    ----------
    frame         : (C, H, W) standardized float32 array; NaN on invalid pixels
    valid_mask    : (H, W) bool — True where observations exist
    reconstructor : callable (B, C, P, P) → (B, C, P, P)
    min_valid_fraction : patches below this fraction have their errors zeroed

    Returns
    -------
    error_map : (H, W) float32 — per-pixel sum-of-squared-error across channels;
                NaN outside valid_mask.
    """
    C, H, W = frame.shape
    # Replace NaN with 0 for the forward pass (mask handles loss weighting)
    frame_filled = np.nan_to_num(frame, nan=0.0)

    offsets, patches, pad_h, pad_w = tile_frame(frame_filled, patch_size, stride)
    N = len(offsets)

    # Build per-patch valid fraction from valid_mask
    padded_mask = np.pad(valid_mask.astype(np.float32), ((0, pad_h), (0, pad_w)), mode="constant")
    patch_valid_fracs = np.array([
        padded_mask[r:r + patch_size, c:c + patch_size].mean()
        for r, c in offsets
    ])

    # Run reconstructor in batches
    recon_list = []
    for start in range(0, N, batch_size):
        batch = patches[start:start + batch_size]
        recon_list.append(reconstructor(batch))
    recon_patches = np.concatenate(recon_list, axis=0)  # (N, C, P, P)

    # Zero out patches below min_valid_fraction (don't contribute to error map)
    low_valid = patch_valid_fracs < min_valid_fraction
    recon_patches[low_valid] = patches[low_valid]  # zero error for these patches

    recon_frame = untile_frame(offsets, recon_patches, H, W, pad_h, pad_w, patch_size, stride)

    # Per-pixel squared error summed across channels
    sq_err = ((recon_frame - frame_filled) ** 2).sum(axis=0)  # (H, W)

    # Mask: NaN outside valid observations
    error_map = np.where(valid_mask, sq_err, np.nan)
    return error_map.astype(np.float32)
