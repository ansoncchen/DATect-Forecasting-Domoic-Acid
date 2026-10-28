"""
3D Convolutional Autoencoder for spatiotemporal anomaly detection (Phase B).

Architecture choice:
- True Conv3d encoder/decoder over (B, C, T, H, W) — proposal's preferred design.
- Stride-2 in time as well as space, so for T=4 the temporal axis collapses:
  4 → 2 → 1 in 2 stride-2 ops. The third spatial stride keeps no temporal stride.
  Bottleneck flat dim stays = 128 × 1 × 8 × 8 = 8192, identical to 2D AE.
- This keeps memory comparable to the 2D AE while letting the model learn
  short-range temporal dynamics (~3 weeks at 8-day cadence).

The reconstruction error at *anchor frame only* (the last index along T) is
the anomaly score for that date — matches what the 2D AE produces and lets
the same evaluation pipeline consume both.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvAE3D(nn.Module):
    """
    3D Conv autoencoder over (B, C, T, H, W) patches.

    With T=4 and 2 stride-2 time-down ops, encoded shape is (B, 128, 1, 8, 8).
    """

    _ENCODED_C = 128
    _ENCODED_T = 1   # T=4 → 2 → 1 after 2 stride-2 time-downs
    _ENCODED_H = 8
    _ENCODED_W = 8

    def __init__(self, in_channels: int = 4, latent_dim: int = 32, temporal_window: int = 4):
        super().__init__()
        if temporal_window != 4:
            # The architecture is parameterized but the stride pattern below
            # collapses exactly T=4 → 1. For other T, adjust strides accordingly.
            raise NotImplementedError(
                f"ConvAE3D currently hardcoded for temporal_window=4 (got {temporal_window}). "
                "Edit the encoder/decoder stride tuples to support other windows."
            )
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.temporal_window = temporal_window
        flat_dim = self._ENCODED_C * self._ENCODED_T * self._ENCODED_H * self._ENCODED_W

        # Stride tuples: (T, H, W)
        # Block 1: stride (2,2,2)  T 4→2, HW 64→32
        # Block 2: stride (2,2,2)  T 2→1, HW 32→16
        # Block 3: stride (1,2,2)  T stays 1, HW 16→8
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
        )
        self.enc_fc = nn.Linear(flat_dim, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, flat_dim)
        # Mirror the encoder stride pattern, reversed
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, in_channels, kernel_size=4, stride=(2, 2, 2), padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.enc_fc(h.flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_fc(z).view(-1, self._ENCODED_C, self._ENCODED_T, self._ENCODED_H, self._ENCODED_W)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def masked_mse_3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Masked MSE generalized to 5-D tensors.

    pred, target : (B, C, T, H, W)
    mask         : (B, 1, T, H, W) float — 1.0 valid, 0.0 invalid
    """
    sq = (pred - target) ** 2 * mask
    n = mask.sum() * pred.shape[1] + 1e-6
    return sq.sum() / n


def make_ae3d_reconstructor(model: ConvAE3D, device: torch.device):
    """Adapter so a trained ConvAE3D plugs into src.infer3d.reconstruct_temporal_frame."""
    model.eval()

    def reconstructor(patches):
        import numpy as np
        x = torch.from_numpy(patches.astype(np.float32)).to(device)
        with torch.no_grad():
            return model(x).cpu().numpy()
    return reconstructor
