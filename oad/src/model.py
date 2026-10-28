"""
2D Convolutional Autoencoder for snapshot ocean-state anomaly detection.

Architecture (proposal §4):
  Encoder: 3 × stride-2 Conv2d → flatten → Linear bottleneck
  Decoder: Linear → reshape → 3 × ConvTranspose2d
  ~500K params at in_channels=4, latent_dim=32.

in_channels is a constructor parameter so channel-ablation experiments can
reuse the same class without subclassing.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvAE(nn.Module):
    """2D Conv autoencoder over (B, C, 64, 64) patches."""

    _ENCODED_H = 8
    _ENCODED_W = 8
    _ENCODED_C = 128

    def __init__(self, in_channels: int = 4, latent_dim: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        flat_dim = self._ENCODED_C * self._ENCODED_H * self._ENCODED_W

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.enc_fc = nn.Linear(flat_dim, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1),
            # No output activation — inputs are zero-mean standardized
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.enc_fc(h.flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_fc(z).view(-1, self._ENCODED_C, self._ENCODED_H, self._ENCODED_W)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    pred, target : (B, C, H, W)
    mask         : (B, 1, H, W) float — 1.0 valid, 0.0 invalid
    """
    sq = (pred - target) ** 2 * mask
    n = mask.sum() * pred.shape[1] + 1e-6
    return sq.sum() / n


def make_ae_reconstructor(model: ConvAE, device: torch.device):
    """Adapter so a trained ConvAE plugs into src.infer.reconstruct_frame."""
    model.eval()

    def reconstructor(patches):
        import numpy as np
        x = torch.from_numpy(patches.astype(np.float32)).to(device)
        with torch.no_grad():
            return model(x).cpu().numpy()
    return reconstructor
