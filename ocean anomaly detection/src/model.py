"""
Convolutional Autoencoder for multivariate ocean-state anomaly detection.

Architecture (from proposal §4):
  Encoder: 3 × stride-2 Conv2d → flatten → Linear bottleneck
  Decoder: Linear → reshape → 3 × ConvTranspose2d
  ~500K params at in_channels=4, latent_dim=32.

in_channels is a constructor parameter so channel-ablation experiments can
reuse the same class without subclassing.

Masked MSE loss weights errors by a per-pixel valid mask, so cloud/land gaps
contribute zero to both the numerator and denominator.
"""
import torch
import torch.nn as nn


class ConvAE(nn.Module):
    """
    Parameters
    ----------
    in_channels : number of satellite channels (4 = chla/k490/nflh/sst, or fewer for ablation)
    latent_dim  : bottleneck dimensionality
    """

    # Spatial size of the encoded feature map before flattening, given 64×64 input
    _ENCODED_H = 8
    _ENCODED_W = 8
    _ENCODED_C = 128  # channels after 3 stride-2 convolutions

    def __init__(self, in_channels: int = 4, latent_dim: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        flat_dim = self._ENCODED_C * self._ENCODED_H * self._ENCODED_W  # 8192

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.enc_fc = nn.Linear(flat_dim, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
            # No activation — inputs are standardized to zero mean, linear output correct
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, 64, 64) → (B, latent_dim)"""
        h = self.encoder(x)
        return self.enc_fc(h.flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B, latent_dim) → (B, C, 64, 64)"""
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
    Per-proposal §4 masked MSE loss.

    Parameters
    ----------
    pred   : (B, C, H, W)
    target : (B, C, H, W)
    mask   : (B, 1, H, W) float — 1.0 for valid pixels, 0.0 for cloud/land

    Returns a scalar loss.
    """
    sq_err = (pred - target) ** 2 * mask
    n_valid = mask.sum() * pred.shape[1] + 1e-6  # channels × valid pixels
    return sq_err.sum() / n_valid


def make_ae_reconstructor(model: ConvAE, device: torch.device):
    """
    Returns a Reconstructor callable (ndarray → ndarray) suitable for
    src.infer.reconstruct_frame.
    """
    model.eval()

    def reconstructor(patches: "np.ndarray") -> "np.ndarray":
        import numpy as np
        x = torch.from_numpy(patches.astype(np.float32)).to(device)
        with torch.no_grad():
            recon = model(x)
        return recon.cpu().numpy()

    return reconstructor
