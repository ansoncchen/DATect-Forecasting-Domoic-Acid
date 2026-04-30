"""
Patch-based training loop for the ConvAE.

Seeds are pinned at the top of every run and saved into the checkpoint so
experiments are exactly reproducible from the checkpoint filename alone.

Checkpoint format (saved via torch.save):
    {
        "model_state": state_dict,
        "in_channels": int,
        "latent_dim": int,
        "seed": int,
        "epoch": int,
        "val_loss": float,
        "train_losses": list[float],
        "val_losses": list[float],
    }
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import xarray as xr
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.model import ConvAE, masked_mse


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PatchDataset(Dataset):
    """
    Randomly samples 64×64 patches from the pre-loaded cube array.
    Rejected patches (< min_valid_fraction) are re-sampled until a valid one is found.
    """

    def __init__(
        self,
        data: np.ndarray,   # (T, C, H, W)
        mask: np.ndarray,   # (T, H, W)
        frame_indices: list[int],
        patch_size: int = config.PATCH_SIZE,
        min_valid_fraction: float = config.MIN_VALID_FRACTION,
        patches_per_epoch: int = config.PATCHES_PER_EPOCH,
        seed: int = config.SEED,
    ):
        self.data = data
        self.mask = mask
        self.frame_indices = frame_indices
        self.patch_size = patch_size
        self.min_valid_fraction = min_valid_fraction
        self.patches_per_epoch = patches_per_epoch
        self.rng = np.random.default_rng(seed)
        self.T, self.C, self.H, self.W = data.shape

    def __len__(self) -> int:
        return self.patches_per_epoch

    def __getitem__(self, _idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        P = self.patch_size
        max_r = self.H - P
        max_c = self.W - P

        for _ in range(200):  # rejection sampling with safety limit
            t = self.rng.choice(self.frame_indices)
            r = self.rng.integers(0, max(max_r, 1))
            c = self.rng.integers(0, max(max_c, 1))
            vm = self.mask[t, r:r + P, c:c + P]
            if vm.mean() < self.min_valid_fraction:
                continue
            patch = self.data[t, :, r:r + P, c:c + P].astype(np.float32)
            patch = np.nan_to_num(patch, nan=0.0)
            mask_patch = vm.astype(np.float32)[np.newaxis]  # (1, P, P)
            return torch.from_numpy(patch), torch.from_numpy(mask_patch)

        # Fallback: return zeros (very rare; only if frame is almost entirely masked)
        P = self.patch_size
        return torch.zeros(self.C, P, P), torch.zeros(1, P, P)


# ---------------------------------------------------------------------------
# Train/val split of frames (random 80/10/10 by frame index)
# ---------------------------------------------------------------------------

def split_frames(n_frames: int, seed: int) -> tuple[list[int], list[int], list[int]]:
    rng = np.random.default_rng(seed)
    idx = list(range(n_frames))
    rng.shuffle(idx)
    n_val = max(1, n_frames // 10)
    n_test = max(1, n_frames // 10)
    return idx[n_val + n_test:], idx[:n_val], idx[n_val:n_val + n_test]


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    cube_path: Path,
    in_channels: int,
    latent_dim: int,
    seed: int = config.SEED,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    lr: float = config.LR,
    patience: int = config.EARLY_STOP_PATIENCE,
    patches_per_epoch: int = config.PATCHES_PER_EPOCH,
    out_dir: Path = config.MODELS_DIR,
    channel_subset: list[str] | None = None,
) -> Path:
    """
    Train the ConvAE and return the path to the best checkpoint.

    channel_subset: if given, only these channel names are used (for ablations).
                    Must be a subset of config.CHANNEL_NAMES.
    """
    # Pin all seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}  latent_dim={latent_dim}  in_channels={in_channels}  seed={seed}")

    # -----------------------------------------------------------------------
    # Load cube
    # -----------------------------------------------------------------------
    ds = xr.open_zarr(cube_path, consolidated=True)
    all_channels = list(ds.attrs["channels"])
    if channel_subset is not None:
        ch_indices = [all_channels.index(ch) for ch in channel_subset]
        data = ds["data"].values[:, ch_indices, :, :]
    else:
        data = ds["data"].values   # (T, C, H, W)
    mask = ds["mask"].values       # (T, H, W)
    ds.close()

    T = data.shape[0]
    train_idx, val_idx, _ = split_frames(T, seed)

    train_ds = PatchDataset(data, mask, train_idx, patches_per_epoch=patches_per_epoch, seed=seed)
    val_ds = PatchDataset(data, mask, val_idx, patches_per_epoch=max(500, patches_per_epoch // 10), seed=seed + 1)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=(device.type == "cuda"))

    # -----------------------------------------------------------------------
    # Model, optimizer, scheduler
    # -----------------------------------------------------------------------
    model = ConvAE(in_channels=in_channels, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    label = f"l{latent_dim}_c{in_channels}_s{seed}"
    if channel_subset:
        label += "_" + "".join(ch[:3] for ch in channel_subset)
    ckpt_path = out_dir / f"ae_{label}.pt"

    best_val_loss = float("inf")
    no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        ep_loss = 0.0
        for patches, masks in train_loader:
            patches, masks = patches.to(device), masks.to(device)
            optimizer.zero_grad()
            recon = model(patches)
            loss = masked_mse(recon, patches, masks)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * patches.size(0)
        ep_loss /= len(train_loader.dataset)

        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for patches, masks in val_loader:
                patches, masks = patches.to(device), masks.to(device)
                recon = model(patches)
                val_loss += masked_mse(recon, patches, masks).item() * patches.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step()
        train_losses.append(ep_loss)
        val_losses.append(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}/{epochs}  train={ep_loss:.5f}  val={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "in_channels": in_channels,
                "latent_dim": latent_dim,
                "seed": seed,
                "epoch": epoch,
                "val_loss": val_loss,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "channel_subset": channel_subset,
            }, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch} (best val={best_val_loss:.5f})")
                break

    print(f"Best checkpoint: {ckpt_path}  val_loss={best_val_loss:.5f}")
    return ckpt_path
