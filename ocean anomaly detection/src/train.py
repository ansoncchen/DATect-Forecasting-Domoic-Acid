"""
Training entry points for both the 2D ConvAE (snapshot) and 3D ConvAE3D (temporal).

Public functions:
    train(...)           — 2D snapshot ConvAE on (B, C, P, P) patches
    train_temporal(...)  — 3D ConvAE3D on (B, C, T, P, P) windows

Both share seed pinning, train/val split, optimizer/scheduler, checkpoint format,
and report the patch-sampler fallback rate per epoch (cloud-cover diagnostic).
"""
from __future__ import annotations

import random
import sys
import threading
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
from src.model3d import ConvAE3D, masked_mse_3d
from src.regions import overall_coastal_mask
from src.dataset_temporal import TemporalPatchDataset


# ---------------------------------------------------------------------------
# 2D snapshot dataset
# ---------------------------------------------------------------------------

class PatchDataset(Dataset):
    """Snapshot (B, C, P, P) patches with rejection sampling."""

    def __init__(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        frame_indices: list[int],
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
        self.frame_indices = frame_indices
        self.patch_size = patch_size
        self.min_valid_fraction = min_valid_fraction
        self.patches_per_epoch = patches_per_epoch
        self.rng = np.random.default_rng(seed)
        self.T, self.C, self.H, self.W = data.shape
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
        max_r = self.H - P
        max_c = self.W - P

        with self._lock:
            self._total_requested += 1

        for _ in range(self.max_sampling_tries):
            t = int(self.rng.choice(self.frame_indices))
            r = int(self.rng.integers(0, max(max_r, 1)))
            c = int(self.rng.integers(0, max(max_c, 1)))
            vm = self.mask[t, r:r + P, c:c + P]
            if vm.mean() < self.min_valid_fraction:
                continue
            if (
                self.coastal_mask is not None
                and self.coastal_min_overlap > 0
                and self.coastal_mask[r:r + P, c:c + P].mean() < self.coastal_min_overlap
            ):
                continue
            patch = self.data[t, :, r:r + P, c:c + P].astype(np.float32)
            patch = np.nan_to_num(patch, nan=0.0)
            mask_patch = vm.astype(np.float32)[np.newaxis]
            return torch.from_numpy(patch), torch.from_numpy(mask_patch)

        with self._lock:
            self._fallback_count += 1
        return torch.zeros(self.C, P, P), torch.zeros(1, P, P)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def split_frames(n_frames: int, seed: int) -> tuple[list[int], list[int], list[int]]:
    """Random 80/10/10 frame index split."""
    rng = np.random.default_rng(seed)
    idx = list(range(n_frames))
    rng.shuffle(idx)
    n_val = max(1, n_frames // 10)
    n_test = max(1, n_frames // 10)
    return idx[n_val + n_test:], idx[:n_val], idx[n_val:n_val + n_test]


def _pin_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_device(force_cpu: bool = False) -> torch.device:
    """
    Device priority: env DATECT_DEVICE > force_cpu > CUDA > MPS > CPU.

    force_cpu is used for ConvAE3D — MPS doesn't support ConvTranspose3D,
    so 3D training on Mac must run on CPU.
    """
    import os
    env_dev = os.environ.get("DATECT_DEVICE", "").lower()
    if env_dev in ("cpu", "cuda", "mps"):
        return torch.device(env_dev)
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_cube(cube_path: Path, channel_subset: list[str] | None):
    ds = xr.open_zarr(cube_path, consolidated=True)
    lat = ds["data"].lat.values
    lon = ds["data"].lon.values
    all_channels = list(ds.attrs["channels"])
    if channel_subset is not None:
        idx = [all_channels.index(ch) for ch in channel_subset]
        data = ds["data"].values[:, idx, :, :]
    else:
        data = ds["data"].values
    mask = ds["mask"].values
    ds.close()
    return data, mask, lat, lon


# ---------------------------------------------------------------------------
# 2D snapshot training
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
    coastal_patch_min_overlap: float | None = None,
    mask_ratio: float = 0.0,
) -> Path:
    """
    Train 2D ConvAE; returns best checkpoint path.

    mask_ratio > 0 enables Phase C MAE-style training (random pixel masking
    during training; loss on hidden-valid pixels only). Inference is unchanged.
    """
    _pin_seeds(seed)
    device = _select_device()
    overlap = (coastal_patch_min_overlap if coastal_patch_min_overlap is not None
               else config.TRAIN_COASTAL_PATCH_MIN_OVERLAP)
    print(f"[2D] Device: {device}  latent_dim={latent_dim}  in_channels={in_channels}  seed={seed}")
    if overlap > 0:
        print(f"  Patch sampling: coastal bbox, min overlap={overlap:.2f}")
    if mask_ratio > 0:
        print(f"  MAE-style: mask_ratio={mask_ratio:.2f}")

    data, mask, lat, lon = _load_cube(cube_path, channel_subset)
    coastal_mask = overall_coastal_mask(lat, lon) if overlap > 0 else None

    T = data.shape[0]
    train_idx, val_idx, _ = split_frames(T, seed)

    train_ds = PatchDataset(data, mask, train_idx,
                            patches_per_epoch=patches_per_epoch, seed=seed,
                            coastal_mask=coastal_mask, coastal_min_overlap=overlap)
    val_ds = PatchDataset(data, mask, val_idx,
                          patches_per_epoch=max(500, patches_per_epoch // 10),
                          seed=seed + 1,
                          coastal_mask=coastal_mask, coastal_min_overlap=overlap)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=(device.type == "cuda"))

    model = ConvAE(in_channels=in_channels, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    out_dir.mkdir(parents=True, exist_ok=True)
    label = f"2d_l{latent_dim}_c{in_channels}_s{seed}"
    if channel_subset:
        label += "_" + "".join(ch[:3] for ch in channel_subset)
    if mask_ratio > 0:
        # e.g. mae030 for ratio=0.30
        label += f"_mae{int(round(mask_ratio*100)):03d}"
    ckpt_path = out_dir / f"ae_{label}.pt"

    return _run_training_loop(
        model, optimizer, scheduler, train_loader, val_loader,
        train_ds, masked_mse,
        epochs=epochs, patience=patience, device=device,
        ckpt_path=ckpt_path,
        mask_ratio=mask_ratio,
        meta={"variant": "2d", "in_channels": in_channels, "latent_dim": latent_dim,
              "seed": seed, "channel_subset": channel_subset,
              "mask_ratio": mask_ratio},
    )


# ---------------------------------------------------------------------------
# 3D temporal training (Phase B)
# ---------------------------------------------------------------------------

def train_temporal(
    cube_path: Path,
    in_channels: int,
    latent_dim: int,
    temporal_window: int = config.TEMPORAL_WINDOW,
    mask_ratio: float = 0.0,
    seed: int = config.SEED,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    lr: float = config.LR,
    patience: int = config.EARLY_STOP_PATIENCE,
    patches_per_epoch: int = config.PATCHES_PER_EPOCH,
    out_dir: Path = config.MODELS_DIR,
    channel_subset: list[str] | None = None,
    coastal_patch_min_overlap: float | None = None,
) -> Path:
    """Train 3D ConvAE3D; returns best checkpoint path."""
    _pin_seeds(seed)
    # MPS doesn't support ConvTranspose3D; force CPU on Mac without CUDA.
    needs_cpu_fallback = (not torch.cuda.is_available()
                          and torch.backends.mps.is_available())
    device = _select_device(force_cpu=needs_cpu_fallback)
    if needs_cpu_fallback:
        print("  Note: 3D ConvAE on Mac → CPU (MPS lacks ConvTranspose3D); use Hyak CUDA for speed.")
    overlap = (coastal_patch_min_overlap if coastal_patch_min_overlap is not None
               else config.TRAIN_COASTAL_PATCH_MIN_OVERLAP)
    print(f"[3D] Device: {device}  latent_dim={latent_dim}  in_channels={in_channels}  "
          f"T={temporal_window}  seed={seed}")
    if mask_ratio > 0:
        print(f"  MAE-style: mask_ratio={mask_ratio:.2f}")
    if overlap > 0:
        print(f"  Patch sampling: coastal bbox, min overlap={overlap:.2f}")

    data, mask, lat, lon = _load_cube(cube_path, channel_subset)
    coastal_mask = overall_coastal_mask(lat, lon) if overlap > 0 else None

    T = data.shape[0]
    # 2D split shuffles frame indices for snapshot training. For temporal,
    # we use the same split but require anchors to have full lookback.
    train_idx, val_idx, _ = split_frames(T, seed)

    train_ds = TemporalPatchDataset(
        data, mask, train_idx,
        temporal_window=temporal_window,
        patches_per_epoch=patches_per_epoch, seed=seed,
        coastal_mask=coastal_mask, coastal_min_overlap=overlap,
    )
    val_ds = TemporalPatchDataset(
        data, mask, val_idx,
        temporal_window=temporal_window,
        patches_per_epoch=max(500, patches_per_epoch // 10),
        seed=seed + 1,
        coastal_mask=coastal_mask, coastal_min_overlap=overlap,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=(device.type == "cuda"))

    model = ConvAE3D(in_channels=in_channels, latent_dim=latent_dim,
                     temporal_window=temporal_window).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    out_dir.mkdir(parents=True, exist_ok=True)
    label = f"3d_l{latent_dim}_c{in_channels}_t{temporal_window}_s{seed}"
    if channel_subset:
        label += "_" + "".join(ch[:3] for ch in channel_subset)
    if mask_ratio > 0:
        label += f"_mae{int(round(mask_ratio*100)):03d}"
    ckpt_path = out_dir / f"ae_{label}.pt"

    return _run_training_loop(
        model, optimizer, scheduler, train_loader, val_loader,
        train_ds, masked_mse_3d,
        epochs=epochs, patience=patience, device=device,
        ckpt_path=ckpt_path,
        mask_ratio=mask_ratio,
        meta={"variant": "3d", "in_channels": in_channels, "latent_dim": latent_dim,
              "temporal_window": temporal_window, "seed": seed,
              "channel_subset": channel_subset, "mask_ratio": mask_ratio},
    )


# ---------------------------------------------------------------------------
# Shared training loop
# ---------------------------------------------------------------------------

def _run_training_loop(
    model, optimizer, scheduler, train_loader, val_loader,
    train_ds, loss_fn,
    *,
    epochs: int, patience: int, device: torch.device,
    ckpt_path: Path, meta: dict,
    mask_ratio: float = 0.0,   # Phase C: MAE-style augmentation
) -> Path:
    """
    If mask_ratio > 0, applies MAE-style training:
      1. Random hidden mask drawn fresh each batch (same per channel; broadcasts)
      2. Input zeroed at hidden positions (model sees corrupted patch)
      3. Loss computed ONLY on pixels that were originally valid AND hidden —
         model must reconstruct what it didn't get to see directly
    Inference is unchanged; this is purely a training regularizer.
    """
    best_val = float("inf")
    no_improve = 0
    train_losses, val_losses = [], []
    mae_on = mask_ratio > 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_ds.reset_fallback_counter()
        ep_loss = 0.0
        n_seen = 0
        for patches, masks in train_loader:
            patches, masks = patches.to(device), masks.to(device)
            optimizer.zero_grad()
            if mae_on:
                # hidden_mask shape matches `masks` (1 channel dim, broadcasts over C)
                # 1 = hidden, 0 = visible
                hidden = (torch.rand_like(masks) < mask_ratio).to(masks.dtype)
                visible_to_model = 1.0 - hidden                 # 1 where model sees data
                corrupted = patches * visible_to_model          # broadcasts (B,1,…) over channels
                recon = model(corrupted)
                # Score loss only where BOTH valid AND hidden — the model had to "fill in"
                loss_weight = masks * hidden
                loss = loss_fn(recon, patches, loss_weight)
            else:
                recon = model(patches)
                loss = loss_fn(recon, patches, masks)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * patches.size(0)
            n_seen += patches.size(0)
        ep_loss /= max(n_seen, 1)

        model.eval()
        val_loss = 0.0
        n_seen_val = 0
        with torch.no_grad():
            for patches, masks in val_loader:
                patches, masks = patches.to(device), masks.to(device)
                # Val loss: same protocol as training (apply MAE mask if mae_on),
                # so loss values are directly comparable to train loss.
                if mae_on:
                    hidden = (torch.rand_like(masks) < mask_ratio).to(masks.dtype)
                    corrupted = patches * (1.0 - hidden)
                    recon = model(corrupted)
                    loss_weight = masks * hidden
                    val_loss += loss_fn(recon, patches, loss_weight).item() * patches.size(0)
                else:
                    recon = model(patches)
                    val_loss += loss_fn(recon, patches, masks).item() * patches.size(0)
                n_seen_val += patches.size(0)
        val_loss /= max(n_seen_val, 1)

        scheduler.step()
        train_losses.append(ep_loss)
        val_losses.append(val_loss)

        fb, tot = train_ds.fallback_rate()
        fb_str = f"  fallback={fb}/{tot}" if fb > 0 else ""
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}/{epochs}  train={ep_loss:.5f}  val={val_loss:.5f}{fb_str}")
        elif fb > 0 and fb / max(tot, 1) > 0.05:
            print(f"  epoch {epoch:3d}  WARNING: high fallback rate "
                  f"{fb}/{tot} ({100*fb/tot:.1f}%)")

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "train_losses": train_losses,
                "val_losses": val_losses,
                **meta,
            }, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch} (best val={best_val:.5f})")
                break

    print(f"Best checkpoint: {ckpt_path}  val_loss={best_val:.5f}")
    return ckpt_path
