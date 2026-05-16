"""
Run inference for AE checkpoints + baselines → regional scores parquet.

Auto-detects 2D vs 3D checkpoints by inspecting the saved 'variant' field.

Usage:
    python scripts/04_run_inference.py --all-ae --baselines-only --baselines-3d
    python scripts/04_run_inference.py --checkpoint models/ae_3d_l32_c4_t4_s42.pt
    python scripts/04_run_inference.py --baselines-only --pca-k 32 64

Output files:
    outputs/scores/{method}.parquet
    outputs/scores/all_scores.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.model import ConvAE, make_ae_reconstructor
from src.model3d import ConvAE3D, make_ae3d_reconstructor
from src.infer import reconstruct_frame
from src.infer3d import reconstruct_temporal_frame
from src.regions import build_region_masks, aggregate_to_regions
from src.baselines import (
    ChlaZScore, MultivarZScore, PCAReconstruction, TemporalPCAReconstruction,
)


def _select_device(force_cpu: bool = False) -> torch.device:
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


def _select_device_for_3d() -> torch.device:
    """MPS lacks ConvTranspose3D → fall back to CPU on Mac without CUDA."""
    needs_cpu = (not torch.cuda.is_available() and torch.backends.mps.is_available())
    return _select_device(force_cpu=needs_cpu)


# ---------------------------------------------------------------------------
# 2D AE inference
# ---------------------------------------------------------------------------

def run_ae2d_inference(ckpt: dict, cube: xr.Dataset, aggregation: str, ckpt_name: str) -> pd.DataFrame:
    in_ch = ckpt["in_channels"]
    latent_dim = ckpt["latent_dim"]
    channel_subset = ckpt.get("channel_subset")
    method_name = f"AE_2d_l{latent_dim}"
    if channel_subset and len(channel_subset) != len(config.CHANNEL_NAMES):
        method_name += "_" + "".join(c[:3] for c in channel_subset)

    device = _select_device()
    model = ConvAE(in_channels=in_ch, latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    reconstructor = make_ae_reconstructor(model, device)

    all_channels = list(cube.attrs["channels"])
    if channel_subset is not None:
        idx = [all_channels.index(ch) for ch in channel_subset]
        data = cube["data"].values[:, idx, :, :]
    else:
        data = cube["data"].values
    mask = cube["mask"].values
    times = pd.DatetimeIndex(cube["data"].time.values)
    region_masks = build_region_masks(cube["data"].lat.values, cube["data"].lon.values)

    rows = []
    for t_idx, ts in enumerate(times):
        err = reconstruct_frame(
            data[t_idx], mask[t_idx], reconstructor,
            patch_size=config.PATCH_SIZE, stride=config.PATCH_STRIDE,
            min_valid_fraction=config.MIN_VALID_FRACTION,
        )
        for region, val in aggregate_to_regions(err, mask[t_idx], region_masks, aggregation).items():
            rows.append({"date": ts.date(), "region": region, "method": method_name,
                         "aggregation": aggregation, "score": val, "checkpoint": ckpt_name})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3D AE inference
# ---------------------------------------------------------------------------

def run_ae3d_inference(ckpt: dict, cube: xr.Dataset, aggregation: str, ckpt_name: str) -> pd.DataFrame:
    in_ch = ckpt["in_channels"]
    latent_dim = ckpt["latent_dim"]
    T = ckpt["temporal_window"]
    channel_subset = ckpt.get("channel_subset")
    method_name = f"AE_3d_l{latent_dim}_t{T}"
    if channel_subset and len(channel_subset) != len(config.CHANNEL_NAMES):
        method_name += "_" + "".join(c[:3] for c in channel_subset)

    device = _select_device_for_3d()
    model = ConvAE3D(in_channels=in_ch, latent_dim=latent_dim, temporal_window=T).to(device)
    model.load_state_dict(ckpt["model_state"])
    reconstructor = make_ae3d_reconstructor(model, device)

    all_channels = list(cube.attrs["channels"])
    if channel_subset is not None:
        idx = [all_channels.index(ch) for ch in channel_subset]
        data = cube["data"].values[:, idx, :, :]
    else:
        data = cube["data"].values
    mask = cube["mask"].values
    times = pd.DatetimeIndex(cube["data"].time.values)
    region_masks = build_region_masks(cube["data"].lat.values, cube["data"].lon.values)

    rows = []
    for anchor in range(T - 1, data.shape[0]):
        ts = times[anchor]
        t_start = anchor - T + 1
        window = np.transpose(data[t_start:t_start + T], (1, 0, 2, 3))  # (C, T, H, W)
        window_mask = mask[t_start:t_start + T]
        anchor_mask = mask[anchor]
        err = reconstruct_temporal_frame(
            window, anchor_mask, window_mask, reconstructor,
            patch_size=config.PATCH_SIZE, stride=config.PATCH_STRIDE,
            min_valid_fraction=config.MIN_VALID_FRACTION,
        )
        for region, val in aggregate_to_regions(err, anchor_mask, region_masks, aggregation).items():
            rows.append({"date": ts.date(), "region": region, "method": method_name,
                         "aggregation": aggregation, "score": val, "checkpoint": ckpt_name})
    return pd.DataFrame(rows)


def run_ae_inference(ckpt_path: Path, cube: xr.Dataset, aggregation: str) -> pd.DataFrame:
    """Auto-dispatch based on checkpoint variant field."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    variant = ckpt.get("variant", "2d")  # legacy checkpoints default to 2D
    if variant == "3d":
        return run_ae3d_inference(ckpt, cube, aggregation, ckpt_path.name)
    return run_ae2d_inference(ckpt, cube, aggregation, ckpt_path.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Run inference for a single AE checkpoint")
    parser.add_argument("--all-ae", action="store_true",
                        help="Run all .pt files in models/")
    parser.add_argument("--baselines-only", action="store_true",
                        help="Run B1, B2, B3 snapshot baselines")
    parser.add_argument("--baselines-3d", action="store_true",
                        help="Run B3T temporal PCA baselines (matched-k for 3D AE)")
    parser.add_argument("--cube", default=str(config.CUBE_PATH))
    parser.add_argument("--aggregation", default="mean",
                        choices=["mean", "top_decile", "max"])
    parser.add_argument("--pca-k", type=int, nargs="+", default=config.PCA_K_SWEEP)
    parser.add_argument("--temporal-window", type=int, default=config.TEMPORAL_WINDOW)
    args = parser.parse_args()

    cube_path = Path(args.cube)
    if not cube_path.exists():
        print(f"ERROR: cube not found at {cube_path}"); sys.exit(1)

    config.SCORES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading cube: {cube_path}")
    cube = xr.open_zarr(cube_path, consolidated=True)

    all_dfs = []

    if args.baselines_only:
        print("\n=== B1: chl-a z-score ===")
        df = ChlaZScore().fit(cube).score(cube, args.aggregation)
        all_dfs.append(df); _save(df, "B1_chla_zscore")

        print("\n=== B2: multivar z-score ===")
        df = MultivarZScore().fit(cube).score(cube, args.aggregation)
        all_dfs.append(df); _save(df, "B2_multivar_zscore")

        for k in args.pca_k:
            print(f"\n=== B3: PCA k={k} ===")
            df = PCAReconstruction(k).fit(cube).score(cube, args.aggregation)
            all_dfs.append(df); _save(df, f"B3_pca_k{k}")

    if args.baselines_3d:
        for k in args.pca_k:
            print(f"\n=== B3T: Temporal PCA k={k} T={args.temporal_window} ===")
            df = TemporalPCAReconstruction(k, args.temporal_window).fit(cube).score(cube, args.aggregation)
            all_dfs.append(df); _save(df, f"B3T_pca_k{k}_t{args.temporal_window}")

    ckpts = []
    if args.checkpoint:
        ckpts = [Path(args.checkpoint)]
    elif args.all_ae:
        ckpts = sorted(config.MODELS_DIR.glob("ae_*.pt"))
        print(f"\nFound {len(ckpts)} AE checkpoints")

    for ckpt_path in ckpts:
        print(f"\n=== AE inference: {ckpt_path.name} ===")
        df = run_ae_inference(ckpt_path, cube, args.aggregation)
        all_dfs.append(df); _save(df, ckpt_path.stem)

    cube.close()

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        out = config.SCORES_DIR / "all_scores.parquet"
        combined.to_parquet(out, index=False)
        print(f"\nAll scores → {out}  ({len(combined)} rows)")


def _save(df: pd.DataFrame, name: str):
    path = config.SCORES_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"  Saved {len(df)} rows → {path}")


if __name__ == "__main__":
    main()
