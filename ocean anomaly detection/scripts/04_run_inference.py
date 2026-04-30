"""
Run inference for AE checkpoints and baselines → regional scores parquet.

Output: outputs/scores/{method}.parquet
        Each file has columns: date, region, method, aggregation, score

Usage:
    # AE inference from a single checkpoint
    python scripts/04_run_inference.py --checkpoint models/ae_l32_c4_s42.pt

    # Run all checkpoints in models/
    python scripts/04_run_inference.py --all-ae

    # Baselines only (no checkpoint needed)
    python scripts/04_run_inference.py --baselines-only

    # Everything
    python scripts/04_run_inference.py --all-ae --baselines-only
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
from src.infer import reconstruct_frame
from src.regions import build_region_masks, aggregate_to_regions
from src.baselines import ChlaZScore, MultivarZScore, PCAReconstruction


# ---------------------------------------------------------------------------
# AE inference
# ---------------------------------------------------------------------------

def run_ae_inference(ckpt_path: Path, cube: xr.Dataset, aggregation: str = "mean") -> pd.DataFrame:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    in_ch = ckpt["in_channels"]
    latent_dim = ckpt["latent_dim"]
    channel_subset = ckpt.get("channel_subset", None)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    model = ConvAE(in_channels=in_ch, latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    reconstructor = make_ae_reconstructor(model, device)

    all_channels = list(cube.attrs["channels"])
    if channel_subset is not None:
        ch_indices = [all_channels.index(ch) for ch in channel_subset]
        data = cube["data"].values[:, ch_indices, :, :]
        method_name = f"AE_l{latent_dim}_{'_'.join(channel_subset[:2])}"
    else:
        data = cube["data"].values
        method_name = f"AE_l{latent_dim}"

    mask = cube["mask"].values
    times = pd.DatetimeIndex(cube["data"].time.values)
    lat = cube["data"].lat.values
    lon = cube["data"].lon.values
    region_masks = build_region_masks(lat, lon)

    rows = []
    for t_idx, ts in enumerate(times):
        frame = data[t_idx]
        vm = mask[t_idx]
        error_map = reconstruct_frame(
            frame, vm, reconstructor,
            patch_size=config.PATCH_SIZE,
            stride=config.PATCH_STRIDE,
            min_valid_fraction=config.MIN_VALID_FRACTION,
        )
        scores = aggregate_to_regions(error_map, vm, region_masks, aggregation)
        for region, val in scores.items():
            rows.append({"date": ts.date(), "region": region, "method": method_name,
                         "aggregation": aggregation, "score": val,
                         "checkpoint": ckpt_path.name})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run inference and produce scores parquet")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a single AE checkpoint")
    parser.add_argument("--all-ae", action="store_true",
                        help="Run all .pt files found in models/")
    parser.add_argument("--baselines-only", action="store_true",
                        help="Only run B1, B2, B3 baselines")
    parser.add_argument("--cube", default=str(config.CUBE_PATH))
    parser.add_argument("--aggregation", default="mean",
                        choices=["mean", "top_decile", "max"])
    parser.add_argument("--pca-k", type=int, nargs="+", default=config.PCA_K_SWEEP,
                        help="PCA component counts to run as B3 baselines")
    args = parser.parse_args()

    cube_path = Path(args.cube)
    if not cube_path.exists():
        print(f"ERROR: cube not found at {cube_path}")
        sys.exit(1)

    config.SCORES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading cube: {cube_path}")
    cube = xr.open_zarr(cube_path, consolidated=True)

    all_dfs = []

    # -----------------------------------------------------------------------
    # Baselines
    # -----------------------------------------------------------------------
    if args.baselines_only or (not args.checkpoint and not args.all_ae):
        print("\n=== B1: chl-a z-score ===")
        b1 = ChlaZScore().fit(cube)
        df = b1.score(cube, args.aggregation)
        all_dfs.append(df)
        _save(df, "B1_chla_zscore")

        print("\n=== B2: multivar z-score ===")
        b2 = MultivarZScore().fit(cube)
        df = b2.score(cube, args.aggregation)
        all_dfs.append(df)
        _save(df, "B2_multivar_zscore")

        for k in args.pca_k:
            print(f"\n=== B3: PCA k={k} ===")
            b3 = PCAReconstruction(k).fit(cube)
            df = b3.score(cube, args.aggregation)
            all_dfs.append(df)
            _save(df, f"B3_pca_k{k}")

    # -----------------------------------------------------------------------
    # AE inference
    # -----------------------------------------------------------------------
    ckpts = []
    if args.checkpoint:
        ckpts = [Path(args.checkpoint)]
    elif args.all_ae:
        ckpts = sorted(config.MODELS_DIR.glob("ae_*.pt"))
        print(f"Found {len(ckpts)} AE checkpoints")

    for ckpt_path in ckpts:
        print(f"\n=== AE inference: {ckpt_path.name} ===")
        df = run_ae_inference(ckpt_path, cube, args.aggregation)
        all_dfs.append(df)
        stem = ckpt_path.stem
        _save(df, stem)

    cube.close()

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = config.SCORES_DIR / "all_scores.parquet"
        combined.to_parquet(combined_path, index=False)
        print(f"\nAll scores → {combined_path}  ({len(combined)} rows)")


def _save(df: pd.DataFrame, name: str) -> None:
    path = config.SCORES_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"  Saved {len(df)} rows → {path}")


if __name__ == "__main__":
    main()
