"""
CLI driver for ConvAE training: 2D snapshot, 3D temporal, sweeps, ablations.

Usage:
    # Single 2D run (default)
    python scripts/03_train_ae.py --latent 32

    # Single 3D run (Phase B)
    python scripts/03_train_ae.py --temporal --latent 32

    # 2D bottleneck sweep
    python scripts/03_train_ae.py --sweep

    # 3D bottleneck sweep
    python scripts/03_train_ae.py --temporal --sweep

    # Channel ablations (works for both 2D and 3D)
    python scripts/03_train_ae.py --ablate-channels
    python scripts/03_train_ae.py --temporal --ablate-channels

    # Quick smoke-test
    python scripts/03_train_ae.py --latent 32 --epochs 5 --debug
    python scripts/03_train_ae.py --temporal --latent 32 --epochs 5 --debug

Checkpoint naming:
    models/ae_2d_l{latent}_c{nch}_s{seed}[_subset].pt
    models/ae_3d_l{latent}_c{nch}_t{T}_s{seed}[_subset].pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.train import train, train_temporal


ABLATION_SETS: list[tuple[str, list[str]]] = [
    ("no_chla",   [c for c in config.CHANNEL_NAMES if c != "chla"]),
    ("no_k490",   [c for c in config.CHANNEL_NAMES if c != "k490"]),
    ("no_nflh",   [c for c in config.CHANNEL_NAMES if c != "nflh"]),
    ("no_sst",    [c for c in config.CHANNEL_NAMES if c != "sst"]),
    ("chla_only", ["chla"]),
]


def main():
    parser = argparse.ArgumentParser(description="Train 2D or 3D ConvAE")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sweep", action="store_true",
                      help=f"Bottleneck sweep: latent_dim in {config.LATENT_SWEEP}")
    mode.add_argument("--ablate-channels", action="store_true",
                      help="Run 5 channel-ablation experiments")
    parser.add_argument("--temporal", action="store_true",
                        help="Use 3D ConvAE3D (Phase B) instead of 2D ConvAE")
    parser.add_argument("--temporal-window", type=int, default=config.TEMPORAL_WINDOW,
                        help="T frames stacked per sample (3D only)")
    parser.add_argument("--latent", type=int, default=config.LATENT_DIM)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--cube", default=str(config.CUBE_PATH))
    parser.add_argument("--debug", action="store_true",
                        help="5 epochs, 500 patches/epoch — fast smoke-test")
    parser.add_argument("--full-domain-patches", action="store_true",
                        help="Sample patches across the entire cube (disable coastal bias)")
    args = parser.parse_args()

    cube_path = Path(args.cube)
    if not cube_path.exists():
        print(f"ERROR: cube not found at {cube_path}")
        print("Run scripts/02_build_cube.py first.")
        sys.exit(1)

    # Detect channel count from the cube (handles cubes built with subsets,
    # e.g. 2010 only has chla+sst because k490/nflh have data gaps that year).
    import xarray as xr
    _ds = xr.open_zarr(cube_path, consolidated=True)
    cube_channels = list(_ds.attrs["channels"])
    _ds.close()
    n_cube_channels = len(cube_channels)
    print(f"  Cube channels: {cube_channels} (in_channels = {n_cube_channels})")

    epochs = 5 if args.debug else args.epochs
    patches = 500 if args.debug else config.PATCHES_PER_EPOCH
    coastal_ov = 0.0 if args.full_domain_patches else None

    train_fn = train_temporal if args.temporal else train

    extra_kwargs = {}
    if args.temporal:
        extra_kwargs["temporal_window"] = args.temporal_window
    variant_tag = f"3D[T={args.temporal_window}]" if args.temporal else "2D"

    if args.sweep:
        print(f"=== {variant_tag} bottleneck sweep ===")
        for ld in config.LATENT_SWEEP:
            print(f"\n--- latent_dim={ld} ---")
            train_fn(
                cube_path=cube_path,
                in_channels=n_cube_channels,
                latent_dim=ld,
                seed=args.seed,
                epochs=epochs,
                patches_per_epoch=patches,
                coastal_patch_min_overlap=coastal_ov,
                **extra_kwargs,
            )

    elif args.ablate_channels:
        print(f"=== {variant_tag} channel ablations ===")
        for label, channels in ABLATION_SETS:
            in_ch = len(channels)
            print(f"\n--- {label}: {channels} (in_channels={in_ch}) ---")
            train_fn(
                cube_path=cube_path,
                in_channels=in_ch,
                latent_dim=config.LATENT_DIM,
                seed=args.seed,
                epochs=epochs,
                patches_per_epoch=patches,
                channel_subset=channels,
                coastal_patch_min_overlap=coastal_ov,
                **extra_kwargs,
            )

    else:
        print(f"=== {variant_tag} single run: latent_dim={args.latent} ===")
        ckpt = train_fn(
            cube_path=cube_path,
            in_channels=n_cube_channels,
            latent_dim=args.latent,
            seed=args.seed,
            epochs=epochs,
            patches_per_epoch=patches,
            coastal_patch_min_overlap=coastal_ov,
            **extra_kwargs,
        )
        print(f"Checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
