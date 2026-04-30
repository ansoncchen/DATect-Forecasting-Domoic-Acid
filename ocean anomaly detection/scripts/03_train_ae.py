"""
CLI driver for ConvAE training: bottleneck sweep and channel ablations.

Usage:
    # Single run at default latent_dim=32
    python scripts/03_train_ae.py --latent 32

    # Full bottleneck sweep (5 runs)
    python scripts/03_train_ae.py --sweep

    # Channel ablations (5 runs — drop each channel + chl-only)
    python scripts/03_train_ae.py --ablate-channels

    # Quick smoke-test: 5 epochs, latent=32
    python scripts/03_train_ae.py --latent 32 --epochs 5 --debug

Output: models/ae_l{latent}_c{in_channels}_s{seed}[_channels].pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.train import train


ABLATION_SETS: list[tuple[str, list[str]]] = [
    ("no_chla",  [c for c in config.CHANNEL_NAMES if c != "chla"]),
    ("no_k490",  [c for c in config.CHANNEL_NAMES if c != "k490"]),
    ("no_nflh",  [c for c in config.CHANNEL_NAMES if c != "nflh"]),
    ("no_sst",   [c for c in config.CHANNEL_NAMES if c != "sst"]),
    ("chla_only", ["chla"]),
]


def main():
    parser = argparse.ArgumentParser(description="Train ConvAE (single run, sweep, or ablation)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sweep", action="store_true",
                      help=f"Run bottleneck sweep: latent_dim in {config.LATENT_SWEEP}")
    mode.add_argument("--ablate-channels", action="store_true",
                      help="Run 5 channel-ablation experiments")
    parser.add_argument("--latent", type=int, default=config.LATENT_DIM,
                        help="Latent dimension for single run")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--cube", default=str(config.CUBE_PATH))
    parser.add_argument("--debug", action="store_true",
                        help="5 epochs, 500 patches/epoch — fast smoke-test")
    parser.add_argument(
        "--full-domain-patches",
        action="store_true",
        help="Sample patches across the entire cube grid (disable coastal bbox overlap)",
    )
    args = parser.parse_args()

    cube_path = Path(args.cube)
    if not cube_path.exists():
        print(f"ERROR: cube not found at {cube_path}")
        print("Run scripts/02_build_cube.py first.")
        sys.exit(1)

    epochs = 5 if args.debug else args.epochs
    patches = 500 if args.debug else config.PATCHES_PER_EPOCH
    coastal_ov = 0.0 if args.full_domain_patches else None

    if args.sweep:
        print(f"=== Bottleneck sweep: latent_dim ∈ {config.LATENT_SWEEP} ===")
        for ld in config.LATENT_SWEEP:
            print(f"\n--- latent_dim={ld} ---")
            train(
                cube_path=cube_path,
                in_channels=len(config.CHANNEL_NAMES),
                latent_dim=ld,
                seed=args.seed,
                epochs=epochs,
                patches_per_epoch=patches,
                coastal_patch_min_overlap=coastal_ov,
            )

    elif args.ablate_channels:
        print("=== Channel ablations ===")
        for label, channels in ABLATION_SETS:
            in_ch = len(channels)
            print(f"\n--- {label}: {channels} (in_channels={in_ch}) ---")
            train(
                cube_path=cube_path,
                in_channels=in_ch,
                latent_dim=config.LATENT_DIM,
                seed=args.seed,
                epochs=epochs,
                patches_per_epoch=patches,
                channel_subset=channels,
                coastal_patch_min_overlap=coastal_ov,
            )

    else:
        print(f"=== Single run: latent_dim={args.latent} ===")
        ckpt = train(
            cube_path=cube_path,
            in_channels=len(config.CHANNEL_NAMES),
            latent_dim=args.latent,
            seed=args.seed,
            epochs=epochs,
            patches_per_epoch=patches,
            coastal_patch_min_overlap=coastal_ov,
        )
        print(f"Checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
