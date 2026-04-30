"""
Build the standardized 4-channel data cube from downloaded NetCDF frames.

Output: data/cube.zarr  with dims (time, channel, lat, lon)
        and a 'mask' variable (time, lat, lon) — True where all 4 channels valid.

Usage:
    python scripts/02_build_cube.py
    python scripts/02_build_cube.py --year 2010      # single-year debug mode
    python scripts/02_build_cube.py --channels chla,sst  # subset of channels
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def _open_channel(name: str, year: int | None = None) -> xr.DataArray:
    """Open all NetCDF files for one channel and concatenate along time."""
    ch_dir = config.DATA_RAW / name
    files = sorted(ch_dir.glob(f"{name}_*.nc"))
    if year is not None:
        files = [f for f in files if f.stem.split("_")[1][:4] == str(year)]
    if not files:
        raise FileNotFoundError(f"No NetCDF files found in {ch_dir}")

    print(f"  {name}: {len(files)} files")
    ds = xr.open_mfdataset(
        files,
        combine="nested",
        concat_dim="time",
        engine="netcdf4",
        decode_times=True,
        parallel=True,
    )
    # Pick the data variable (different names per dataset)
    _, _, var_name, _ = next(c for c in config.CHANNELS if c[0] == name)
    da = ds[var_name]
    # Drop altitude/depth dim if present
    if "altitude" in da.dims:
        da = da.squeeze("altitude", drop=True)
    if "depth" in da.dims:
        da = da.squeeze("depth", drop=True)
    return da.rename(name)


def _log_transform(da: xr.DataArray, log_transform: bool) -> xr.DataArray:
    if log_transform:
        return np.log(da.clip(min=config.LOG_CLIP_MIN))
    return da


def _standardize(da: xr.DataArray) -> tuple[xr.DataArray, float, float]:
    """Global mean/std standardization. Returns (standardized, mean, std)."""
    vals = da.values
    mu = float(np.nanmean(vals))
    sigma = float(np.nanstd(vals))
    standardized = (da - mu) / (sigma + 1e-8)
    return standardized, mu, sigma


def main():
    parser = argparse.ArgumentParser(description="Build data cube from downloaded NetCDF frames")
    parser.add_argument("--year", type=int, default=None, help="Single year (debug mode)")
    parser.add_argument(
        "--channels", default=",".join(config.CHANNEL_NAMES),
        help="Channels to include in the cube (must be subset of config.CHANNEL_NAMES)"
    )
    parser.add_argument("--out", default=str(config.CUBE_PATH), help="Output Zarr path")
    args = parser.parse_args()

    channel_names = [c.strip() for c in args.channels.split(",")]
    channel_map = {c[0]: c for c in config.CHANNELS}
    selected = [channel_map[n] for n in channel_names if n in channel_map]

    out_path = Path(args.out)
    if args.year is not None:
        out_path = out_path.with_name(f"cube_{args.year}.zarr")

    print(f"Building cube: {out_path}")
    print(f"Channels: {[c[0] for c in selected]}")
    if args.year:
        print(f"Year filter: {args.year}")

    # -----------------------------------------------------------------------
    # 1. Load + transform each channel
    # -----------------------------------------------------------------------
    channel_das = {}
    stats = {}
    for name, _, _, log_flag in selected:
        print(f"\nLoading {name}…")
        da = _open_channel(name, year=args.year)
        da = _log_transform(da, log_flag)
        da, mu, sigma = _standardize(da)
        channel_das[name] = da
        stats[name] = {"mean": mu, "std": sigma, "log_transform": log_flag}
        print(f"  mean={mu:.4f}  std={sigma:.4f}")

    # -----------------------------------------------------------------------
    # 2. Align all channels to chl-a grid (no-op if same MODIS grid)
    # -----------------------------------------------------------------------
    reference = channel_das[channel_names[0]]
    for name in channel_names[1:]:
        if channel_das[name].shape != reference.shape:
            print(f"  Regridding {name} to reference grid…")
            channel_das[name] = channel_das[name].interp_like(reference)

    # -----------------------------------------------------------------------
    # 3. Stack into (time, channel, lat, lon)
    # -----------------------------------------------------------------------
    stacked = xr.concat(
        [channel_das[n] for n in channel_names],
        dim="channel",
    ).assign_coords(channel=channel_names)
    stacked = stacked.transpose("time", "channel", "lat", "lon")

    # -----------------------------------------------------------------------
    # 4. Build valid mask: True where all channels are non-NaN
    # -----------------------------------------------------------------------
    mask = ~np.isnan(stacked).any(dim="channel")

    ds_out = xr.Dataset(
        {
            "data": stacked,
            "mask": mask,
        }
    )

    # Store per-channel stats as attributes for inference inversion
    for name, s in stats.items():
        ds_out.attrs[f"{name}_mean"] = s["mean"]
        ds_out.attrs[f"{name}_std"] = s["std"]
        ds_out.attrs[f"{name}_log_transform"] = int(s["log_transform"])
    ds_out.attrs["channels"] = channel_names

    # -----------------------------------------------------------------------
    # 5. Write Zarr (chunk ~one year per slab)
    # -----------------------------------------------------------------------
    n_times = len(stacked.time)
    chunk_t = min(46, n_times)  # ~1 year of 8-day composites
    H = len(stacked.lat)
    W = len(stacked.lon)
    encoding = {
        "data": {"chunks": (chunk_t, len(channel_names), H, W)},
        "mask": {"chunks": (chunk_t, H, W)},
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        import shutil
        shutil.rmtree(out_path)
    ds_out.to_zarr(out_path, encoding=encoding, consolidated=True)

    print(f"\nCube written: {out_path}")
    print(f"  Shape: time={n_times}, channel={len(channel_names)}, lat={H}, lon={W}")
    print(f"  Approx size: {n_times * len(channel_names) * H * W * 4 / 1e9:.2f} GB (float32)")


if __name__ == "__main__":
    main()
