"""
Build the standardized multi-channel data cube from downloaded NetCDF frames.

Critical: ERDDAP serves the 8-day composite indexed by every day in the window,
so the downloader can produce one .nc file per *daily* timestamp even though
they all contain the same 8-day composite. We deduplicate on the `time`
coordinate inside the files, not the filename.

Output: data/cube.zarr  with dims (time, channel, lat, lon) + mask(time, lat, lon)

Usage:
    python scripts/02_build_cube.py
    python scripts/02_build_cube.py --year 2010                # single year debug
    python scripts/02_build_cube.py --channels chla,sst
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def _open_channel_dedup(name: str, year: int | None = None) -> xr.DataArray:
    """
    Open all NetCDFs for one channel, concatenate along time, deduplicate.

    Returns a DataArray with monotonic-unique time, dims (time, lat, lon).
    """
    ch_dir = config.DATA_RAW / name
    files = sorted(ch_dir.glob(f"{name}_*.nc"))
    if year is not None:
        files = [f for f in files if f.stem.split("_")[1][:4] == str(year)]
    if not files:
        raise FileNotFoundError(f"No NetCDFs found in {ch_dir}")
    print(f"  {name}: {len(files)} files on disk")

    ds = xr.open_mfdataset(
        files, combine="nested", concat_dim="time",
        engine="netcdf4", decode_times=True, parallel=True,
    )
    _, _, var_name, _ = next(c for c in config.CHANNELS if c[0] == name)
    da = ds[var_name]
    for d in ("altitude", "depth"):
        if d in da.dims:
            da = da.squeeze(d, drop=True)

    # ERDDAP NetCDFs use 'latitude'/'longitude'; normalize to 'lat'/'lon'
    rename_map = {}
    if "latitude" in da.dims:
        rename_map["latitude"] = "lat"
    if "longitude" in da.dims:
        rename_map["longitude"] = "lon"
    if rename_map:
        da = da.rename(rename_map)

    # Deduplicate on time coordinate
    df = pd.DataFrame({"time": da["time"].values, "idx": np.arange(len(da["time"]))})
    df = df.drop_duplicates("time").sort_values("time")
    keep = df["idx"].values
    da = da.isel(time=keep)
    print(f"    {len(keep)} unique composites after dedup")
    return da.rename(name)


def _log_transform(da: xr.DataArray, log_flag: bool) -> xr.DataArray:
    if log_flag:
        return np.log(da.clip(min=config.LOG_CLIP_MIN))
    return da


def _standardize(da: xr.DataArray) -> tuple[xr.DataArray, float, float]:
    mu = float(np.nanmean(da.values))
    sigma = float(np.nanstd(da.values))
    return (da - mu) / (sigma + 1e-8), mu, sigma


def _align_times(channel_das: dict[str, xr.DataArray]) -> dict[str, xr.DataArray]:
    """Intersect time axes across channels (so the stacked cube has aligned dates)."""
    common = None
    for da in channel_das.values():
        t = set(pd.DatetimeIndex(da["time"].values))
        common = t if common is None else common & t
    common_sorted = sorted(common)
    print(f"\n  Common time axis: {len(common_sorted)} composites "
          f"({common_sorted[0]} → {common_sorted[-1]})")
    return {name: da.sel(time=common_sorted) for name, da in channel_das.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--channels", default=",".join(config.CHANNEL_NAMES))
    parser.add_argument("--out", default=str(config.CUBE_PATH))
    args = parser.parse_args()

    channel_names = [c.strip() for c in args.channels.split(",")]
    channel_map = {c[0]: c for c in config.CHANNELS}
    selected = [channel_map[n] for n in channel_names if n in channel_map]

    out_path = Path(args.out)
    if args.year is not None:
        out_path = out_path.with_name(f"cube_{args.year}.zarr")
    print(f"Output: {out_path}\nChannels: {[c[0] for c in selected]}"
          + (f"\nYear filter: {args.year}" if args.year else ""))

    channel_das = {}
    stats = {}
    for name, _, _, log_flag in selected:
        print(f"\nLoading {name}…")
        da = _open_channel_dedup(name, year=args.year)
        da = _log_transform(da, log_flag)
        da, mu, sigma = _standardize(da)
        channel_das[name] = da
        stats[name] = {"mean": mu, "std": sigma, "log_transform": log_flag}
        print(f"    mean={mu:.4f}  std={sigma:.4f}")

    channel_das = _align_times(channel_das)

    reference = channel_das[channel_names[0]]
    for name in channel_names[1:]:
        if channel_das[name].shape != reference.shape:
            print(f"  Regridding {name} to reference grid…")
            channel_das[name] = channel_das[name].interp_like(reference)

    stacked = xr.concat(
        [channel_das[n] for n in channel_names], dim="channel"
    ).assign_coords(channel=channel_names).transpose("time", "channel", "lat", "lon")

    # Rechunk dask arrays to match target zarr chunks (avoids partial-write errors)
    n_t_chunk = min(46, len(stacked.time))
    stacked = stacked.chunk({"time": n_t_chunk, "channel": -1, "lat": -1, "lon": -1})

    mask = ~np.isnan(stacked).any(dim="channel")
    ds_out = xr.Dataset({"data": stacked, "mask": mask})
    for name, s in stats.items():
        ds_out.attrs[f"{name}_mean"] = s["mean"]
        ds_out.attrs[f"{name}_std"] = s["std"]
        ds_out.attrs[f"{name}_log_transform"] = int(s["log_transform"])
    ds_out.attrs["channels"] = channel_names

    n_times = len(stacked.time)
    H = len(stacked.lat)
    W = len(stacked.lon)
    chunk_t = min(46, n_times)
    encoding = {
        "data": {"chunks": (chunk_t, len(channel_names), H, W)},
        "mask": {"chunks": (chunk_t, H, W)},
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        shutil.rmtree(out_path)
    ds_out.to_zarr(out_path, encoding=encoding, consolidated=True, zarr_format=2)

    print(f"\nCube written: {out_path}")
    print(f"  Shape: time={n_times}, channel={len(channel_names)}, lat={H}, lon={W}")
    print(f"  Approx size: {n_times * len(channel_names) * H * W * 4 / 1e9:.2f} GB (float32)")


if __name__ == "__main__":
    main()
