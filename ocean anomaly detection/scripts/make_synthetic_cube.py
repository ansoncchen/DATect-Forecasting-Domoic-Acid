"""
Generate a synthetic data cube for end-to-end pipeline testing.

Produces realistic-shape data (T, 4, H, W) + mask with seasonal cycle,
spatial structure, and event spikes, so we can verify train/infer/evaluate
without waiting for real downloads to finish.

Usage:
    python scripts/make_synthetic_cube.py
    python scripts/make_synthetic_cube.py --out data/cube_synth.zarr --n-times 120
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.CUBE_PATH.with_name("cube_synth.zarr")))
    ap.add_argument("--n-times", type=int, default=120,
                    help="Number of 8-day composite frames (default ~2.5 years)")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--lat-res", type=int, default=160)
    ap.add_argument("--lon-res", type=int, default=200)
    args = ap.parse_args()

    rng = np.random.default_rng(config.SEED)
    times = pd.date_range(args.start, periods=args.n_times, freq="8D")
    lat = np.linspace(config.LAT_MIN, config.LAT_MAX, args.lat_res)
    lon = np.linspace(config.LON_MIN, config.LON_MAX, args.lon_res)

    print(f"Synthesizing cube: ({args.n_times}, 4, {args.lat_res}, {args.lon_res})")
    print(f"  time range: {times[0].date()} → {times[-1].date()}")

    # Spatial structure: coastal upwelling band (high values near eastern edge)
    lon_norm = (lon - lon.min()) / (lon.max() - lon.min())   # 0..1, 1=east coast
    coast_kernel = np.exp(-3 * (1 - lon_norm))               # higher near coast
    lat_norm = (lat - lat.min()) / (lat.max() - lat.min())   # 0..1, 1=north
    base_field = np.outer(0.5 + 0.2 * np.cos(2 * np.pi * lat_norm), coast_kernel)  # (H, W)

    # Per-channel: seasonal cycle + noise + occasional bloom-like spikes
    data = np.zeros((args.n_times, 4, args.lat_res, args.lon_res), dtype=np.float32)
    for t_idx, ts in enumerate(times):
        doy = ts.dayofyear
        seasonal = 1.0 + 0.6 * np.sin(2 * np.pi * (doy - 80) / 365)  # spring peak
        # Inject anomalies at specific times to mimic events
        anomaly = 0.0
        for event_anchor in (40, 80, 100):
            if abs(t_idx - event_anchor) < 5:
                anomaly += 1.5 * np.exp(-(t_idx - event_anchor) ** 2 / 4)
        noise = rng.normal(0, 0.15, (4, args.lat_res, args.lon_res)).astype(np.float32)
        # Channel-specific scaling — log-transformed channels look like log values
        chla = base_field * seasonal * 1.0 + anomaly * coast_kernel + noise[0] * 0.4
        k490 = base_field * seasonal * 0.8 + 0.5 * anomaly * coast_kernel + noise[1] * 0.3
        nflh = base_field * seasonal * 1.2 + 0.3 * anomaly * coast_kernel + noise[2] * 0.4
        # SST: opposite seasonal phase (warmer in summer), no anomaly correlation
        sst = 12.0 + 4.0 * np.sin(2 * np.pi * (doy - 200) / 365) + noise[3] * 0.5 \
              - 1.5 * coast_kernel  # upwelling cools coast
        data[t_idx, 0] = chla.astype(np.float32)
        data[t_idx, 1] = k490.astype(np.float32)
        data[t_idx, 2] = nflh.astype(np.float32)
        data[t_idx, 3] = sst.astype(np.float32)

    # Realistic cloud mask: 30% missing per pixel, with spatial coherence
    mask = np.ones((args.n_times, args.lat_res, args.lon_res), dtype=bool)
    for t_idx in range(args.n_times):
        n_clouds = rng.integers(3, 10)
        for _ in range(n_clouds):
            r0 = rng.integers(0, args.lat_res)
            c0 = rng.integers(0, args.lon_res)
            radius = rng.integers(8, 25)
            for r in range(max(0, r0 - radius), min(args.lat_res, r0 + radius)):
                for c in range(max(0, c0 - radius), min(args.lon_res, c0 + radius)):
                    if (r - r0) ** 2 + (c - c0) ** 2 < radius ** 2:
                        mask[t_idx, r, c] = False
    print(f"  Cloud mask: mean valid fraction = {mask.mean():.2f}")

    # Apply mask to data (NaN where invalid)
    data_nan = data.astype(np.float32).copy()
    data_nan[~np.broadcast_to(mask[:, np.newaxis, :, :], data.shape)] = np.nan

    # Log-transform first 3 channels, then standardize each channel
    channels = ["chla", "k490", "nflh", "sst"]
    stats = {}
    for i, ch in enumerate(channels):
        if ch != "sst":
            data_nan[:, i] = np.log(np.clip(data_nan[:, i], config.LOG_CLIP_MIN, None))
        mu = float(np.nanmean(data_nan[:, i]))
        sigma = float(np.nanstd(data_nan[:, i]))
        data_nan[:, i] = (data_nan[:, i] - mu) / (sigma + 1e-8)
        stats[ch] = {"mean": mu, "std": sigma, "log": ch != "sst"}
        print(f"  {ch}: mean={mu:.3f}  std={sigma:.3f}")

    ds = xr.Dataset(
        {
            "data": (("time", "channel", "lat", "lon"), data_nan),
            "mask": (("time", "lat", "lon"), mask),
        },
        coords={
            "time": times,
            "channel": channels,
            "lat": lat,
            "lon": lon,
        },
        attrs={"channels": channels,
               **{f"{ch}_mean": s["mean"] for ch, s in stats.items()},
               **{f"{ch}_std": s["std"] for ch, s in stats.items()},
               **{f"{ch}_log_transform": int(s["log"]) for ch, s in stats.items()}},
    )

    out_path = Path(args.out)
    if out_path.exists():
        shutil.rmtree(out_path)
    encoding = {
        "data": {"chunks": (min(46, args.n_times), 4, args.lat_res, args.lon_res)},
        "mask": {"chunks": (min(46, args.n_times), args.lat_res, args.lon_res)},
    }
    ds.to_zarr(out_path, encoding=encoding, consolidated=True, zarr_format=2)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
