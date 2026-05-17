"""
Download MODIS Aqua 8-day composites from ERDDAP for the PNW study domain.

Two-stage discovery + download:
  1. Fetch the time axis once per dataset (small request) → list of unique 8-day timestamps.
  2. Download one frame per unique timestamp, in parallel.

Resumable: existing non-empty files are skipped.

Usage:
    python scripts/01_download.py                                          # full sweep
    python scripts/01_download.py --channels chla --start 2010-01-01 \\
        --end 2010-12-31 --workers 8                                       # one-year smoke
    python scripts/01_download.py --full-res                               # 0.0125° (slow)
"""
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def _erddap_url(dataset_id: str, var: str, date_str: str, stride: int) -> str:
    """Single-frame URL: one date."""
    t = f"{date_str}T00:00:00Z"
    lat_sel = f"[({config.LAT_MIN}):{stride}:({config.LAT_MAX})]"
    lon_sel = f"[({config.LON_MIN}):{stride}:({config.LON_MAX})]"
    depth_sel = "[(0.0):1:(0.0)]"
    time_sel = f"[({t}):1:({t})]"
    return f"{config.ERDDAP_BASE}/{dataset_id}.nc?{var}{time_sel}{depth_sel}{lat_sel}{lon_sel}"


def _erddap_url_range(dataset_id: str, var: str, start: str, end: str, stride: int) -> str:
    """
    Date-range URL: pulls all 8-day composites between start and end inclusive
    in a single ERDDAP request. Matches the original ocean-anomaly proposal's
    "Year-chunked NetCDF requests" pattern — ~100× fewer requests than per-frame,
    drastically reduces ERDDAP 429 rate-limiting pressure.
    """
    t_start = f"{start}T00:00:00Z"
    t_end = f"{end}T00:00:00Z"
    lat_sel = f"[({config.LAT_MIN}):{stride}:({config.LAT_MAX})]"
    lon_sel = f"[({config.LON_MIN}):{stride}:({config.LON_MAX})]"
    depth_sel = "[(0.0):1:(0.0)]"
    time_sel = f"[({t_start}):1:({t_end})]"
    return f"{config.ERDDAP_BASE}/{dataset_id}.nc?{var}{time_sel}{depth_sel}{lat_sel}{lon_sel}"


def _download_frame(url: str, dest: Path, retries: int = 8, timeout: int = 1800):
    """
    Download with retry. HTTP 429 (rate limit) backs off more aggressively
    than other transient errors. We extend retries to 8 so a brief rate-limit
    window doesn't permanently fail the frame.
    """
    if dest.exists() and dest.stat().st_size > 1024:
        return str(dest), True, "exists"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                if r.status_code == 429:
                    # rate limited — long backoff (15, 30, 60, 120, ...)
                    backoff = min(15 * (2 ** attempt), 300)
                    if attempt < retries - 1:
                        time.sleep(backoff)
                        continue
                    return str(dest), False, "HTTP 429 (rate limited; retries exhausted)"
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        f.write(chunk)
            tmp.rename(dest)
            return str(dest), True, "ok"
        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            if attempt < retries - 1:
                time.sleep(min(2 ** attempt, 60))
            else:
                return str(dest), False, str(exc)
    return str(dest), False, "exhausted retries"


def _get_unique_composite_dates(dataset_id: str, start: str, end: str) -> list[str]:
    """Fetch the ERDDAP time axis and return unique 8-day composite dates."""
    url = (f"{config.ERDDAP_BASE}/{dataset_id}.nc?time"
           f"[({start}T00:00:00Z):1:({end}T00:00:00Z)]")
    with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
        for attempt in range(3):
            try:
                with requests.get(url, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(tmp.name, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1 << 16):
                            f.write(chunk)
                break
            except Exception:
                time.sleep(2 ** attempt)
        else:
            print(f"  WARNING: time axis fetch failed for {dataset_id}; using synthetic 8-day dates")
            return [str(d.date()) for d in pd.date_range(start, end, freq="8D")]
        ds = xr.open_dataset(tmp.name, engine="netcdf4")
        times = ds["time"].values
        ds.close()
    # Deduplicate (ERDDAP serves daily rolling 8-day composites)
    unique = sorted({str(pd.Timestamp(t).date()) for t in times})
    return unique


def _filter_to_8day_anchors(dates: list[str]) -> list[str]:
    """
    Sub-sample a list of daily timestamps to one native 8-day composite per anchor.

    MODIS 8-day products use anchors at DOY {1, 9, 17, ...} or {5, 13, 21, ...}
    depending on the specific dataset. Rather than hardcoding the anchor offset,
    we detect the most common DOY offset modulo 8 in the input list and keep
    only those dates. Falls back to "every 8th file" if the input is sparse.
    """
    if not dates:
        return dates
    doys = [pd.Timestamp(d).dayofyear for d in dates]
    if len(doys) < 8:
        return dates  # already sparse; keep all
    # Find the most common modulo-8 residue
    residues = [doy % 8 for doy in doys]
    from collections import Counter
    common_residue = Counter(residues).most_common(1)[0][0]
    filtered = [d for d, doy in zip(dates, doys) if doy % 8 == common_residue]
    if not filtered:
        # Defensive: every 8th file
        filtered = dates[::8]
    return filtered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", default=",".join(config.CHANNEL_NAMES))
    parser.add_argument("--start", default=config.DOWNLOAD_START)
    parser.add_argument("--end", default=config.DOWNLOAD_END)
    parser.add_argument("--stride", type=int, default=config.DEFAULT_STRIDE)
    parser.add_argument("--workers", type=int, default=config.DEFAULT_WORKERS)
    parser.add_argument("--full-res", action="store_true", help="Override stride to 1")
    parser.add_argument("--anchor-only", action="store_true",
                        help="Subsample to standard MODIS 8-day anchor dates (~46/yr/channel). "
                             "Default keeps all daily rolling 8-day composites (~364/yr).")
    parser.add_argument("--per-year", action="store_true",
                        help="Use one ERDDAP request per (channel, year) instead of per-frame. "
                             "Drops total request count ~365x, avoiding rate-limit hell. "
                             "Recommended for multi-year runs. Yields one NetCDF per channel-year.")
    args = parser.parse_args()

    stride = 1 if args.full_res else args.stride
    requested = [c.strip() for c in args.channels.split(",")]
    channel_map = {c[0]: c for c in config.CHANNELS}
    selected = [channel_map[n] for n in requested if n in channel_map]
    if not selected:
        print(f"ERROR: no valid channels in '{args.channels}'"); sys.exit(1)

    log_path = config.FIGURES_DIR.parent / "download_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = []
    if args.per_year:
        # Per-year mode: one ERDDAP request fetches a full year of one channel.
        # Filename pattern: {channel}_{year}.nc (matches proposal §3).
        start_year = int(args.start[:4])
        end_year = int(args.end[:4])
        for name, dataset_id, var, _ in selected:
            out_dir = config.DATA_RAW / name
            out_dir.mkdir(parents=True, exist_ok=True)
            for year in range(start_year, end_year + 1):
                y_start = max(f"{year}-01-01", args.start)
                y_end = min(f"{year}-12-31", args.end)
                tasks.append((
                    _erddap_url_range(dataset_id, var, y_start, y_end, stride),
                    out_dir / f"{name}_{year}.nc",
                    name, str(year),
                ))
        print(f"\n=== PER-YEAR MODE ===")
        print(f"Total requests: {len(tasks)} ({len(selected)} channels × {end_year - start_year + 1} years)  "
              f"stride={stride}  workers={args.workers}\n")
    else:
        # Per-frame mode (slow for multi-year ranges due to ERDDAP rate limits).
        for name, dataset_id, var, _ in selected:
            print(f"Fetching time axis for {name} ({dataset_id})…")
            dates = _get_unique_composite_dates(dataset_id, args.start, args.end)
            print(f"  {len(dates)} unique daily timestamps")
            if args.anchor_only:
                dates = _filter_to_8day_anchors(dates)
                print(f"  --anchor-only: filtered to {len(dates)} native 8-day MODIS anchor dates")
            else:
                print(f"  Keeping all {len(dates)} daily rolling 8-day composites (denser signal)")
            out_dir = config.DATA_RAW / name
            out_dir.mkdir(parents=True, exist_ok=True)
            for d in dates:
                tasks.append((
                    _erddap_url(dataset_id, var, d, stride),
                    out_dir / f"{name}_{d.replace('-', '')}.nc",
                    name, d,
                ))

        print(f"\nTotal frames: {len(tasks)} ({len(selected)} channels × dates)  "
              f"stride={stride}  workers={args.workers}\n")

    results = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_download_frame, url, dest): (ch, d)
                   for url, dest, ch, d in tasks}
        for fut in as_completed(futures):
            ch, d = futures[fut]
            path, ok, msg = fut.result()
            completed += 1
            status = "OK" if ok else "FAIL"
            if not ok:
                print(f"  [{completed}/{len(tasks)}] {status} {ch} {d}: {msg}")
            elif completed % 100 == 0:
                print(f"  [{completed}/{len(tasks)}] {status} …")
            results.append({"channel": ch, "date": d, "path": path,
                            "status": status, "message": msg})

    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["channel", "date", "path", "status", "message"])
        w.writeheader()
        w.writerows(results)

    failed = sum(1 for r in results if r["status"] == "FAIL")
    print(f"\nDone. {len(results) - failed}/{len(results)} succeeded. Log: {log_path}")
    if failed:
        print(f"  {failed} failures — re-run to resume (existing files are skipped).")


if __name__ == "__main__":
    main()
