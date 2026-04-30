"""
Download MODIS Aqua 8-day composites from ERDDAP for the PNW study domain.

Usage:
    python scripts/01_download.py
    python scripts/01_download.py --channels chla --start 2010-01-01 --end 2010-12-31
    python scripts/01_download.py --workers 4 --stride 1  # full-res override

Output:
    data/raw/{channel}/{channel}_{YYYYMMDD}.nc   — one file per 8-day composite
    outputs/download_log.csv                      — status record for each file
"""
import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# ---------------------------------------------------------------------------
# URL builder
# ---------------------------------------------------------------------------

def _erddap_url(dataset_id: str, var: str, date_str: str, stride: int) -> str:
    """
    Build a single-frame ERDDAP griddap NetCDF URL.

    date_str: "YYYY-MM-DD"
    ERDDAP time selector for a single 8-day frame: (date):1:(date)
    Longitude in PM180 convention (negative west).
    """
    t = f"{date_str}T00:00:00Z"
    lat_sel = f"[({config.LAT_MIN}):{stride}:({config.LAT_MAX})]"
    lon_sel = f"[({config.LON_MIN}):{stride}:({config.LON_MAX})]"
    depth_sel = "[(0.0):1:(0.0)]"
    time_sel = f"[({t}):1:({t})]"
    query = f"{var}{time_sel}{depth_sel}{lat_sel}{lon_sel}"
    return f"{config.ERDDAP_BASE}/{dataset_id}.nc?{query}"


# ---------------------------------------------------------------------------
# Single-file download with retry
# ---------------------------------------------------------------------------

def _download_frame(
    url: str,
    dest: Path,
    retries: int = 5,
    timeout: int = 300,
) -> tuple[str, bool, str]:
    """Download url → dest. Returns (dest_str, success, message)."""
    if dest.exists() and dest.stat().st_size > 0:
        return str(dest), True, "already exists"

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")

    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                if r.status_code != 200:
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        f.write(chunk)
            tmp.rename(dest)
            return str(dest), True, "ok"
        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            wait = 2 ** attempt
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                return str(dest), False, str(exc)
    return str(dest), False, "exhausted retries"


# ---------------------------------------------------------------------------
# Enumerate available composite dates for a dataset via metadata request
# ---------------------------------------------------------------------------

def _get_time_axis(dataset_id: str, var: str, start: str, end: str) -> list[str]:
    """
    Fetch only the time coordinate for the dataset between start and end.
    Returns a list of ISO date strings ("YYYY-MM-DD").
    """
    url = (
        f"{config.ERDDAP_BASE}/{dataset_id}.nc?time"
        f"[({start}T00:00:00Z):1:({end}T00:00:00Z)]"
    )
    import tempfile, xarray as xr
    with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
        ok = False
        for attempt in range(3):
            try:
                with requests.get(url, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(tmp.name, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1 << 16):
                            f.write(chunk)
                ok = True
                break
            except Exception as e:
                time.sleep(2 ** attempt)
        if not ok:
            print(f"  WARNING: could not fetch time axis for {dataset_id}. Using date range directly.")
            return []
        ds = xr.open_dataset(tmp.name, engine="netcdf4")
        times = ds["time"].values
        ds.close()
    import pandas as pd
    return [str(pd.Timestamp(t).date()) for t in times]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download MODIS Aqua ERDDAP frames")
    parser.add_argument(
        "--channels", default=",".join(config.CHANNEL_NAMES),
        help="Comma-separated channel names to download (default: all)"
    )
    parser.add_argument("--start", default=config.DOWNLOAD_START)
    parser.add_argument("--end", default=config.DOWNLOAD_END)
    parser.add_argument("--stride", type=int, default=config.DEFAULT_STRIDE)
    parser.add_argument("--workers", type=int, default=config.DEFAULT_WORKERS)
    parser.add_argument("--full-res", action="store_true", help="Override stride to 1")
    args = parser.parse_args()

    stride = 1 if args.full_res else args.stride
    channels_requested = [c.strip() for c in args.channels.split(",")]

    channel_map = {c[0]: c for c in config.CHANNELS}
    selected = [channel_map[n] for n in channels_requested if n in channel_map]
    if not selected:
        print(f"ERROR: no valid channels in '{args.channels}'")
        sys.exit(1)

    log_path = config.FIGURES_DIR.parent / "download_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = []  # (url, dest, channel_name, date_str)
    for name, dataset_id, var, _ in selected:
        print(f"Fetching time axis for {name} ({dataset_id})…")
        dates = _get_time_axis(dataset_id, var, args.start, args.end)
        if not dates:
            # Fallback: use pandas date_range at ~8-day intervals
            import pandas as pd
            dates = [str(d.date()) for d in pd.date_range(args.start, args.end, freq="8D")]
            print(f"  Falling back to {len(dates)} synthetic 8-day dates")
        else:
            print(f"  Found {len(dates)} composite dates")

        out_dir = config.DATA_RAW / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for d in dates:
            url = _erddap_url(dataset_id, var, d, stride)
            dest = out_dir / f"{name}_{d.replace('-', '')}.nc"
            tasks.append((url, dest, name, d))

    print(f"\nTotal frames to download: {len(tasks)} ({len(selected)} channels × dates)")
    print(f"Stride: {stride}  Workers: {args.workers}\n")

    results = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_download_frame, url, dest): (ch, d) for url, dest, ch, d in tasks}
        for future in as_completed(futures):
            ch, d = futures[future]
            path, ok, msg = future.result()
            completed += 1
            status = "OK" if ok else "FAIL"
            if not ok:
                print(f"  [{completed}/{len(tasks)}] {status} {ch} {d}: {msg}")
            elif completed % 100 == 0:
                print(f"  [{completed}/{len(tasks)}] {status} …")
            results.append({"channel": ch, "date": d, "path": path, "status": status, "message": msg})

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["channel", "date", "path", "status", "message"])
        writer.writeheader()
        writer.writerows(results)

    failed = sum(1 for r in results if r["status"] == "FAIL")
    print(f"\nDone. {len(results) - failed}/{len(results)} succeeded. Log: {log_path}")
    if failed:
        print(f"  {failed} failures — re-run to resume (existing files are skipped).")


if __name__ == "__main__":
    main()
