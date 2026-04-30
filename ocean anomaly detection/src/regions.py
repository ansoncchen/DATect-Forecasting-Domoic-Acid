"""
Alongshore region polygons and rasterized mask helpers.

Regions are defined as (lat_min, lat_max, lon_min, lon_max) boxes aligned to
the study domain. They cover the WA/OR/N.CA coastline in 3 broad bands.
"""
import numpy as np
from typing import NamedTuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


class Region(NamedTuple):
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


REGIONS: list[Region] = [
    Region("Olympic Coast (WA)",   47.0, 49.0, -125.5, -123.5),
    Region("SW Washington / Long Beach", 45.5, 47.0, -124.5, -123.0),
    Region("Central Oregon",       43.5, 45.5, -125.0, -123.5),
    Region("Southern OR / N CA",   41.0, 43.5, -125.5, -123.5),
]


def build_region_masks(lat: np.ndarray, lon: np.ndarray) -> dict[str, np.ndarray]:
    """
    Returns {region_name: bool array shape (len(lat), len(lon))} where True = inside region.
    lat and lon are 1-D coordinate arrays.
    """
    masks = {}
    for r in REGIONS:
        lat_mask = (lat >= r.lat_min) & (lat <= r.lat_max)
        lon_mask = (lon >= r.lon_min) & (lon <= r.lon_max)
        masks[r.name] = np.outer(lat_mask, lon_mask)
    return masks


def aggregate_to_regions(
    error_map: np.ndarray,
    valid_mask: np.ndarray,
    region_masks: dict[str, np.ndarray],
    aggregation: str = "mean",
) -> dict[str, float]:
    """
    Reduce a per-pixel error map to one scalar per region.

    Parameters
    ----------
    error_map   : (H, W) array of per-pixel reconstruction error (summed over channels)
    valid_mask  : (H, W) bool array — pixels with valid satellite obs
    region_masks: output of build_region_masks
    aggregation : "mean", "top_decile", or "max"
    """
    results = {}
    for name, rmask in region_masks.items():
        combined = rmask & valid_mask
        vals = error_map[combined]
        if vals.size == 0:
            results[name] = float("nan")
            continue
        if aggregation == "mean":
            results[name] = float(vals.mean())
        elif aggregation == "top_decile":
            results[name] = float(np.percentile(vals, 90))
        elif aggregation == "max":
            results[name] = float(vals.max())
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
    return results
