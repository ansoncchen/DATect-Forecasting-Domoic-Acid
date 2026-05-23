"""Generate poster-ready schematic figures for the OAD project.

If the Hyak cube exists at `data/cube.zarr`, the study-region figure renders a
real MODIS Aqua composite frame. Otherwise it falls back to an explicitly
labeled synthetic MODIS-style preview for local-only use.
"""
from __future__ import annotations

import json
import subprocess
import textwrap
import urllib.request
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "poster_assets"
OUT.mkdir(parents=True, exist_ok=True)
CUBE = ROOT / "data" / "cube.zarr"
TARGET_COMPOSITE_DATE = "2015-07-15"
MAP_CACHE = OUT / "map_cache"
MAP_CACHE.mkdir(parents=True, exist_ok=True)

LAT_MIN, LAT_MAX = 41.0, 49.0
LON_MIN, LON_MAX = -131.0, -120.8

REGIONS = [
    ("Olympic Coast\n(WA)", 47.0, 49.0, -125.5, -123.5, "#3b82f6"),
    ("SW Washington /\nLong Beach", 45.5, 47.0, -124.5, -123.0, "#10b981"),
    ("Central\nOregon", 43.5, 45.5, -125.0, -123.5, "#f59e0b"),
    ("Southern OR /\nN CA", 41.0, 43.5, -125.5, -123.5, "#ef4444"),
]

STATE_GEOJSON_URLS = {
    "Washington": "https://raw.githubusercontent.com/glynnbird/usstatesgeojson/master/washington.geojson",
    "Oregon": "https://raw.githubusercontent.com/glynnbird/usstatesgeojson/master/oregon.geojson",
    "California": "https://raw.githubusercontent.com/glynnbird/usstatesgeojson/master/california.geojson",
}


def _style_axis(ax):
    ax.set_facecolor("#e7f1f7")
    ax.tick_params(colors="#334155", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")


def _synthetic_composite(seed: int = 7) -> np.ndarray:
    """MODIS-like RGB proxy: chl-a/fluorescence as green, SST as red, Kd490 as blue."""
    rng = np.random.default_rng(seed)
    h, w = 420, 360
    y, x = np.mgrid[0:h, 0:w]
    coast = 0.58 * w + 0.10 * w * np.sin((y / h) * 3.6 * np.pi) - 0.05 * y
    offshore = x < coast
    dist = np.clip((coast - x) / w, 0, 1)
    alongshore = (np.sin(y / 33.0) + 0.45 * np.cos(y / 71.0) + 1.5) / 3.0
    eddies = (
        0.35 * np.sin((x + 1.8 * y) / 30.0)
        + 0.25 * np.cos((2.1 * x - y) / 45.0)
        + rng.normal(0, 0.035, (h, w))
    )

    chla = np.clip(0.18 + 0.75 * dist * alongshore + eddies, 0, 1)
    sst = np.clip(0.72 - 0.35 * dist + 0.12 * np.sin(y / 60.0) - 0.10 * eddies, 0, 1)
    k490 = np.clip(0.20 + 0.55 * dist + 0.10 * np.cos(x / 40.0), 0, 1)

    rgb = np.dstack([sst, 0.78 * chla + 0.10, 0.55 * k490 + 0.20])
    land = ~offshore
    rgb[land] = np.array([0.72, 0.68, 0.56])

    # Cloud/missing-data streaks, shown in white to match masked-pixel handling.
    clouds = (
        (np.sin((x + y) / 34.0) > 0.93)
        | (((x - 40) ** 2 / 9000 + (y - 110) ** 2 / 1800) < 1)
        | (((x - 260) ** 2 / 14000 + (y - 310) ** 2 / 2600) < 1)
    )
    rgb[clouds & offshore] = np.array([0.96, 0.97, 0.98])
    return rgb


def _load_state_geojson(name: str) -> dict | None:
    """Load cached state GeoJSON, downloading once if needed."""
    path = MAP_CACHE / f"{name.lower()}.geojson"
    if not path.exists():
        try:
            with urllib.request.urlopen(STATE_GEOJSON_URLS[name], timeout=20) as resp:
                path.write_bytes(resp.read())
        except Exception as exc:
            print(f"Warning: could not download {name} GeoJSON: {exc}")
            return None
    with path.open() as f:
        return json.load(f)


def _iter_polygon_rings(geometry: dict):
    """Yield exterior rings from Polygon/MultiPolygon GeoJSON geometry."""
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if gtype == "Polygon":
        for poly in [coords]:
            if poly:
                yield poly[0]
    elif gtype == "MultiPolygon":
        for poly in coords:
            if poly:
                yield poly[0]


def _plot_real_pacific_coast(ax) -> bool:
    """Draw WA/OR/CA polygons from real GeoJSON geometry."""
    ok = False
    for state in ("Washington", "Oregon", "California"):
        gj = _load_state_geojson(state)
        if not gj:
            continue
        features = gj.get("features", [gj] if gj.get("type") == "Feature" else [])
        for feat in features:
            geom = feat.get("geometry", {})
            for ring in _iter_polygon_rings(geom):
                arr = np.asarray(ring, dtype=float)
                if arr.ndim != 2 or arr.shape[1] < 2:
                    continue
                # Skip tiny islands far outside the poster extent.
                if arr[:, 0].max() < LON_MIN or arr[:, 0].min() > LON_MAX:
                    continue
                if arr[:, 1].max() < LAT_MIN or arr[:, 1].min() > LAT_MAX:
                    continue
                ax.fill(arr[:, 0], arr[:, 1], facecolor="#d7c7a3", edgecolor="#6b5a46",
                        linewidth=0.8, zorder=2)
                ok = True
    return ok


def _normalize_band(arr: np.ndarray, valid: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    vals = arr[valid & np.isfinite(arr)]
    if vals.size == 0:
        return np.zeros_like(arr, dtype=float)
    p_lo, p_hi = np.nanpercentile(vals, [lo, hi])
    return np.clip((arr - p_lo) / (p_hi - p_lo + 1e-8), 0, 1)


def _real_composite_from_cube(target_date: str = TARGET_COMPOSITE_DATE) -> tuple[np.ndarray, str] | None:
    """Render an RGB proxy from the standardized OAD cube if it is available."""
    if not CUBE.exists():
        return None

    import pandas as pd
    import xarray as xr

    ds = xr.open_zarr(CUBE, consolidated=True)
    try:
        channels = list(ds.attrs.get("channels", []))
        if not {"chla", "k490", "nflh", "sst"}.issubset(channels):
            return None
        target = np.datetime64(target_date)
        times = pd.to_datetime(ds["time"].values)
        idx = int(np.argmin(np.abs(times.values.astype("datetime64[ns]") - target.astype("datetime64[ns]"))))
        date_label = times[idx].strftime("%Y-%m-%d")

        frame = ds["data"].isel(time=idx).values.astype(float)
        valid = ds["mask"].isel(time=idx).values.astype(bool)
        ch = {name: frame[channels.index(name)] for name in channels}

        chla = _normalize_band(ch["chla"], valid)
        k490 = _normalize_band(ch["k490"], valid)
        nflh = _normalize_band(ch["nflh"], valid)
        sst = _normalize_band(ch["sst"], valid)

        # False-color ocean RGB: warm SST = red, productivity = green, turbidity = blue.
        rgb = np.dstack([
            0.25 + 0.75 * sst,
            0.15 + 0.55 * chla + 0.25 * nflh,
            0.20 + 0.65 * k490,
        ])
        rgb = np.clip(rgb, 0, 1)
        rgb[~valid] = np.array([0.94, 0.95, 0.96])
        return rgb, date_label
    finally:
        ds.close()


def _real_channel_panels(target_date: str = TARGET_COMPOSITE_DATE):
    """Return normalized single-channel panels from the cube for a clean poster inset."""
    if not CUBE.exists():
        return None

    import pandas as pd
    import xarray as xr

    ds = xr.open_zarr(CUBE, consolidated=True)
    try:
        channels = list(ds.attrs.get("channels", []))
        required = ["chla", "k490", "nflh", "sst"]
        if not set(required).issubset(channels):
            return None
        target = np.datetime64(target_date)
        times = pd.to_datetime(ds["time"].values)
        idx = int(np.argmin(np.abs(times.values.astype("datetime64[ns]") - target.astype("datetime64[ns]"))))
        date_label = times[idx].strftime("%Y-%m-%d")
        frame = ds["data"].isel(time=idx).values.astype(float)
        valid = ds["mask"].isel(time=idx).values.astype(bool)
        panels = {}
        for ch in required:
            band = frame[channels.index(ch)]
            norm = _normalize_band(band, valid)
            norm = np.where(valid, norm, np.nan)
            panels[ch] = norm
        return panels, date_label
    finally:
        ds.close()


def _plot_channel_panel_grid(fig, outer_gs, channel_panels):
    panel_gs = outer_gs.subgridspec(2, 2, wspace=0.08, hspace=0.18)
    cmaps = {
        "chla": "viridis",
        "k490": "cividis",
        "nflh": "magma",
        "sst": "coolwarm",
    }
    titles = {
        "chla": "Chlorophyll-a",
        "k490": "Diffuse attenuation (Kd490)",
        "nflh": "Fluorescence line height",
        "sst": "Sea surface temperature",
    }
    for i, ch in enumerate(["chla", "k490", "nflh", "sst"]):
        ax = fig.add_subplot(panel_gs[i // 2, i % 2])
        cmap = plt.get_cmap(cmaps[ch]).copy()
        cmap.set_bad("#eef2f7")
        im = ax.imshow(channel_panels[ch], origin="lower", cmap=cmap, vmin=0, vmax=1)
        ax.set_title(titles[ch], fontsize=10.5, weight="bold", pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#cbd5e1")
            spine.set_linewidth(0.8)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.015)
        cb.set_ticks([0, 1])
        cb.set_ticklabels(["low", "high"])
        cb.ax.tick_params(labelsize=7, length=0)


def make_study_region() -> None:
    fig = plt.figure(figsize=(14.2, 7.6), dpi=220)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.28], wspace=0.16)
    ax = fig.add_subplot(gs[0, 0])
    comp_ax = fig.add_subplot(gs[0, 1])

    _style_axis(ax)
    ax.set_title("Study Region: U.S. Pacific Northwest Coast", loc="left", fontsize=17, weight="bold")
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(color="white", linewidth=1.0, alpha=0.85)

    if not _plot_real_pacific_coast(ax):
        # Last-resort fallback if the public GeoJSON cannot be fetched.
        ax.axvspan(-124.5, LON_MAX, color="#d7c7a3", zorder=1)
    ax.text(-122.6, 48.0, "WA", color="#5b4636", fontsize=11, weight="bold", zorder=4)
    ax.text(-122.6, 44.5, "OR", color="#5b4636", fontsize=11, weight="bold", zorder=4)
    ax.text(-122.4, 41.5, "CA", color="#5b4636", fontsize=11, weight="bold", zorder=4)
    ax.text(-129.9, 48.35, "Pacific Ocean", color="#2563eb", fontsize=11, weight="bold")

    overall = patches.Rectangle(
        (-125.5, 41.0), 2.5, 8.0, facecolor="#0f172a", alpha=0.05,
        edgecolor="#0f172a", linewidth=1.6, linestyle="--"
    )
    ax.add_patch(overall)
    ax.text(-125.42, 48.72, "Overall coastal envelope", fontsize=9, color="#0f172a")

    for name, lat0, lat1, lon0, lon1, color in REGIONS:
        rect = patches.Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0,
                                 facecolor=color, alpha=0.24, edgecolor=color, linewidth=2.2)
        ax.add_patch(rect)
        ax.text(lon0 + 0.05, (lat0 + lat1) / 2, name, va="center", ha="left",
                fontsize=8.7, color="#0f172a", weight="bold")

    ax.text(
        LON_MIN + 0.15, LAT_MIN + 0.25,
        "41-49°N, 131-120.8°W\nMODIS Aqua, 2003-2024",
        fontsize=9.2,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cbd5e1", alpha=0.88),
    )

    real = _real_composite_from_cube()
    if real is None:
        comp, date_label = _synthetic_composite(), "synthetic fallback"
        caption = "Synthetic fallback shown because `data/cube.zarr` is not available locally."
    else:
        comp, date_label = real
        caption = (
            f"False-color 4-channel composite nearest {TARGET_COMPOSITE_DATE}: {date_label}. "
            "White/light gray pixels are masked clouds, land, or gaps."
        )

    comp_ax.imshow(comp, origin="lower")
    comp_ax.set_axis_off()
    comp_ax.set_title("Sample 4-Channel Composite", loc="left", fontsize=17, weight="bold")
    comp_ax.text(
        0.0,
        -0.045,
        caption,
        transform=comp_ax.transAxes,
        fontsize=9.4,
        color="#475569",
        va="top",
    )

    fig.suptitle("Ocean Anomaly Detection Inputs and Regions", fontsize=21, weight="bold", y=0.995)
    fig.savefig(OUT / "study_region_sample_composite.png", bbox_inches="tight")
    fig.savefig(OUT / "study_region_sample_composite.svg", bbox_inches="tight")
    plt.close(fig)


def _box(ax, xy, wh, text, fc, ec="#334155", fs=10.5, lw=1.6):
    rect = patches.FancyBboxPatch(
        xy, wh[0], wh[1], boxstyle="round,pad=0.02,rounding_size=0.035",
        facecolor=fc, edgecolor=ec, linewidth=lw
    )
    ax.add_patch(rect)
    ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] / 2, text, ha="center", va="center",
            fontsize=fs, color="#0f172a", linespacing=1.2)
    return rect


def _arrow(ax, start, end, color="#334155", lw=1.8, rad=0.0):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=5, shrinkB=5,
                        connectionstyle=f"arc3,rad={rad}"),
    )


def make_architecture() -> None:
    dot = textwrap.dedent(
        """
        digraph G {
          graph [
            rankdir=TB,
            bgcolor="white",
            pad="0.28",
            nodesep="0.34",
            ranksep="0.42",
            splines=true,
            outputorder=edgesfirst,
            fontname="Helvetica",
            labelloc="t",
            label=<
              <FONT POINT-SIZE="30"><B>3D Masked Autoencoder Anomaly-Detection Pipeline</B></FONT><BR/>
              <FONT POINT-SIZE="15" COLOR="#475569">Train on masked MODIS structure, then aggregate reconstruction error into regional anomaly scores.</FONT>
            >
          ];
          node [
            shape=box,
            style="rounded,filled",
            color="#334155",
            penwidth=1.6,
            fontname="Helvetica",
            fontsize=16,
            margin="0.15,0.09"
          ];
          edge [
            color="#475569",
            penwidth=1.8,
            arrowsize=0.8,
            fontname="Helvetica",
            fontsize=12
          ];

          input [fillcolor="#dbeafe", label=< <B>MODIS input window</B><BR/>4 dates x 4 channels<BR/><FONT POINT-SIZE="12">chl-a, Kd490, fluorescence, SST</FONT> >];
          valid [fillcolor="#f8fafc", label=< <B>Validity mask</B><BR/>cloud / land / gaps ignored >];
          hide [fillcolor="#fee2e2", label=< <B>Masked-autoencoder training</B><BR/>70% of valid pixels hidden<BR/><FONT POINT-SIZE="12">loss only on hidden valid pixels</FONT> >];

          subgraph cluster_encoder {
            label="Encoder";
            color="#86efac";
            penwidth=1.2;
            style="rounded";
            fontname="Helvetica-Bold";
            fontsize=18;
            enc1 [fillcolor="#dcfce7", label=< <B>Conv3D 1</B><BR/>T 4 to 2<BR/>HW 64 to 32<BR/>C 4 to 32 >];
            enc2 [fillcolor="#bbf7d0", label=< <B>Conv3D 2</B><BR/>T 2 to 1<BR/>HW 32 to 16<BR/>C 32 to 64 >];
            enc3 [fillcolor="#86efac", label=< <B>Conv3D 3</B><BR/>T stays 1<BR/>HW 16 to 8<BR/>C 64 to 128 >];
          }

          latent [fillcolor="#fef3c7", label=< <B>Latent bottleneck</B><BR/>128 x 1 x 8 x 8 to z<BR/>latent dim = 32 >];

          subgraph cluster_decoder {
            label="Scoring path";
            color="#fdba74";
            penwidth=1.2;
            style="rounded";
            fontname="Helvetica-Bold";
            fontsize=18;
            decode [fillcolor="#ffedd5", label=< <B>Decoder</B><BR/>dense expand + transpose Conv3D<BR/>reconstruct 4-frame window >];
            score [fillcolor="#e9d5ff", label=< <B>Anchor-frame error</B><BR/>score final frame only<BR/>sum channel MSE >];
            region [fillcolor="#ddd6fe", label=< <B>Regional anomaly score</B><BR/>mean error over 5 coastal polygons >];
          }

          baseline [fillcolor="#eef2ff", label=< <B>Baselines</B><BR/>matched-k PCA<BR/>chl-a z-score<BR/>multivariate climatology >];
          eval [fillcolor="#f1f5f9", label=< <B>Label-free evaluation</B><BR/>seasonal cycle | event response | forecastability<BR/>channel ablations | cloud and sensor checks >];

          { rank=same; input; valid; }
          { rank=same; baseline; eval; }

          input -> hide;
          valid -> hide;
          hide -> enc1 -> enc2 -> enc3 -> latent -> decode -> score -> region -> eval;
          baseline -> eval;
        }
        """
    ).strip()
    dot_path = OUT / "architecture_diagram.dot"
    dot_path.write_text(dot)
    subprocess.run(["dot", "-Tsvg", str(dot_path), "-o", str(OUT / "architecture_diagram.svg")], check=True)
    subprocess.run(["dot", "-Tpng", "-Gdpi=220", str(dot_path), "-o", str(OUT / "architecture_diagram.png")], check=True)


def main() -> None:
    if CUBE.exists() or not (OUT / "study_region_sample_composite.png").exists():
        make_study_region()
    else:
        print("Skipping study-region render because data/cube.zarr is not local.")
    make_architecture()
    print(f"Wrote poster assets to {OUT}")


if __name__ == "__main__":
    main()
