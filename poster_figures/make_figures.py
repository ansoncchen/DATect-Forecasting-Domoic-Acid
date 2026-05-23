"""Generate poster figures: (1) study-region map + sample composite, (2) AE architecture."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LogNorm
from pathlib import Path

OUT = Path(__file__).parent

# ---------- DATect sites (lat, lon) ----------
SITES = {
    "Kalaloch":       (47.61, -124.37),
    "Quinault":       (47.34, -124.30),
    "Copalis":        (47.11, -124.18),
    "Twin Harbors":   (46.86, -124.11),
    "Long Beach":     (46.50, -124.07),
    "Clatsop Beach":  (46.02, -123.94),
    "Cannon Beach":   (45.89, -123.97),
    "Newport":        (44.64, -124.07),
    "Coos Bay":       (43.37, -124.30),
    "Gold Beach":     (42.40, -124.42),
}

# OAD regions (from src/regions.py)
REGIONS = [
    ("Olympic Coast",      47.0, 49.0, -125.5, -123.5, "#1f77b4"),
    ("SW WA / Long Beach", 45.5, 47.0, -124.5, -123.0, "#2ca02c"),
    ("Central Oregon",     43.5, 45.5, -125.0, -123.5, "#ff7f0e"),
    ("Southern OR / N CA", 41.0, 43.5, -125.5, -123.5, "#d62728"),
]

# ESP mooring (Juan de Fuca eddy, approx)
ESP = ("ESP mooring (JdF eddy)", 48.32, -125.78)


def figure1_map_and_frame():
    d = np.load(OUT / "sample_frame.npz")
    chla = d["data"][0]    # channel 0 = chla
    mask = d["mask"]
    lat, lon = d["lat"], d["lon"]
    date = str(d["date"])
    chla_masked = np.where(mask & np.isfinite(chla) & (chla > 0), chla, np.nan)

    fig, (axM, axF) = plt.subplots(1, 2, figsize=(13, 8),
                                    gridspec_kw={"width_ratios": [1, 1.05]})

    # ---- LEFT: study-region map ----
    axM.set_facecolor("#e8f1f7")
    # crude coastline shading: shade lon > -124.2 as land-ish (right of map)
    # (avoids cartopy dep; the data frame on the right shows real coastline)
    axM.add_patch(Rectangle((-123.5, 41), 3.5, 8, color="#d9d2bf", zorder=0))
    for name, lat0, lat1, lon0, lon1, c in REGIONS:
        axM.add_patch(Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0,
                                fill=True, facecolor=c, alpha=0.18,
                                edgecolor=c, linewidth=1.8, zorder=1))
        # region name on the LEFT edge to avoid colliding with site labels
        axM.text(lon0 - 0.08, (lat0 + lat1) / 2, name, fontsize=8.5, color=c,
                 weight="bold", ha="right", va="center", zorder=3)
    # DA sites
    for s, (la, lo) in SITES.items():
        axM.plot(lo, la, "o", ms=7, color="#222", mec="white", mew=1.2, zorder=4)
        axM.annotate(s, (lo, la), xytext=(6, 0), textcoords="offset points",
                     fontsize=8, va="center", zorder=4)
    # ESP mooring
    axM.plot(ESP[2], ESP[1], "*", ms=18, color="#fff200", mec="black", mew=1.2, zorder=5)
    axM.annotate(ESP[0], (ESP[2], ESP[1]), xytext=(-8, 12),
                 textcoords="offset points", fontsize=8.5, ha="right",
                 weight="bold", zorder=5)

    axM.set_xlim(-126.5, -122.8)
    axM.set_ylim(40.8, 49.2)
    axM.set_xlabel("Longitude (°E)")
    axM.set_ylabel("Latitude (°N)")
    axM.set_title("A. Study region — 10 DA beaches + 4 OAD regions", fontsize=11, weight="bold")
    axM.grid(alpha=0.25, linestyle=":")
    axM.set_aspect(1.3)

    # ---- RIGHT: sample chla composite frame ----
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    im = axF.imshow(chla_masked, origin="lower", extent=extent,
                    cmap="viridis", norm=LogNorm(vmin=0.05, vmax=20), aspect="auto")
    # overlay region boxes for context
    for name, lat0, lat1, lon0, lon1, c in REGIONS:
        axF.add_patch(Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0,
                                fill=False, edgecolor="white", linewidth=1.4, zorder=3))
    axF.set_xlim(-131, -120.8)
    axF.set_ylim(41, 49)
    axF.set_xlabel("Longitude (°E)")
    axF.set_title(f"B. Sample 8-day composite (chl-a, mg/m³) — {date}", fontsize=11, weight="bold")
    cb = plt.colorbar(im, ax=axF, shrink=0.78, pad=0.02)
    cb.set_label("chlorophyll-a (mg/m³)")
    ocean = mask.sum()
    valid_frac = float(np.isfinite(chla_masked).sum() / max(ocean, 1))
    axF.text(0.02, 0.02, f"valid ocean pixels: {valid_frac:.0%}\n4 channels: chla, Kd490, nFLH, SST",
             transform=axF.transAxes, fontsize=8.5, color="white",
             bbox=dict(boxstyle="round", facecolor="black", alpha=0.55))

    plt.tight_layout()
    out = OUT / "fig1_study_region_and_composite.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def figure2_architecture():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis("off")

    def block(x, y, w, h, text, fc="#cfe2f3", ec="#1f4e79", fontsize=9.5, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                     facecolor=fc, edgecolor=ec, linewidth=1.6))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, weight=weight, wrap=True)

    def arrow(x1, y1, x2, y2, label=None):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                      arrowstyle="-|>", mutation_scale=18,
                                      color="#333", linewidth=1.6))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.18, label, ha="center", fontsize=8.5,
                    style="italic", color="#444")

    # Input
    block(0.3, 2.3, 1.9, 1.4,
          "Input cube\n4 ch × 4 frames\n321 × 409",
          fc="#fff2cc", ec="#bf9000", weight="bold")
    ax.text(1.25, 1.95, "chla, Kd490,\nnFLH, SST",
            ha="center", fontsize=7.5, color="#7a5a00", style="italic")

    # Masking
    block(2.7, 2.3, 1.7, 1.4,
          "Random mask\n(70 % pixels\nhidden)",
          fc="#f4cccc", ec="#990000")

    # Encoder
    block(4.9, 2.3, 2.0, 1.4,
          "3D Conv encoder\n(4 ch → 32 → 64 → 128)",
          fc="#cfe2f3", ec="#1f4e79", weight="bold")

    # Latent
    block(7.4, 2.6, 1.3, 0.8,
          "Latent\nl = 32",
          fc="#d9ead3", ec="#274e13", weight="bold")

    # Decoder
    block(9.2, 2.3, 2.0, 1.4,
          "3D Conv decoder\n(128 → 64 → 32 → 4)",
          fc="#cfe2f3", ec="#1f4e79", weight="bold")

    # Reconstruction
    block(11.7, 2.3, 2.0, 1.4,
          "Reconstruction\n4 ch × 4 frames",
          fc="#fff2cc", ec="#bf9000", weight="bold")

    # Arrows along the main pipeline
    arrow(2.2, 3.0, 2.7, 3.0)
    arrow(4.4, 3.0, 4.9, 3.0)
    arrow(6.9, 3.0, 7.4, 3.0)
    arrow(8.7, 3.0, 9.2, 3.0)
    arrow(11.2, 3.0, 11.7, 3.0)

    # Loss / anomaly score callout
    block(7.0, 0.3, 5.0, 1.2,
          "Anomaly score  =  per-pixel  ||input − reconstruction||²\n"
          "→ pooled into 5 region scalars (Olympic, SW WA, C. OR, S. OR / N. CA, Overall)",
          fc="#ead1dc", ec="#741b47", fontsize=9.5)
    arrow(12.7, 2.3, 11.5, 1.5)
    arrow(1.25, 2.3, 7.0, 0.9, label="compare")

    # Training caption
    ax.text(7, 5.6, "Masked-autoencoder pretraining on 22 yr of MODIS Aqua "
                    "(2003–2024, ~4,700 daily 8-day composites)",
            ha="center", fontsize=11, weight="bold", color="#1f4e79")
    ax.text(7, 5.15, "Model learns to fill in hidden pixels from visible context → "
                    "reconstruction error = how surprising today's ocean state is",
            ha="center", fontsize=9.5, style="italic", color="#444")

    # Bottom-left training detail
    ax.text(0.3, 0.4,
            "Training: MAE loss, Adam, 50 epochs\n"
            "Checkpoint: ae_3d_l32_c4_t4_s42_mae050",
            fontsize=8, color="#555", family="monospace")

    plt.tight_layout()
    out = OUT / "fig2_architecture.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    figure1_map_and_frame()
    figure2_architecture()
