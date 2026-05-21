"""
Chain 4: ESP-measured offshore particulate DA as a feature for WA sites.

ESP (Environmental Sample Processor) at NEMO mooring measured particulate
DA in seawater every few days during 2016-2018 (Moore et al. 2021) and
2021-2023 (NWFSC summary). Coverage is intermittent but during active periods
this is the directly-measured upstream toxin signal.

OAD correlation with ESP pDA was r=+0.33 (§16) — proving the offshore signal
is real. The challenge is the 24 km transport gap + 1-2 week bioaccumulation
that broke the satellite-to-beach chain. Direct ESP pDA might bridge that gap
better because it's the actual toxin (not a proxy for bloom intensity).

New features per (date, site in WA region):
  - esp_pda           (particulate DA at NEMO mooring, ng/L, lag-safe)
  - esp_pda_lag1w
  - esp_pda_lag2w
  - esp_pda_30day_max
  - esp_pda_available (binary 1/0 for whether ESP was deployed; signals data
                       presence to the model so it can learn to discount NaN)

Data sources:
  /Users/ansonchen/Downloads/ChaBa ESP database.xlsx        (2016-2018)
  /Users/ansonchen/Downloads/Summary NWFSC 2021_2023 ...    (2021-2023, harder to parse)

For v1 of this chain, use just the ChaBa file (2016-2018). NWFSC extension
needs the Array ID parsed for dates ("da21aug11..." → 2021-08-11).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

CHAIN_NAME = "esp_offshore_pda"
NEW_FEATURES = [
    "esp_pda",
    "esp_pda_lag1w",
    "esp_pda_lag2w",
    "esp_pda_30day_max",
    "esp_pda_available",
]
TARGET_SITES = ["Kalaloch", "Quinault", "Copalis", "Twin Harbors",
                "Long Beach", "Clatsop Beach", "Cannon Beach"]
LEAK_SHIFT_DAYS = 0  # ESP is in-situ, no satellite-composite lag

ESP_XLSX = "/Users/ansonchen/Downloads/ChaBa ESP database.xlsx"


def _load_esp_pda_daily() -> pd.DataFrame:
    esp = pd.read_excel(ESP_XLSX, sheet_name="cELISA")
    esp["Date"] = pd.to_datetime(esp["Date"])
    out = esp[["Date", "DA concentration (ng/L)"]].rename(
        columns={"Date": "date", "DA concentration (ng/L)": "esp_pda"}
    )
    # Daily aggregation (multiple samples per day occasionally)
    out = out.groupby("date", as_index=False)["esp_pda"].mean()
    out["esp_pda_available"] = 1
    return out


def add_features(df_in: pd.DataFrame) -> pd.DataFrame:
    esp = _load_esp_pda_daily().sort_values("date")
    df = df_in.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["site", "date"]).reset_index(drop=True)

    # ASOF merge: for each row date R, take most recent ESP sample within past 30 days.
    # Match-by date (no per-site needed; ESP is one offshore series).
    # Apply to ALL rows globally then propagate.
    unique_dates = pd.DataFrame({"date": df["date"].drop_duplicates().sort_values().values})
    asof = pd.merge_asof(
        unique_dates, esp,
        on="date", direction="backward", tolerance=pd.Timedelta(days=30),
    )
    out = df.merge(asof, on="date", how="left")
    out["esp_pda_available"] = out["esp_pda_available"].fillna(0).astype(int)

    pieces = []
    for site, sub in out.groupby("site", group_keys=False):
        sub = sub.sort_values("date").reset_index(drop=True).copy()
        sub["esp_pda_lag1w"] = sub["esp_pda"].shift(1)
        sub["esp_pda_lag2w"] = sub["esp_pda"].shift(2)
        sub["esp_pda_30day_max"] = sub["esp_pda"].rolling(window=4, min_periods=1).max()
        pieces.append(sub)
    out = pd.concat(pieces).sort_values(["site", "date"]).reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%m/%d/%Y")
    return out
