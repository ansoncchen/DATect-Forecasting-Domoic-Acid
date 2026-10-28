"""
Parsing of raw domoic acid (DA) and Pseudo-nitzschia (PN) measurement files.

One implementation shared by dataset-creation.py (the weekly training table),
raw_data_forecaster (raw evaluation targets) and the API (latest reading):

- Below-detection DA readings such as ``"<1"`` become half the detection limit
  (LOD/2, the standard substitution for non-detects: ``"<1"`` -> 0.5). They used
  to be dropped, which also let the weekly table decay-fill those weeks from an
  older, higher reading instead of recording a near-zero measurement.
- PN cell counts written with thousands separators (``"12,345"``) are parsed.
  They used to be dropped, which lost most bloom-peak counts at the
  Washington sites.
- Dates are parsed after stripping surrounding whitespace.
- Anything else that is not a number becomes NaN / NaT.
"""
from __future__ import annotations

import pandas as pd

_BELOW_DETECTION = r"^<\s*(\d+(?:\.\d+)?)$"
_THOUSANDS = r"^\d{1,3}(?:,\d{3})+(?:\.\d+)?$"


def parse_da_values(values) -> pd.Series:
    """DA in ug/g; ``"<x"`` (below detection limit x) becomes x / 2."""
    raw = pd.Series(values, copy=False)
    out = pd.to_numeric(raw, errors="coerce").astype(float)
    limit = raw.astype(str).str.strip().str.extract(_BELOW_DETECTION, expand=False)
    below = limit.notna()
    out[below] = limit[below].astype(float) / 2
    return out


def parse_pn_values(values) -> pd.Series:
    """PN in cells/L; accepts thousands separators (``"1,234,567"``)."""
    raw = pd.Series(values, copy=False)
    out = pd.to_numeric(raw, errors="coerce").astype(float)
    text = raw.astype(str).str.strip()
    grouped = text.str.match(_THOUSANDS)
    out[grouped] = text[grouped].str.replace(",", "", regex=False).astype(float)
    return out


def parse_dates(values) -> pd.Series:
    """Dates, tolerating stray whitespace (e.g. ``" 7/11/19"``)."""
    raw = pd.Series(values, copy=False)
    if pd.api.types.is_datetime64_any_dtype(raw):
        return raw
    return pd.to_datetime(raw.astype("string").str.strip(), errors="coerce")
