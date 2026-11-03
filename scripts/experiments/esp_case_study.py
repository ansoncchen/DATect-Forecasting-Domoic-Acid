#!/usr/bin/env python3
"""ESP case study (docs/FORECAST_EXPERIMENTS.md §7): did detectable offshore particulate DA
in the 14 days before the anchor precede Washington onsets, against ESP-covered quiet weeks?

Descriptive only. The ESP files were shared privately and are never committed; this
script reads them from the paths given and writes counts only.

    python -I scripts/experiments/esp_case_study.py --chaba <ChaBa ESP database.xlsx> \
        --nwfsc <Summary NWFSC 2021_2023 data final_11022023.xls>
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from onset_report import OUT  # noqa: E402
from onset_study import COMMITTED, REGIONS  # noqa: E402

WINDOW_DAYS = 14
PERIODS = {"2016–2018 (design)": ("2016-01-01", "2019-01-01"), "2021 (check)": ("2021-01-01", "2022-01-01"),
           "2022–2023 (check, holdout years)": ("2022-01-01", "2024-01-01")}
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def load_chaba(path: str) -> pd.DataFrame:
    e = pd.read_excel(path, sheet_name="cELISA")
    conc, llod = e["DA concentration (ng/L)"].astype(float), e[[c for c in e.columns if c.strip() == "LLOD"][0]].astype(float)
    return pd.DataFrame({"date": pd.to_datetime(e["Date"]).dt.normalize(), "pda": conc, "detect": conc > llod})


def load_nwfsc(path: str) -> pd.DataFrame:
    s = pd.read_excel(path, header=None)
    rows = []
    for aid, conc, llod in zip(s[0].astype(str), s[1], s[2]):
        m = re.match(r"^da(\d{2})([a-z]{3})(\d{2})(\d{2})h", aid.strip().lower())
        if not m or m.group(2) not in MONTHS:
            continue
        day = pd.Timestamp(2000 + int(m.group(1)), MONTHS[m.group(2)], int(m.group(3)))
        conc, llod = float(conc), float(llod)
        rows.append({"date": day, "pda": conc, "detect": conc > llod})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chaba", required=True)
    ap.add_argument("--nwfsc", required=True)
    args = ap.parse_args()
    esp = pd.concat([load_chaba(args.chaba), load_nwfsc(args.nwfsc)], ignore_index=True).sort_values("date")
    d = pd.read_parquet(ROOT / COMMITTED)
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["site"].isin(REGIONS["WA"]) & (d["naive_prediction"] < 20)].reset_index(drop=True)
    d["onset"] = d["actual_da"] >= 20
    anchor = d["date"] - pd.Timedelta(days=7)
    covered, detect, pmax = [], [], []
    ed, ev, edet = esp["date"].to_numpy(), esp["pda"].to_numpy(), esp["detect"].to_numpy()
    for a in anchor:
        m = (ed >= np.datetime64(a - pd.Timedelta(days=WINDOW_DAYS))) & (ed <= np.datetime64(a))
        covered.append(bool(m.any()))
        detect.append(bool(edet[m].any()))
        pmax.append(float(ev[m].max()) if m.any() else np.nan)
    d["covered"], d["detect"], d["pda_max"] = covered, detect, pmax
    print(f"ESP samples: {len(esp)} ({esp['date'].min().date()} to {esp['date'].max().date()}), "
          f"{int(esp['detect'].sum())} detectable")
    rows = []
    for name, (a, b) in PERIODS.items():
        w = d[(d["date"] >= a) & (d["date"] < b)]
        c = w[w["covered"]]
        on, qu = c[c["onset"]], c[~c["onset"]]
        table = [[int(on["detect"].sum()), int((~on["detect"]).sum())], [int(qu["detect"].sum()), int((~qu["detect"]).sum())]]
        p = fisher_exact(table)[1] if len(c) and (np.array(table).sum(axis=0) > 0).all() else np.nan
        det, nod = c[c["detect"]], c[~c["detect"]]
        rows.append({"period": name, "wa_onsets": int(w["onset"].sum()), "wa_quiet": int((~w["onset"]).sum()),
                     "covered_onsets": len(on), "covered_quiet": len(qu),
                     "onsets_with_detectable_pda": table[0][0], "quiet_with_detectable_pda": table[1][0],
                     "onset_rate_if_detectable": float(det["onset"].mean()) if len(det) else np.nan,
                     "onset_rate_if_not": float(nod["onset"].mean()) if len(nod) else np.nan,
                     "median_max_pda_onsets": float(on["pda_max"].median()) if len(on) else np.nan,
                     "median_max_pda_quiet": float(qu["pda_max"].median()) if len(qu) else np.nan,
                     "fisher_p": p})
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    # Only counts and rates are committed; the pDA medians summarise a handful of private samples.
    out.drop(columns=["median_max_pda_onsets", "median_max_pda_quiet"]).to_csv(
        OUT / "esp_case_study.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    print(out.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
