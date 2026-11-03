"""
The API's precomputed store: Delta tables built on Spark, read with deltalake.

One directory (``DATECT_PRECOMPUTED_DIR``, the lake's ``gold/``) holds four tables:

    forecasts      generate_single_forecast output per (site, date, task, model), as
                   lossless JSON (:func:`encode`); datect_spark/forecast_serving.py
    retrospective  /api/retrospective records per (task, model)
    spectral       /api/visualizations/spectral plots per site ("all_sites" for all)
    correlation    /api/visualizations/correlation heatmap per site
                   (the last three: datect_spark/precompute_store.py)

The API loads them once with ``deltalake`` (delta-rs: Rust, no JVM) into in-memory
dicts and answers from them. A missing table or key, or a row built from other code,
config, data or library versions (``config_hash``), falls back to computing on demand.
Unset ``DATECT_PRECOMPUTED_DIR`` = always compute.

``python -m backend.precomputed sample <dir> --site S --date D`` writes a small
forecasts table without Spark (for local checks and CI).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
import platform
import re
import threading
from typing import NamedTuple

import numpy as np
import pandas as pd

from backend.records import _dec, _enc, decode, encode, same  # noqa: F401 (re-exported)

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE_ENV = "DATECT_PRECOMPUTED_DIR"
COLUMNS = {
    "forecasts": ["site", "forecast_date", "task", "model_type", "result_json", "model_version", "config_hash"],
    "retrospective": ["task", "model_type", "records_json", "config_hash"],
    "spectral": ["site", "plots_json", "config_hash"],
    "correlation": ["site", "plot_json", "config_hash"],
}
ALL_SITES = "all_sites"


# ── What a stored row depends on ─────────────────────────────────────────────

# DATECT_* variables the engine reads that never change a stored row
NOT_ROW_INPUTS = {"DATECT_PARALLEL_BACKEND", "DATECT_TEMPORAL_HOLDOUT_CUTOFF"}


def config_hash(repo_root: str = REPO_ROOT) -> str:
    """sha256 over everything a stored row depends on: the engine, config, chart and
    row-format source (backend/records.py), the tuned hyperparameters, the input data,
    the DATECT_* variables the engine reads (and the files the *_JSON ones point at),
    the model library versions, the CPU architecture and the Python version. A row
    carrying another hash is not served.

    It is conservative: setting a variable to its default value also changes it."""
    import scipy
    import sklearn
    import xgboost

    sources = sorted(glob.glob(os.path.join(repo_root, "forecasting", "*.py")))
    sources += [os.path.join(repo_root, "config.py"), os.path.join(repo_root, "backend", "visualizations.py"),
                os.path.join(repo_root, "backend", "records.py")]
    data = [os.path.join(repo_root, "config", "tuned_hyperparameters.json"),
            os.path.join(repo_root, "data", "processed", "final_output.parquet")]
    data += sorted(glob.glob(os.path.join(repo_root, "data", "raw", "da-input", "*.csv")))
    h = hashlib.sha256()
    env_names = set()
    for path in sources + data:
        with open(path, "rb") as f:
            content = f.read()
        h.update(os.path.relpath(path, repo_root).encode() + b"\0" + content + b"\0")
        if path in sources:
            env_names |= {n.decode() for n in re.findall(rb"DATECT_[A-Z_]+", content)}
    for name in sorted(env_names - NOT_ROW_INPUTS):
        value = os.environ.get(name)
        h.update(f"{name}={value if value is not None else '<unset>'}\0".encode())
        if name.endswith("_JSON") and value and os.path.isfile(value):
            with open(value, "rb") as f:
                h.update(f.read() + b"\0")
    for module in (np, pd, sklearn, xgboost, scipy):
        h.update(f"{module.__name__}=={module.__version__}\0".encode())
    h.update(f"machine={platform.machine()}\0".encode())
    h.update(f"python={'.'.join(platform.python_version_tuple()[:2])}\0".encode())
    return h.hexdigest()


# ── The in-memory store ──────────────────────────────────────────────────────

class Hit(NamedTuple):
    result: dict | None  # None: the engine had too little data, as on demand
    model_version: str


class Store:
    """The four tables of one store directory, indexed in memory."""

    def __init__(self, root: str, expected_hash: str | None = None):
        self.root = root
        self.expected_hash = expected_hash or config_hash()
        self.tables: dict[str, dict] = {}
        self._index: dict[str, dict] = {name: {} for name in COLUMNS}
        for name, columns in COLUMNS.items():
            path = os.path.join(root, name)
            if not os.path.isdir(path):
                continue  # an optional table: its requests are computed
            try:
                self._load(name, columns, path)
            except Exception as exc:  # one unreadable table must not disable the others
                self._index[name] = {}
                self.tables[name] = {"error": f"{type(exc).__name__}: {exc}"}
                logger.warning("Precomputed table %s unreadable (%s); computing on demand", path, exc)

    def _load(self, name: str, columns: list[str], path: str) -> None:
        from deltalake import DeltaTable

        dt = DeltaTable(path)
        rows = dt.to_pyarrow_table(columns=columns).to_pydict()
        stale: dict[str, int] = {}
        for values in zip(*(rows[c] for c in columns)):
            row = dict(zip(columns, values))
            if name == "forecasts":
                if row["result_json"] is None:
                    continue  # the engine raised; on demand it raises again
                key = (row["site"], row["forecast_date"].isoformat(), row["task"], row["model_type"])
                payload = (row["result_json"], row["model_version"])
            else:  # key columns, then the JSON payload, then the hash
                key, payload = tuple(row[c] for c in columns[:-2]), row[columns[-2]]
                if payload is None:
                    continue
            if row["config_hash"] != self.expected_hash:
                stale[row["config_hash"]] = stale.get(row["config_hash"], 0) + 1
                continue
            self._index[name][key] = payload
        self.tables[name] = {"version": dt.version(), "entries": len(self._index[name]),
                             "stale_rows": sum(stale.values()), "stale_config_hashes": stale}

    @property
    def stale_rows(self) -> int:
        return sum(t.get("stale_rows", 0) for t in self.tables.values())

    def forecast(self, site: str, forecast_date, task: str, model_type: str) -> Hit | None:
        key = (site, pd.Timestamp(forecast_date).date().isoformat(), task, model_type)
        entry = self._index["forecasts"].get(key)
        if entry is None:
            return None
        try:
            return Hit(decode(entry[0]), entry[1])
        except Exception as exc:  # a bad row is a miss: compute on demand
            logger.warning("Unreadable precomputed forecast %s (%s)", key, exc)
            return None

    def _json(self, table: str, key: tuple):
        text = self._index[table].get(key)
        if text is None:
            return None
        try:
            return json.loads(text)
        except Exception as exc:
            logger.warning("Unreadable precomputed %s row %s (%s)", table, key, exc)
            return None

    def retrospective(self, task: str, model_type: str) -> list | None:
        return self._json("retrospective", (task, model_type))

    def spectral(self, site: str | None) -> list | None:
        return self._json("spectral", (site or ALL_SITES,))

    def correlation(self, site: str) -> dict | None:
        return self._json("correlation", (site,))


_store: Store | None = None
_loaded_root: str | None = None
_load_error: str | None = None
_lock = threading.Lock()


def load_from_env() -> Store | None:
    """Load the store named by DATECT_PRECOMPUTED_DIR once per path; None (compute on
    demand) when it is unset or cannot be read."""
    global _store, _loaded_root, _load_error
    root = os.environ.get(STORE_ENV)
    if not root:
        return None
    with _lock:
        if _loaded_root != root:
            _loaded_root, _store, _load_error = root, None, None
            try:
                _store = Store(root)
            except Exception as exc:  # serving must never break the API
                _load_error = f"{type(exc).__name__}: {exc}"
                logger.warning("Precomputed store unavailable (%s); computing on demand", _load_error)
                return None
            empty = not any(t.get("entries") for t in _store.tables.values())
            log = logger.warning if _store.stale_rows or empty else logger.info
            log("Precomputed store %s: %s; this API's config hash %s", root,
                {n: t.get("error") or (t["entries"], f"{t['stale_rows']} stale")
                 for n, t in _store.tables.items()} or "no tables",
                _store.expected_hash[:12])
        return _store


def status() -> dict | None:
    """What /health reports: None without a store, else each table's size and staleness."""
    store = load_from_env()
    if store is None:
        root = os.environ.get(STORE_ENV)
        return {"dir": root, "exists": os.path.isdir(root), "tables": {}, "error": _load_error} if root else None
    return {"dir": store.root, "exists": os.path.isdir(store.root),
            "expected_config_hash": store.expected_hash, "tables": store.tables}


def check_store(root: str, require: tuple[str, ...] = tuple(COLUMNS)) -> list[str]:
    """Why ``root`` is not a complete store current for this environment (empty: it
    is). The image build runs this, so a deploy never ships a store whose requests
    would all be recomputed (a retrospective takes minutes)."""
    if not os.path.isdir(root):
        return [f"{root}: no such directory"]
    store = Store(root)
    problems = []
    for name in require:
        t = store.tables.get(name)
        if t is None:
            problems.append(f"{name}: missing")
        elif "error" in t:
            problems.append(f"{name}: unreadable ({t['error']})")
        elif t["stale_rows"]:
            problems.append(f"{name}: {t['stale_rows']} stale rows (built from other code, config, data or "
                            f"library versions; this environment's config hash is {store.expected_hash[:12]})")
        elif not t["entries"]:
            problems.append(f"{name}: empty")
    return problems


def forecast(site: str, forecast_date, task: str, model_type: str) -> Hit | None:
    store = load_from_env()
    return store.forecast(site, forecast_date, task, model_type) if store is not None else None


def retrospective(task: str, model_type: str) -> list | None:
    store = load_from_env()
    return store.retrospective(task, model_type) if store is not None else None


def spectral(site: str | None) -> list | None:
    store = load_from_env()
    return store.spectral(site) if store is not None else None


def correlation(site: str) -> dict | None:
    store = load_from_env()
    return store.correlation(site) if store is not None else None


def table_info() -> dict:
    """Where retrospective results came from, for /api/retrospective's response."""
    store = load_from_env()
    if store is None or "retrospective" not in store.tables:
        return {"dir": os.environ.get(STORE_ENV), "table_version": None}
    return {"dir": store.root, "table_version": store.tables["retrospective"]["version"]}


# ── A small forecasts table without Spark (local checks, CI) ────────────────

def write_sample(root: str, pairs: list[tuple[str, str]], model_type: str = "ensemble") -> int:
    """Compute regression + classification forecasts for (site, YYYY-MM-DD) pairs with
    the engine and write them as the store's forecasts table."""
    import datetime

    from deltalake import write_deltalake

    import config
    from forecasting import ForecastEngine

    engine = ForecastEngine(validate_on_init=False)
    chash = config_hash()
    rows = []
    for site, day in pairs:
        for task in ("regression", "classification"):
            model = "logistic" if model_type == "linear" and task == "classification" else model_type
            result = engine.generate_single_forecast(config.FINAL_OUTPUT_PATH, pd.Timestamp(day), site, task, model)
            rows.append({"site": site, "forecast_date": datetime.date.fromisoformat(day), "task": task,
                         "model_type": model, "result_json": encode(result),
                         "model_version": "sample", "config_hash": chash})
    write_deltalake(os.path.join(root, "forecasts"), pd.DataFrame(rows), mode="overwrite")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precomputed store tools")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="write a small forecasts table without Spark")
    s.add_argument("root")
    s.add_argument("--site", required=True)
    s.add_argument("--date", action="append", required=True)
    s.add_argument("--model", default="ensemble")
    sub.add_parser("hash", help="print this environment's config hash")
    c = sub.add_parser("check", help="exit 1 unless the store is complete and current here")
    c.add_argument("root")
    args = parser.parse_args()
    if args.cmd == "check":
        args.root = os.path.abspath(args.root)
    os.chdir(REPO_ROOT)  # config paths are relative to the repo root
    if args.cmd == "sample":
        n = write_sample(args.root, [(args.site, d) for d in args.date], args.model)
        print(f"wrote {n} forecasts to {os.path.join(args.root, 'forecasts')} (config hash {config_hash()[:12]})")
    elif args.cmd == "check":
        problems = check_store(args.root)
        for problem in problems:
            print(f"precomputed store: {problem}")
        if problems:
            raise SystemExit(1)
        print(f"precomputed store {args.root} is complete and current (config hash {config_hash()[:12]})")
    else:
        print(config_hash())


if __name__ == "__main__":
    main()
