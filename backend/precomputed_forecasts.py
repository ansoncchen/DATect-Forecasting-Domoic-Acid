"""
Forecasts precomputed on Spark, served from a Delta table.

``datect_spark/forecast_serving.py`` stores the output of
``RawForecastEngine.generate_single_forecast`` for each (site, date, task, model) in
``gold/forecasts`` as lossless JSON (:func:`encode`). The API loads the table once
with ``deltalake`` (delta-rs: Rust, no JVM) into an in-memory dict and answers from it;
a key the table lacks, or a table built from other code, config, data or library
versions (``config_hash``), falls back to computing the forecast on demand.

    DATECT_FORECAST_TABLE   path or URI of gold/forecasts; unset = always compute
"""
from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import re
import threading
from typing import NamedTuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE_ENV = "DATECT_FORECAST_TABLE"
COLUMNS = ["site", "forecast_date", "task", "model_type", "result_json", "model_version", "config_hash"]


# ── Lossless JSON for generate_single_forecast output ────────────────────────
# Floats round-trip exactly through json (repr); numpy scalars, timestamps and
# tuples are tagged so decode() returns the same types the engine returned. Types
# are matched exactly: anything else (subclasses, numpy strings, n-d arrays) raises,
# so the precompute stores an error row (a miss) instead of a lossy copy.
_TAGS = ("__ts__", "__np__", "__nd__", "__tuple__")


def _enc(obj):
    kind = type(obj)
    if obj is None or kind in (bool, str, int, float):
        return obj
    if kind is pd.Timestamp:
        return {"__ts__": obj.isoformat()}
    if isinstance(obj, np.generic) and obj.dtype.kind in "biuf":
        return {"__np__": obj.dtype.str, "v": obj.item()}
    if kind is np.ndarray and obj.ndim == 1 and obj.dtype.kind in "biuf":
        return {"__nd__": obj.dtype.str, "v": [_enc(x) for x in obj.tolist()]}
    if kind is tuple:
        return {"__tuple__": [_enc(x) for x in obj]}
    if kind is list:
        return [_enc(x) for x in obj]
    if kind is dict:
        if not all(type(k) is str and k not in _TAGS for k in obj):
            raise TypeError("dict keys must be str and not a codec tag")
        return {k: _enc(v) for k, v in obj.items()}
    raise TypeError(f"cannot encode {kind.__name__}")


def _dec(obj):
    if isinstance(obj, list):
        return [_dec(x) for x in obj]
    if not isinstance(obj, dict):
        return obj
    if "__ts__" in obj:
        return pd.Timestamp(obj["__ts__"])
    if "__np__" in obj:
        return np.dtype(obj["__np__"]).type(obj["v"])
    if "__nd__" in obj:
        return np.array(obj["v"], dtype=np.dtype(obj["__nd__"]))
    if "__tuple__" in obj:
        return tuple(_dec(x) for x in obj["__tuple__"])
    return {k: _dec(v) for k, v in obj.items()}


def encode(result: dict | None) -> str:
    return json.dumps(_enc(result), allow_nan=True)


def decode(text: str) -> dict | None:
    return _dec(json.loads(text))


def same(a, b) -> bool:
    """Exact equality with types (NaN equals NaN): the parity criterion."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
    if isinstance(a, (float, np.floating)) and a != a:
        return b != b
    return bool(a == b)


# ── What a forecast depends on ───────────────────────────────────────────────

def config_hash(repo_root: str = REPO_ROOT) -> str:
    """sha256 over everything a forecast depends on: the engine and config source,
    the tuned hyperparameters, the input data, the DATECT_* variables the engine
    reads, and the model library versions. A table whose rows carry another hash
    was built from something else and is not served."""
    import scipy
    import sklearn
    import xgboost

    sources = sorted(glob.glob(os.path.join(repo_root, "forecasting", "*.py")))
    sources.append(os.path.join(repo_root, "config.py"))
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
            env_names |= set(re.findall(rb"DATECT_[A-Z_]+", content))
    for name in sorted(env_names):
        h.update(name + b"=" + os.environ.get(name.decode(), "<unset>").encode() + b"\0")
    for module in (np, pd, sklearn, xgboost, scipy):
        h.update(f"{module.__name__}=={module.__version__}\0".encode())
    return h.hexdigest()


# ── The in-memory index ──────────────────────────────────────────────────────

class Hit(NamedTuple):
    result: dict | None  # None: the engine had too little data, as on demand
    model_version: str


class PrecomputedForecasts:
    """(site, date, task, model) -> encoded result, from one Delta table version."""

    def __init__(self, uri: str, expected_hash: str | None = None):
        from deltalake import DeltaTable

        dt = DeltaTable(uri)
        self.uri = uri
        self.table_version = dt.version()
        rows = dt.to_pyarrow_table(columns=COLUMNS).to_pydict()
        self.expected_hash = expected_hash or config_hash()
        self._index: dict[tuple, tuple[str, str]] = {}
        self.stale_hashes: dict[str, int] = {}
        for site, day, task, model, text, version, row_hash in zip(*(rows[c] for c in COLUMNS)):
            if text is None:  # the engine raised; on demand it raises again
                continue
            if row_hash != self.expected_hash:
                self.stale_hashes[row_hash] = self.stale_hashes.get(row_hash, 0) + 1
                continue
            self._index[(site, day.isoformat(), task, model)] = (text, version)
        self.stale_rows = sum(self.stale_hashes.values())

    def __len__(self) -> int:
        return len(self._index)

    def get(self, site: str, forecast_date, task: str, model_type: str) -> Hit | None:
        entry = self._index.get((site, pd.Timestamp(forecast_date).date().isoformat(), task, model_type))
        if entry is None:
            return None
        try:
            return Hit(decode(entry[0]), entry[1])
        except Exception as exc:  # a bad row is a miss: compute on demand
            logger.warning("Unreadable precomputed forecast %s %s %s %s (%s)",
                           site, forecast_date, task, model_type, exc)
            return None


_store: PrecomputedForecasts | None = None
_loaded_uri: str | None = None
_load_error: str | None = None
_lock = threading.Lock()


def load_from_env() -> PrecomputedForecasts | None:
    """Load the table named by DATECT_FORECAST_TABLE once per URI; None (compute on
    demand) when it is unset or cannot be read."""
    global _store, _loaded_uri, _load_error
    uri = os.environ.get(TABLE_ENV)
    if not uri:
        return None
    with _lock:
        if _loaded_uri != uri:
            _loaded_uri, _store, _load_error = uri, None, None
            try:
                _store = PrecomputedForecasts(uri)
            except Exception as exc:  # serving must never break the API
                _load_error = f"{type(exc).__name__}: {exc}"
                logger.warning("Precomputed forecasts unavailable (%s); computing on demand", _load_error)
                return None
            log = logger.warning if _store.stale_rows or not len(_store) else logger.info
            log("Precomputed forecasts: %d entries from %s (version %d); %d stale rows ignored "
                "(this API's config hash %s, stale rows carry %s)", len(_store), uri,
                _store.table_version, _store.stale_rows, _store.expected_hash[:12],
                {h[:12]: n for h, n in _store.stale_hashes.items()})
        return _store


def status() -> dict | None:
    """What /health reports: None without a table, else its size and staleness."""
    store = load_from_env()
    if store is None:
        uri = os.environ.get(TABLE_ENV)
        return {"uri": uri, "entries": 0, "error": _load_error} if uri else None
    return {"uri": store.uri, "table_version": store.table_version, "entries": len(store),
            "stale_rows": store.stale_rows, "expected_config_hash": store.expected_hash,
            "stale_config_hashes": store.stale_hashes}


def lookup(site: str, forecast_date, task: str, model_type: str) -> Hit | None:
    store = load_from_env()
    return store.get(site, forecast_date, task, model_type) if store is not None else None
