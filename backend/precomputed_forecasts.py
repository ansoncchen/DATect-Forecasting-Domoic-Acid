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
# tuples are tagged so decode() returns the same types the engine returned.

def _enc(obj):
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, float)) and not isinstance(obj, (np.integer, np.floating)):
        return obj
    if isinstance(obj, pd.Timestamp):
        return {"__ts__": obj.isoformat()}
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return {"__np__": obj.dtype.str, "v": obj.item()}
    if isinstance(obj, np.ndarray):
        return {"__nd__": obj.dtype.str, "v": [_enc(x) for x in obj.tolist()]}
    if isinstance(obj, tuple):
        return {"__tuple__": [_enc(x) for x in obj]}
    if isinstance(obj, list):
        return [_enc(x) for x in obj]
    if isinstance(obj, dict):
        if not all(isinstance(k, str) for k in obj):
            raise TypeError("only str keys are supported")
        return {k: _enc(v) for k, v in obj.items()}
    raise TypeError(f"cannot encode {type(obj).__name__}")


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
        self.stale_rows = 0
        for site, day, task, model, text, version, row_hash in zip(*(rows[c] for c in COLUMNS)):
            if text is None:  # the engine raised; on demand it raises again
                continue
            if row_hash != self.expected_hash:
                self.stale_rows += 1
                continue
            self._index[(site, day.isoformat(), task, model)] = (text, version)

    def __len__(self) -> int:
        return len(self._index)

    def get(self, site: str, forecast_date, task: str, model_type: str) -> Hit | None:
        entry = self._index.get((site, pd.Timestamp(forecast_date).date().isoformat(), task, model_type))
        if entry is None:
            return None
        return Hit(decode(entry[0]), entry[1])


_store: PrecomputedForecasts | None = None
_loaded_uri: str | None = None
_lock = threading.Lock()


def load_from_env() -> PrecomputedForecasts | None:
    """Load the table named by DATECT_FORECAST_TABLE once per URI; None (compute on
    demand) when it is unset or cannot be read."""
    global _store, _loaded_uri
    uri = os.environ.get(TABLE_ENV)
    if not uri:
        return None
    with _lock:
        if _loaded_uri != uri:
            _loaded_uri, _store = uri, None
            try:
                _store = PrecomputedForecasts(uri)
                logger.info("Precomputed forecasts: %d entries from %s (version %d, %d stale rows ignored)",
                            len(_store), uri, _store.table_version, _store.stale_rows)
            except Exception as exc:  # serving must never break the API
                logger.warning("Precomputed forecasts unavailable (%s); computing on demand", exc)
        return _store


def lookup(site: str, forecast_date, task: str, model_type: str) -> Hit | None:
    store = load_from_env()
    return store.get(site, forecast_date, task, model_type) if store is not None else None
