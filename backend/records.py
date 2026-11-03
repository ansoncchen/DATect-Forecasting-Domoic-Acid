"""
How stored rows are written: the JSON the API returns for retrospective records,
and the lossless codec for generate_single_forecast output.

Both the API (computing on demand) and the Spark builds of the precomputed store
(datect_spark/) use these functions, and backend/precomputed.config_hash covers this
file: a change here makes stored rows stale instead of serving an old format.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd


def clean_for_json(obj):
    """
    Clean values for JSON serialization - handles inf/nan, numpy types, and nested structures.
    
    Args:
        obj: Value to clean (can be dict, list, numpy array, scalar, etc.)
    
    Returns:
        JSON-serializable value
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.head(10).to_dict('records') if not obj.empty else []
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    # Check for pandas NA values
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def retrospective_records(results_df: pd.DataFrame) -> list:
    """run_retrospective_evaluation's rows as the JSON records /api/retrospective
    returns (canonical keys, floats cleaned); the precompute stores these too."""
    records = []
    for _, row in results_df.iterrows():
        record = {
            "date": row['date'].strftime('%Y-%m-%d') if pd.notnull(row['date']) else None,
            "site": row['site'],
            "actual_da": clean_for_json(row['actual_da']) if 'actual_da' in row and pd.notnull(row['actual_da']) else None,
            "predicted_da": clean_for_json(row['predicted_da']) if 'predicted_da' in row and pd.notnull(row['predicted_da']) else None,
            "actual_category": clean_for_json(row['actual_category']) if 'actual_category' in row and pd.notnull(row['actual_category']) else None,
            "predicted_category": clean_for_json(row['predicted_category']) if 'predicted_category' in row and pd.notnull(row['predicted_category']) else None
        }
        if 'anchor_date' in results_df.columns and pd.notnull(row.get('anchor_date', None)):
            record['anchor_date'] = row['anchor_date'].strftime('%Y-%m-%d')
        if 'spike_probability' in results_df.columns and pd.notnull(row.get('spike_probability', None)):
            record['spike_probability'] = clean_for_json(row['spike_probability'])
            record['spike_alert'] = bool(row.get('spike_alert', False))
        records.append(record)
    return records


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
