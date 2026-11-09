"""compare_golden: exact by default, with only explicitly declared exceptions."""
import importlib.util
from pathlib import Path

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "compare_golden", Path(__file__).resolve().parents[1] / "scripts" / "eval" / "compare_golden.py"
)
compare_golden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare_golden)

REL = "results/chronological/chronological_regression_ensemble_x.parquet"
COL = "predicted_category_logistic"


def _frame():
    return pd.DataFrame({"site": ["a", "b"], "predicted_da": [1.0, 2.0], COL: [None, None]})


def _write(root: Path, df: pd.DataFrame) -> Path:
    path = root / REL
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return root


def _pair(tmp_path, base_df, cand_df):
    return _write(tmp_path / "base", base_df), _write(tmp_path / "cand", cand_df)


def test_identical_runs_pass(tmp_path):
    base, cand = _pair(tmp_path, _frame(), _frame())
    assert compare_golden.compare(base, cand) == 0


def test_dropped_column_fails_unless_allowed(tmp_path, capsys):
    base, cand = _pair(tmp_path, _frame(), _frame().drop(columns=[COL]))
    assert compare_golden.compare(base, cand) != 0
    assert compare_golden.compare(base, cand, allow_dropped_columns=frozenset({COL})) == 0
    assert f"DROPPED (allowed): {REL}: {COL}" in capsys.readouterr().out


def test_unlisted_dropped_column_still_fails(tmp_path):
    base, cand = _pair(tmp_path, _frame(), _frame().drop(columns=[COL, "predicted_da"]))
    assert compare_golden.compare(base, cand, allow_dropped_columns=frozenset({COL})) != 0


def test_value_change_fails_even_with_allowances(tmp_path):
    changed = _frame().drop(columns=[COL])
    changed.loc[0, "predicted_da"] = 1.5
    base, cand = _pair(tmp_path, _frame(), changed)
    assert compare_golden.compare(base, cand, allow_dropped_columns=frozenset({COL})) != 0


def test_added_column_fails(tmp_path):
    base, cand = _pair(tmp_path, _frame(), _frame().assign(extra=1))
    assert compare_golden.compare(base, cand, allow_dropped_columns=frozenset({COL})) != 0


def test_missing_file_fails_unless_allowed(tmp_path, capsys):
    base = _write(tmp_path / "base", _frame())
    cand = tmp_path / "cand"
    cand.mkdir()
    assert compare_golden.compare(base, cand) != 0
    assert compare_golden.compare(base, cand, allow_missing=frozenset({REL})) == 0
    assert f"MISSING (allowed): {REL}" in capsys.readouterr().out
