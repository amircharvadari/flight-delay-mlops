"""Tests for scripts/compare_validation_reports.py (evidence tooling must be trustworthy too)."""

import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_validation_reports.py"
spec = importlib.util.spec_from_file_location("compare_validation_reports", SCRIPT)
cmp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmp)


def write(tmp_path, name, rows):
    p = tmp_path / name
    pd.DataFrame(rows).to_csv(p, index=False)
    return str(p)


BASE = [
    {"month": "2019-01", "status": "pass", "rows": 10, "excluded_pct": 1.5, "warnings": None},
    {"month": "2019-02", "status": "warn", "rows": 20, "excluded_pct": 2.0, "warnings": "1 rows quarantined (out-of-range)"},
]


def test_identical_reports_have_no_differences(tmp_path):
    b = write(tmp_path, "b.csv", BASE)
    a = write(tmp_path, "a.csv", BASE)
    assert cmp.main([b, a, "--strict"]) == 0


def test_numeric_and_status_differences_are_detected(tmp_path):
    changed = [dict(r) for r in BASE]
    changed[0]["rows"] = 11
    changed[1]["status"] = "fail"
    b = write(tmp_path, "b.csv", BASE)
    a = write(tmp_path, "a.csv", changed)
    r = cmp.compare(cmp.load(b), cmp.load(a))
    assert r["differences"]["rows"] == ["2019-01"]
    assert r["differences"]["status"] == ["2019-02"]
    assert cmp.main([b, a, "--strict"]) == 1


def test_wording_normalisation_and_new_columns(tmp_path):
    changed = [dict(r) for r in BASE]
    changed[1]["warnings"] = "1 rows quarantined (missing or out-of-range)"
    for r in changed:
        r["training_rows"] = r["rows"]
    b = write(tmp_path, "b.csv", BASE)
    a = write(tmp_path, "a.csv", changed)
    raw = cmp.compare(cmp.load(b), cmp.load(a))
    assert raw["differences"]["warnings"] == ["2019-02"]
    assert raw["columns_only_after"] == ["training_rows"]
    norm = cmp.compare(cmp.load(b), cmp.load(a),
                       replacements=[("missing or out-of-range", "out-of-range")])
    assert "warnings" not in norm["differences"]
