"""M4 grid: enumeration, crash-safe store, resume, identity check, selection rule."""

import json
import os
from dataclasses import asdict

import pandas as pd
import pytest

import flightdelay.grid as G
from flightdelay.grid import (
    FitSpec,
    IdentityMismatch,
    atomic_write_bytes,
    check_or_write_identity,
    collect_results,
    decision_points,
    enumerate_fits,
    is_done,
    load_grid_config,
    run_grid,
    subprocess_executor,
)
from flightdelay.selection import select
from tests.test_models import labelled_df

CFG = load_grid_config()


# --- enumeration ------------------------------------------------------------------

def test_decision_points_and_fit_count():
    pts = decision_points(CFG)
    assert pts == ["2022-01", "2022-04", "2022-07", "2022-10", "2023-01", "2023-04",
                   "2023-07", "2023-10", "2024-01", "2024-04", "2024-07", "2024-10"]
    fits = enumerate_fits(CFG)
    assert len(fits) == 192 and len({f.key for f in fits}) == 192


def test_windows_and_test_months():
    f = {(x.window, x.model): x for x in enumerate_fits(CFG, ["2022-01"], models=["lightgbm"])}
    assert f[("12", "lightgbm")].train_from == "2021-02"
    assert f[("24", "lightgbm")].train_from == "2020-02"
    assert f[("36", "lightgbm")].train_from == "2019-02"
    assert f[("expanding", "lightgbm")].train_from == "2019-01"
    assert all(x.train_to == "2022-01" and x.test_month == "2022-02" for x in f.values())


def test_grid_never_touches_the_holdout():
    fits = enumerate_fits(CFG)
    assert max(f.test_month for f in fits) == "2024-11"
    with pytest.raises(PermissionError, match="D-020"):
        enumerate_fits(CFG, ["2024-12"])


# --- crash-safe writes, identity ----------------------------------------------------

def test_atomic_write_leaves_old_file_on_failure(tmp_path, monkeypatch):
    target = tmp_path / "r.json"
    atomic_write_bytes(target, b"old")
    monkeypatch.setattr(G.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("crash")))
    with pytest.raises(OSError):
        atomic_write_bytes(target, b"new")
    assert target.read_bytes() == b"old"
    assert [p.name for p in tmp_path.iterdir()] == ["r.json"]   # no temp file left


def test_identity_mismatch_refuses_to_mix(tmp_path):
    ident = {"git_commit": "a", "code_sha256": "x", "config_sha256": {"g": "1"}}
    check_or_write_identity(tmp_path, ident)
    check_or_write_identity(tmp_path, ident)                       # same: fine
    for change in ({"git_commit": "b"}, {"code_sha256": "y"}, {"config_sha256": {"g": "2"}}):
        with pytest.raises(IdentityMismatch):
            check_or_write_identity(tmp_path, {**ident, **change})


# --- resume with an injected executor ---------------------------------------------------

def _fake_executor(fail_keys=()):
    calls = []

    def ex(fit, out_dir, opts):
        calls.append(fit.key)
        if fit.key in fail_keys:
            return {"returncode": 1, "error": "boom"}
        pred = G.prediction_path(out_dir, fit.key)
        G.atomic_write_parquet(pred, pd.DataFrame({"row_id": [0, 1], "score": [0.1, 0.9]}))
        G.atomic_write_json(G.fit_json_path(out_dir, fit.key), {
            **asdict(fit), "key": fit.key, "status": "ok", "average_precision": 0.3,
            "fit_seconds": 1.0, "peak_rss_mb": 10.0, "predictions_sha256": G.sha256_file(pred)})
        return {"returncode": 0}
    return ex, calls


def test_resume_runs_only_unfinished_fits(tmp_path):
    fits = enumerate_fits(CFG, ["2022-01"], ["12"])
    ident = {"git_commit": "a"}
    ex, calls = _fake_executor(fail_keys={fits[1].key})
    s = run_grid(fits, tmp_path, ident, {}, executor=ex, log=lambda *_: None)
    assert s == {"total": 4, "done": 3, "started_now": 4, "pending": 1} and len(calls) == 4
    assert json.loads(G.fit_json_path(tmp_path, fits[1].key).read_text())["status"] == "failed"
    ex2, calls2 = _fake_executor()
    s = run_grid(fits, tmp_path, ident, {}, executor=ex2, log=lambda *_: None)
    assert calls2 == [fits[1].key] and s["done"] == 4


def test_partial_or_corrupt_fits_are_rerun(tmp_path):
    fits = enumerate_fits(CFG, ["2022-01"], ["12"], ["historical_rate", "logreg"])
    ident = {"git_commit": "a"}
    ex, _ = _fake_executor()
    run_grid(fits, tmp_path, ident, {}, executor=ex, log=lambda *_: None)
    G.fit_json_path(tmp_path, fits[0].key).write_text("{ truncated")         # crash mid-json
    G.prediction_path(tmp_path, fits[1].key).write_bytes(b"other content")   # hash mismatch
    assert not is_done(tmp_path, fits[0].key) and not is_done(tmp_path, fits[1].key)
    ex2, calls2 = _fake_executor()
    run_grid(fits, tmp_path, ident, {}, executor=ex2, log=lambda *_: None)
    assert sorted(calls2) == sorted(f.key for f in fits)


def test_max_fits_limits_one_session(tmp_path):
    fits = enumerate_fits(CFG, ["2022-01"], ["12"])
    ex, calls = _fake_executor()
    s = run_grid(fits, tmp_path, {"git_commit": "a"}, {}, executor=ex, max_fits=2, log=lambda *_: None)
    assert len(calls) == 2 and s["pending"] == 2


# --- real subprocess path on small synthetic data (no dependency on local data) -----------

@pytest.mark.filterwarnings("ignore:The ``noload`` loader strategy is deprecated")
def test_subprocess_fit_end_to_end_and_paired_predictions(tmp_path):
    proc_dir = tmp_path / "processed"
    proc_dir.mkdir()
    for i, m in enumerate(["2021-12", "2022-01", "2022-02"]):
        labelled_df(n=400, seed=30 + i, year=int(m[:4]), month=int(m[5:])).to_parquet(
            proc_dir / f"{m[:4]}_{m[5:]}.parquet", index=False)
    fits = [FitSpec("2022-01", "2", mdl, "2021-12", "2022-01", "2022-02")
            for mdl in ("historical_rate", "logreg")]
    ident = {"git_commit": "test", "code_sha256": "test"}
    opts = {"experiment": "t", "mlflow": False, "fit_timeout_seconds": 120,
            "identity": ident, "processed_dir": str(proc_dir)}
    out = tmp_path / "grid"
    s = run_grid(fits, out, ident, opts, executor=subprocess_executor, log=lambda *_: None)
    assert s["done"] == 2
    res = collect_results(out)
    assert set(res["status"]) == {"ok"} and (out / "results.csv").exists()
    r = res.iloc[0]
    for col in ("scored_rows", "flagged_rows", "labelled_rows", "flagged_labelled_rows",
                "realised_alert_rate_scored", "realised_alert_rate_labelled", "target_alert_rate",
                "tau", "fit_seconds", "peak_rss_mb", "test_parquet_sha256"):
        assert col in res.columns and pd.notna(r[col])
    # paired analysis: same rows, same labels, same input file for both models
    a = pd.read_parquet(G.prediction_path(out, fits[0].key))
    b = pd.read_parquet(G.prediction_path(out, fits[1].key))
    assert a["row_id"].equals(b["row_id"]) and a["label"].equals(b["label"])
    assert res["test_parquet_sha256"].nunique() == 1
    # resume after completion starts nothing
    assert run_grid(fits, out, ident, opts, executor=subprocess_executor,
                    log=lambda *_: None)["started_now"] == 0


def test_subprocess_timeout_is_recorded(tmp_path, monkeypatch):
    fit = FitSpec("2022-01", "2", "logreg", "2021-12", "2022-01", "2022-02")
    import subprocess as sp

    def slow(*a, **k):
        raise sp.TimeoutExpired(cmd="x", timeout=1)
    monkeypatch.setattr(G.subprocess, "run", slow)
    s = run_grid([fit], tmp_path, {"git_commit": "a"}, {"fit_timeout_seconds": 1},
                 log=lambda *_: None)
    r = json.loads(G.fit_json_path(tmp_path, fit.key).read_text())
    assert s["done"] == 0 and r["status"] == "timeout"


# --- selection rule (D-033, D-035) -------------------------------------------------------

def _results(spec, points=("p1", "p2")):
    rows = []
    for (model, window), (ap, t, mem) in spec.items():
        for p in points:
            rows.append({"status": "ok", "model": model, "window": window, "decision_point": p,
                         "average_precision": ap, "fit_seconds": t, "peak_rss_mb": mem,
                         "roc_auc": 0.6, "precision": 0.3, "recall": 0.3, "f1": 0.3})
    return pd.DataFrame(rows)


def _cfg(windows=("12", "36")):
    return {**CFG, "windows": list(windows)}


REFS_LOW = {("historical_rate", "12"): (0.20, 1, 900), ("historical_rate", "36"): (0.21, 2, 1500),
            ("logreg", "12"): (0.22, 20, 2000), ("logreg", "36"): (0.23, 40, 3000)}


def test_best_wins_when_no_cheaper_option_is_close():
    r = _results({("lightgbm", "12"): (0.30, 50, 1000), ("lightgbm", "36"): (0.32, 130, 3000),
                  ("xgboost", "12"): (0.29, 60, 1300), ("xgboost", "36"): (0.31, 180, 3300), **REFS_LOW})
    s = select(r, _cfg(), ["p1", "p2"])
    assert (s["selected_model"], s["selected_window"]) == ("lightgbm", "36")


def test_cheaper_option_within_margin_wins():
    r = _results({("lightgbm", "12"): (0.316, 50, 1000), ("lightgbm", "36"): (0.320, 130, 3000),
                  ("xgboost", "12"): (0.29, 60, 1300), ("xgboost", "36"): (0.31, 180, 3300), **REFS_LOW})
    s = select(r, _cfg(), ["p1", "p2"])
    assert (s["selected_model"], s["selected_window"]) == ("lightgbm", "12")


def test_time_tie_then_memory_decides():
    r = _results({("lightgbm", "12"): (0.320, 100, 3000), ("lightgbm", "36"): (0.318, 105, 1000),
                  ("xgboost", "12"): (0.29, 60, 1300), ("xgboost", "36"): (0.20, 180, 3300), **REFS_LOW})
    s = select(r, _cfg(), ["p1", "p2"])
    assert (s["selected_model"], s["selected_window"]) == ("lightgbm", "36")


def test_time_and_memory_tie_then_ap_decides():
    r = _results({("lightgbm", "12"): (0.318, 100, 1000), ("lightgbm", "36"): (0.320, 105, 1050),
                  ("xgboost", "12"): (0.29, 60, 1300), ("xgboost", "36"): (0.20, 180, 3300), **REFS_LOW})
    s = select(r, _cfg(), ["p1", "p2"])
    assert (s["selected_model"], s["selected_window"]) == ("lightgbm", "36")


def test_reference_models_are_never_selected():
    r = _results({("lightgbm", "12"): (0.30, 50, 1000), ("lightgbm", "36"): (0.30, 50, 1000),
                  ("xgboost", "12"): (0.30, 50, 1000), ("xgboost", "36"): (0.30, 50, 1000),
                  **REFS_LOW, ("historical_rate", "12"): (0.90, 1, 100)})
    s = select(r, _cfg(), ["p1", "p2"])
    assert s["selected_model"] in ("lightgbm", "xgboost")
    assert s["acceptance"]["outcome"] == "candidate_below_reference"


def test_incomplete_grid_is_refused_or_marked_not_a_decision():
    r = _results({("lightgbm", "12"): (0.30, 50, 1000)})
    with pytest.raises(ValueError, match="incomplete"):
        select(r, _cfg(), ["p1", "p2"])
    s = select(r, _cfg(), ["p1", "p2"], allow_incomplete=True)
    assert s["is_decision"] is False


def test_selection_tolerances_are_fixed_in_config():
    assert CFG["selection"] == {"metric": "average_precision", "ap_margin": 0.005,
                                "time_tie_tolerance": 0.10, "memory_tie_tolerance": 0.10}
    assert os.path.basename(str(G.GRID_CONFIG)) == "grid.yaml"



# --- baseline acceptance check (D-036) ------------------------------------------------

CANDS = {("lightgbm", "12"): (0.300, 50, 1000), ("lightgbm", "36"): (0.290, 130, 3000),
         ("xgboost", "12"): (0.280, 60, 1300), ("xgboost", "36"): (0.270, 180, 3300)}


def _refs(best_ref_ap):
    return {("historical_rate", "12"): (0.20, 1, 900), ("historical_rate", "36"): (0.21, 2, 1500),
            ("logreg", "12"): (best_ref_ap, 20, 2000), ("logreg", "36"): (0.25, 40, 3000)}


@pytest.mark.parametrize("ref_ap,outcome,superior", [
    (0.310, "candidate_below_reference", False),      # reference higher
    (0.297, "improvement_not_demonstrated", False),   # candidate higher by 0.003 < 0.005
    (0.295, "candidate_better", True),                # candidate higher by exactly 0.005
    (0.260, "candidate_better", True),
])
def test_acceptance_outcomes(ref_ap, outcome, superior):
    s = select(_results({**CANDS, **_refs(ref_ap)}), _cfg(), ["p1", "p2"])
    a = s["acceptance"]
    assert (s["selected_model"], s["selected_window"]) == ("lightgbm", "12")   # selection unchanged
    assert a["outcome"] == outcome and a["candidate_may_be_presented_as_superior"] is superior
    assert (a["best_reference_model"], a["best_reference_window"]) == ("logreg", "12")
    assert a["candidate_minus_reference_ap"] == pytest.approx(0.300 - ref_ap)


def test_acceptance_check_does_not_change_the_candidate_selection():
    a = select(_results({**CANDS, **_refs(0.40)}), _cfg(), ["p1", "p2"])
    b = select(_results({**CANDS, **_refs(0.10)}), _cfg(), ["p1", "p2"])
    assert (a["selected_model"], a["selected_window"]) == (b["selected_model"], b["selected_window"])
    assert [r for r in a["table"]] == [r for r in b["table"]]


def test_missing_reference_results_make_the_grid_incomplete():
    r = _results({**CANDS, ("logreg", "12"): (0.25, 20, 2000)})
    with pytest.raises(ValueError, match="reference"):
        select(r, _cfg(), ["p1", "p2"])
    s = select(r, _cfg(), ["p1", "p2"], allow_incomplete=True)
    assert s["is_decision"] is False and s["acceptance"]["references_complete"] is False



def test_no_reference_results_cannot_be_presented_as_superior():
    s = select(_results(CANDS), _cfg(), ["p1", "p2"], allow_incomplete=True)
    a = s["acceptance"]
    assert a["outcome"] == "references_missing" and a["candidate_may_be_presented_as_superior"] is False
    assert s["is_decision"] is False
