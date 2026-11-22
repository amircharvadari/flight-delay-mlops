"""E3 replay engine on synthetic months (M7)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from flightdelay.config import load_config
from flightdelay.policy import ModelRef, check_time_order
from flightdelay.replay import Replay, compare, load_monitoring_config
from flightdelay.training import months_between
from tests.test_models import labelled_df

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")
MONTHS = months_between("2020-01", "2020-06")
INITIAL = ("2019-01", "2019-12")


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    d = tmp_path_factory.mktemp("processed")
    for i, m in enumerate(months_between("2019-01", "2020-06")):
        df = labelled_df(n=400, seed=100 + i, year=int(m[:4]), month=int(m[5:]))
        df.loc[:2, "Distance"] = None                    # unscorable rows (O-016)
        df.loc[3:5, "ArrDel15"] = np.nan                 # unlabelled rows (D-013)
        df.to_parquet(d / f"{m[:4]}_{m[5:]}.parquet", index=False)
    return d


def _replay(processed, out, mcfg=None):
    return Replay(processed, out, "lightgbm", 12, mcfg or load_monitoring_config(), log=lambda *_: None)


def _always_trigger():
    m = load_monitoring_config()
    m["trigger"]["delta"] = -1.0                         # test only: AUC < baseline + 1 always
    return m


def test_three_policies_run_and_keep_time_order(processed, tmp_path):
    r = _replay(processed, tmp_path)
    res = {p: r.run_policy(p, MONTHS, INITIAL) for p in ("never_retrained", "triggered", "scheduled")}
    for p, out in res.items():
        assert len(out["monthly"]) == len(MONTHS)
        check_time_order(out["events"], load_config())
        assert (out["monthly"]["serving_train_to"] < out["monthly"]["month"]).all()
        assert (tmp_path / "policies" / p / "summary.json").exists()
    assert res["never_retrained"]["summary"]["n_trained"] == 0
    assert (res["never_retrained"]["monthly"]["serving_model"] == "lightgbm_2019-01_2019-12").all()
    assert res["scheduled"]["summary"]["n_trained"] >= 2
    # same months, same rows: the initial model's AP in 2020-01 is identical across policies
    ap = {p: o["monthly"].iloc[0]["average_precision"] for p, o in res.items()}
    assert len(set(ap.values())) == 1
    c = compare(res)
    assert set(c) == {"triggered_minus_never_retrained", "scheduled_minus_never_retrained",
                      "scheduled_minus_triggered"}


def test_candidates_train_on_data_up_to_the_decision_month(processed, tmp_path):
    r = _replay(processed, tmp_path, _always_trigger())
    out = r.run_policy("triggered", MONTHS, INITIAL)
    trains = [e for e in out["events"] if e["event"] == "train_candidate"]
    assert trains and all(e["train_to"] == e["month"] for e in trains)
    for e in trains:
        meta = json.loads((tmp_path / "models" / f"lightgbm_{e['train_from']}_{e['train_to']}.json").read_text())
        assert meta["train_to"] == e["month"]
    monthly = out["monthly"]
    assert monthly["shadow_candidate_ap"].notna().any()


def test_resume_reuses_caches_and_reproduces_decisions(processed, tmp_path):
    first = _replay(processed, tmp_path).run_policy("scheduled", MONTHS, INITIAL)
    again = _replay(processed, tmp_path)
    second = again.run_policy("scheduled", MONTHS, INITIAL)
    assert again.fits_this_run == 0
    pd.testing.assert_frame_equal(first["monthly"], second["monthly"])
    assert first["events"] == second["events"]


def test_signals_and_frozen_threshold(processed, tmp_path):
    out = _replay(processed, tmp_path).run_policy("never_retrained", MONTHS, INITIAL)
    row = out["monthly"].iloc[0]
    assert row["unscorable_share"] == pytest.approx(3 / 400)
    assert row["labelled_share_of_scored"] == pytest.approx(394 / 397)
    assert row["calibration_month"] == "2020-01"         # month after the training window (D-038)
    assert 0 < row["frozen_alert_rate"] < 1
    assert out["monthly"]["frozen_tau"].nunique() == 1    # frozen: never recomputed
    assert out["monthly"]["prevalence_last_year"].notna().all()


def test_holdout_and_training_months_are_refused(processed, tmp_path):
    r = _replay(processed, tmp_path)
    with pytest.raises(ValueError, match="holdout"):
        r.run_policy("never_retrained", ["2024-12", "2025-01"], INITIAL)
    with pytest.raises(AssertionError, match="inside its training data"):
        r.score(ModelRef("x", "2019-01", "2019-12"), "2019-12")


def test_cli_refuses_identity_change(processed, tmp_path, monkeypatch):
    from flightdelay import run_replay
    args = ["--out-dir", str(tmp_path), "--processed-dir", str(processed), "--first-month",
            "2020-01", "--last-month", "2020-02", "--allow-dirty", "--no-mlflow",
            "--policies", "never_retrained"]
    assert run_replay.main(args) == 0
    assert (tmp_path / "comparison.json").exists()
    changed = [*args[:-1], "scheduled"]
    assert run_replay.main(changed) == 3                  # different plan -> refused
