"""E4 final evaluation gates and the holdout lock (D-020, D-045)."""
from __future__ import annotations

import json

import pytest

from flightdelay import run_replay
from flightdelay.config import load_config
from flightdelay.policy import ModelRef, Observation, PolicyState, check_time_order, step
from flightdelay.replay import Replay, load_monitoring_config
from flightdelay.training import months_between
from tests.test_models import labelled_df

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")
CFG = load_config()
MCFG = {"trigger": {"delta": 0.03}, "promotion": {"margin": 0.005}, "rollback": {"margin": 0.005}}


def test_policy_holdout_is_locked_by_default_and_unlocked_only_explicitly():
    s = PolicyState(serving=ModelRef("m", "2024-01", "2024-12", baseline_auc=0.65))
    with pytest.raises(ValueError, match="holdout"):
        step(s, Observation("2025-03", 0.5, 0.2), "scheduled", MCFG, base_cfg=CFG)
    s2, actions = step(s, Observation("2025-03", 0.5, 0.2), "scheduled", MCFG, base_cfg=CFG,
                       unlock_holdout=True, last_month="2025-12")
    assert actions[0]["train_to"] == "2025-03" and actions[0]["shadow_month"] == "2025-04"
    check_time_order(s2.events, CFG, unlock_holdout=True)
    with pytest.raises(AssertionError, match="holdout"):
        check_time_order(s2.events, CFG)


def test_last_month_ends_the_run():
    s = PolicyState(serving=ModelRef("m", "2024-01", "2024-12", baseline_auc=0.65))
    s2, actions = step(s, Observation("2025-12", 0.5, 0.2), "scheduled", MCFG, base_cfg=CFG,
                       unlock_holdout=True, last_month="2025-12")
    assert actions == [] and s2.events[-1]["event"] == "trigger_skipped_end_of_replay"
    with pytest.raises(ValueError, match="after the last month"):
        step(s, Observation("2026-01", 0.5, 0.2), "scheduled", MCFG, base_cfg=CFG,
             unlock_holdout=True, last_month="2025-12")


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    d = tmp_path_factory.mktemp("processed_e4")
    for i, m in enumerate(months_between("2024-01", "2025-12")):
        labelled_df(n=300, seed=500 + i, year=int(m[:4]), month=int(m[5:])).to_parquet(
            d / f"{m[:4]}_{m[5:]}.parquet", index=False)
    return d


def test_replay_engine_refuses_2025_unless_unlocked(processed, tmp_path):
    locked = Replay(processed, tmp_path / "a", log=lambda *_: None)
    with pytest.raises(ValueError, match="holdout"):
        locked.run_policy("never_retrained", ["2025-01"], ("2024-01", "2024-12"))
    unlocked = Replay(processed, tmp_path / "b", log=lambda *_: None, unlock_holdout=True)
    out = unlocked.run_policy("never_retrained", ["2025-01", "2025-02"], ("2024-01", "2024-12"))
    assert list(out["monthly"]["month"]) == ["2025-01", "2025-02"]


def test_standard_cli_still_refuses_2025(processed, tmp_path):
    with pytest.raises(ValueError, match="holdout"):
        run_replay.main(["--out-dir", str(tmp_path), "--processed-dir", str(processed),
                         "--first-month", "2024-12", "--last-month", "2025-01", "--allow-dirty",
                         "--no-mlflow", "--policies", "never_retrained"])


@pytest.fixture
def gate(monkeypatch):
    """Pretend the tree is clean at commit 'abc' and the tag points to `state['tag']`."""
    state = {"tag": "abc", "dirty": False}
    real_identity = run_replay.identity

    def fake_identity(*a, **k):
        return {**real_identity(*a, **k), "git_commit": "abc", "git_dirty": state["dirty"]}

    monkeypatch.setattr(run_replay, "identity", fake_identity)
    monkeypatch.setattr(run_replay, "freeze_commit", lambda tag: state["tag"])
    return state


def _e4(processed, out):
    return run_replay.main(["--e4", "--out-dir", str(out), "--processed-dir", str(processed),
                            "--no-mlflow"])


@pytest.mark.parametrize("tag,dirty", [(None, False), ("other", False), ("abc", True)])
def test_e4_refused_without_matching_freeze_or_with_dirty_tree(processed, tmp_path, gate, tag, dirty):
    gate.update(tag=tag, dirty=dirty)
    assert _e4(processed, tmp_path) == 4
    assert not (tmp_path / "identity.json").exists()


def test_e4_overrides_are_refused(processed, tmp_path, gate):
    assert run_replay.main(["--e4", "--out-dir", str(tmp_path), "--first-month", "2025-01",
                            "--no-mlflow"]) == 2


def test_e4_runs_once_with_the_preregistered_plan(processed, tmp_path, gate):
    assert _e4(processed, tmp_path) == 0
    comp = json.loads((tmp_path / "comparison.json").read_text())
    assert comp["experiment"] == "E4" and set(comp["hypotheses"]) == {"H1", "H2"}
    assert set(comp["summaries"]) == {"never_retrained", "triggered", "scheduled"}
    for s in comp["summaries"].values():
        assert (s["first_month"], s["last_month"]) == ("2025-01", "2025-12")
    starts = (tmp_path / "e4_starts.jsonl").read_text().splitlines()
    assert len(starts) == 1 and json.loads(starts[0])["commit"] == "abc"
    ident = json.loads((tmp_path / "identity.json").read_text())
    assert ident["freeze_tag"] == "e4-freeze" and "e4_config_sha256" in ident
    assert _e4(processed, tmp_path) == 4                     # completed: never again
    assert len((tmp_path / "e4_starts.jsonl").read_text().splitlines()) == 1


def test_e4_config_matches_d045():
    import yaml

    from flightdelay.policy import POLICIES
    e4 = yaml.safe_load(run_replay.E4_CONFIG.read_text())
    assert e4["months"] == ["2025-01", "2025-12"] and e4["initial_train"] == ["2024-01", "2024-12"]
    assert e4["policies"] == list(POLICIES) and e4["freeze_tag"] == "e4-freeze"
    assert load_monitoring_config()["trigger"]["delta"] == 0.03   # thresholds unchanged
