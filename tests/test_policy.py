"""Retraining, promotion, rollback and time order (D-012, D-040 to D-044)."""
from __future__ import annotations

import numpy as np
import pytest

from flightdelay.config import load_config
from flightdelay.grid import add_months
from flightdelay.policy import (
    ModelRef,
    Observation,
    PolicyState,
    check_time_order,
    models_to_score,
    step,
)
from flightdelay.training import months_between

CFG = load_config()
MCFG = {"trigger": {"delta": 0.03}, "promotion": {"margin": 0.005}, "rollback": {"margin": 0.005}}


def _init(baseline=0.65):
    return PolicyState(serving=ModelRef("initial", "2019-01", "2019-12", baseline_auc=baseline))


def _step(state, obs, policy="triggered"):
    return step(state, obs, policy, MCFG, base_cfg=CFG)


def test_initial_model_gets_its_baseline_in_its_first_month_without_trigger():
    s = PolicyState(serving=ModelRef("initial", "2019-01", "2019-12"))
    s, actions = _step(s, Observation("2020-01", 0.60, 0.30), "scheduled")
    assert s.serving.baseline_auc == 0.60 and actions == []


@pytest.mark.parametrize("auc,fires", [(0.65, False), (0.621, False), (0.62, False), (0.619, True)])
def test_trigger_threshold(auc, fires):
    """Fires only if AUC < baseline - 0.03 (0.62 itself does not fire)."""
    _, actions = _step(_init(0.65), Observation("2020-05", auc, 0.3))
    assert bool(actions) is fires


def test_never_retrained_never_trains():
    s = _init()
    for m in months_between("2020-01", "2024-12"):
        s, actions = _step(s, Observation(m, 0.40, 0.1), "never_retrained")
        assert actions == []
    assert s.serving.model_id == "initial"


def test_candidate_window_and_shadow_month():
    _, actions = _step(_init(0.65), Observation("2021-06", 0.60, 0.3))
    assert actions == [{"type": "train_candidate", "model_id": "triggered_2021-06",
                        "train_from": "2020-07", "train_to": "2021-06", "shadow_month": "2021-07"}]


def test_promotion_serves_from_t_plus_2_and_previous_is_watched():
    s, _ = _step(_init(0.65), Observation("2021-06", 0.60, 0.30))
    assert set(models_to_score(s, "2021-07")) == {"serving", "candidate"}
    s, actions = _step(s, Observation("2021-07", 0.60, 0.300, candidate_auc=0.66,
                                      candidate_ap=0.305))       # exactly + margin -> promote
    assert actions == []                                         # no trigger in a change month
    assert s.serving.model_id == "triggered_2021-06" and s.serving.baseline_auc == 0.66
    assert s.previous.model_id == "initial" and s.watch_month == "2021-08"
    assert set(models_to_score(s, "2021-08")) == {"serving", "previous"}
    promote = next(e for e in s.events if e["event"] == "promote")
    assert promote["serves_from"] == "2021-08" == add_months("2021-06", 2)


def test_discard_keeps_the_model_and_allows_a_new_trigger():
    s, _ = _step(_init(0.65), Observation("2021-06", 0.60, 0.30))
    s, actions = _step(s, Observation("2021-07", 0.60, 0.300, candidate_auc=0.66,
                                      candidate_ap=0.3049))
    assert s.serving.model_id == "initial"
    assert any(e["event"] == "discard" for e in s.events)
    assert actions and actions[0]["train_to"] == "2021-07"     # still degraded -> new candidate


def _promoted():
    s, _ = _step(_init(0.65), Observation("2021-06", 0.60, 0.30))
    s, _ = _step(s, Observation("2021-07", 0.60, 0.30, candidate_auc=0.66, candidate_ap=0.32))
    return s


def test_rollback_when_previous_is_better_by_the_margin():
    s, _ = _step(_promoted(), Observation("2021-08", 0.62, 0.300, previous_ap=0.305))
    assert s.serving.model_id == "initial" and s.serving.baseline_auc == 0.65
    assert s.previous.model_id == "triggered_2021-06" and s.watch_month is None
    rb = next(e for e in s.events if e["event"] == "rollback")
    assert rb["serves_from"] == "2021-09"


def test_no_rollback_below_margin_and_watch_ends():
    s, _ = _step(_promoted(), Observation("2021-08", 0.66, 0.300, previous_ap=0.3049))
    assert s.serving.model_id == "triggered_2021-06" and s.watch_month is None
    assert "previous" not in models_to_score(s, "2021-09")


def test_missing_shadow_scores_are_an_error():
    s, _ = _step(_init(0.65), Observation("2021-06", 0.60, 0.30))
    with pytest.raises(ValueError, match="shadow"):
        _step(s, Observation("2021-07", 0.60, 0.30))
    with pytest.raises(ValueError, match="watch"):
        _step(_promoted(), Observation("2021-08", 0.66, 0.30))


def test_holdout_is_locked():
    with pytest.raises(ValueError, match="holdout"):
        _step(_init(), Observation("2025-01", 0.6, 0.3))
    s, actions = _step(_init(0.65), Observation("2024-12", 0.50, 0.2), "scheduled")
    assert actions == [] and s.events[-1]["event"] == "trigger_skipped_end_of_replay"


@pytest.mark.parametrize("policy", ["never_retrained", "triggered", "scheduled"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_full_replay_simulation_respects_time_order(policy, seed):
    """60 months with random metrics: every model serves only after its data."""
    rng = np.random.default_rng(seed)
    s = PolicyState(serving=ModelRef("initial", "2019-01", "2019-12"))
    trained = {"initial": "2019-12"}
    for m in months_between("2020-01", "2024-12"):
        scored = models_to_score(s, m)
        for role, ref in scored.items():
            assert ref.train_to < m, (role, ref, m)          # no model scores its own data
        obs = Observation(m, rng.uniform(0.55, 0.7), rng.uniform(0.2, 0.4),
                          candidate_auc=rng.uniform(0.55, 0.7) if "candidate" in scored else None,
                          candidate_ap=rng.uniform(0.2, 0.4) if "candidate" in scored else None,
                          previous_ap=rng.uniform(0.2, 0.4) if "previous" in scored else None)
        s, actions = _step(s, obs, policy)
        for a in actions:
            assert a["train_to"] == m and a["shadow_month"] == add_months(m, 1)
            trained[a["model_id"]] = a["train_to"]
    check_time_order(s.events, CFG)
    if policy == "scheduled":
        assert sum(e["event"] == "train_candidate" for e in s.events) >= 25


@pytest.mark.parametrize("bad", [
    [{"month": "2021-06", "event": "train_candidate", "candidate": "c", "train_to": "2021-07",
      "shadow_month": "2021-08", "serving": "m"}],
    [{"month": "2021-06", "event": "train_candidate", "candidate": "c", "train_to": "2021-06",
      "shadow_month": "2021-08", "serving": "m"}],
    [{"month": "2021-06", "event": "train_candidate", "candidate": "c", "train_to": "2021-06",
      "shadow_month": "2021-07", "serving": "m"},
     {"month": "2021-07", "event": "promote", "candidate": "c", "serves_from": "2021-07",
      "serving": "m"}],
    [{"month": "2025-02", "event": "baseline_set", "serving": "m"}],
])
def test_time_order_checker_catches_violations(bad):
    with pytest.raises(AssertionError):
        check_time_order(bad, CFG)
