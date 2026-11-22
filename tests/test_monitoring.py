"""Monitoring signals (D-008, D-043) and the reason for the AUC trigger (D-040)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from flightdelay import monitoring as mon
from flightdelay.config import load_config
from flightdelay.policy import ModelRef, Observation, PolicyState, step

MCFG = {"trigger": {"delta": 0.03}, "promotion": {"margin": 0.005},
        "rollback": {"margin": 0.005},
        "feature_drift": {"numeric_bins": 10, "psi_floor": 1e-4, "warning": 0.10, "alert": 0.25,
                          "features": ["distance", "origin"]},
        "prevalence": {"alert_abs_diff": 0.05}}


def test_psi_is_zero_for_identical_shares_and_matches_hand_value():
    assert mon.psi_from_shares(np.array([0.5, 0.5]), np.array([0.5, 0.5]), 1e-4) == 0.0
    expected = (0.7 - 0.5) * np.log(0.7 / 0.5) + (0.3 - 0.5) * np.log(0.3 / 0.5)
    assert mon.psi_from_shares(np.array([0.5, 0.5]), np.array([0.7, 0.3]), 1e-4) == pytest.approx(expected)
    # empty bins are floored, so PSI stays finite
    assert np.isfinite(mon.psi_from_shares(np.array([1.0, 0.0]), np.array([0.0, 1.0]), 1e-4))


def _frame(rng, n, shift=0.0, origin_p=(0.5, 0.3, 0.2)):
    return pd.DataFrame({
        "distance": pd.Series(rng.normal(1000 + shift, 300, n), dtype="float32"),
        "origin": pd.Categorical(rng.choice(["ATL", "JFK", "LAX"], n, p=origin_p),
                                 categories=["ATL", "JFK", "LAX"]),
    })


def test_numeric_and_categorical_drift_levels():
    rng = np.random.default_rng(0)
    vocab = {"origin": ["ATL", "JFK", "LAX"]}
    ref = _frame(rng, 50_000)
    same = mon.feature_drift_signal(ref, _frame(rng, 50_000), vocab, MCFG)
    assert same["max_psi"] < 0.01 and same["n_alert"] == 0
    shifted = mon.feature_drift_signal(ref, _frame(rng, 50_000, shift=300), vocab, MCFG)
    assert shifted["levels"]["distance"] == "alert" and shifted["max_psi_feature"] == "distance"
    mix = mon.feature_drift_signal(ref, _frame(rng, 50_000, origin_p=(0.2, 0.3, 0.5)), vocab, MCFG)
    assert mix["levels"]["origin"] == "alert" and mix["levels"]["distance"] == "ok"


def test_missing_values_and_unseen_categories_have_their_own_bin():
    edges = mon.numeric_bin_edges(pd.Series([1.0, 2.0, 3.0, 4.0]), 2)
    assert mon.numeric_counts(pd.Series([1.0, np.nan, 4.0]), edges).tolist() == [1.0, 1.0, 1.0]
    cats = pd.Categorical(["ATL", None, "JFK"], categories=["ATL", "JFK"])
    assert mon.categorical_counts(pd.Series(cats), ["ATL", "JFK"]).tolist() == [1.0, 1.0, 1.0]


def test_reference_month_is_the_same_calendar_month_in_the_window():
    window = [f"2019-{m:02d}" for m in range(1, 13)]
    assert mon.reference_month(window, "2023-05") == "2019-05"
    with pytest.raises(ValueError):
        mon.reference_month(window[:6], "2020-09")


def test_prevalence_volume_and_concept_signals():
    assert mon.prevalence_signal(0.25, 0.19, MCFG)["alert"] is True
    assert mon.prevalence_signal(0.22, 0.19, MCFG)["alert"] is False
    assert mon.prevalence_signal(0.22, None, MCFG)["alert"] is False
    v = mon.volume_signal(rows=1000, usable_rows=990, labelled_rows=970, rows_last_year=2000)
    assert v["unscorable_rows"] == 10 and v["volume_ratio_vs_last_year"] == 0.5
    assert v["labelled_share_of_scored"] == pytest.approx(970 / 990)
    assert mon.concept_drift_inference(False, None) == "no_degradation"
    assert mon.concept_drift_inference(True, {"n_alert": 0}) == "suspected_concept_drift"
    assert mon.concept_drift_inference(True, {"n_alert": 2}) == "degradation_with_feature_drift"


def test_unseen_category_share():
    df = pd.DataFrame({"Reporting_Airline": ["AA", "ZZ"], "Origin": ["ATL", "ATL"],
                       "Dest": ["JFK", "JFK"]})
    vocab = {"airline": ["AA"], "origin": ["ATL"], "dest": ["JFK"]}
    assert mon.unseen_category_share(df, vocab) == 0.5


def test_prevalence_change_alone_moves_ap_but_does_not_trigger_retraining():
    """D-008 / D-040: same class-conditional score distributions, lower prevalence.
    AP falls with prevalence, AUC does not, so the AUC trigger does not fire."""
    rng = np.random.default_rng(1)

    def batch(prev, n=200_000):
        y = rng.random(n) < prev
        s = np.where(y, rng.beta(3, 2, n), rng.beta(2, 3, n))
        return y, s

    y1, s1 = batch(0.25)
    y2, s2 = batch(0.12)
    ap1, ap2 = average_precision_score(y1, s1), average_precision_score(y2, s2)
    auc1, auc2 = roc_auc_score(y1, s1), roc_auc_score(y2, s2)
    assert ap1 - ap2 > 0.10                    # AP drops with prevalence
    assert abs(auc1 - auc2) < 0.005            # AUC unchanged
    state = PolicyState(serving=ModelRef("m0", "2019-01", "2019-12", baseline_auc=auc1))
    state, actions = step(state, Observation("2020-03", auc2, ap2), "triggered", MCFG,
                          base_cfg=load_config())
    assert actions == [] and not any(e["event"] == "degradation" for e in state.events)
