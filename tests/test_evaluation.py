"""D-034 threshold and batch evaluation."""

import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from flightdelay.evaluation import alert_threshold, evaluate_batch
from flightdelay.models import FlightDelayModel
from flightdelay.validate import training_row_mask, usable_features_mask
from tests.test_models import labelled_df


def test_threshold_is_kth_largest_score():
    t = alert_threshold(np.array([0.5, 0.9, 0.7, 0.6, 0.8]), 0.4)
    assert t["k"] == 2 and t["tau"] == 0.8 and t["scored_rows"] == 5


def test_threshold_k_is_at_least_one_and_rounds():
    assert alert_threshold(np.array([0.1, 0.2, 0.3]), 0.0001)["k"] == 1
    assert alert_threshold(np.linspace(0, 1, 10), 0.25)["k"] == 2   # round(2.5) -> 2 (banker's rounding)
    assert alert_threshold(np.linspace(0, 1, 10), 0.35)["k"] == 4   # round(3.5) -> 4


def test_threshold_rejects_bad_input():
    with pytest.raises(ValueError):
        alert_threshold(np.array([]), 0.2)
    with pytest.raises(ValueError):
        alert_threshold(np.array([0.1]), 1.5)


def test_threshold_signature_has_no_labels():
    assert list(inspect.signature(alert_threshold).parameters) == ["scores", "pi_train"]


def _model_and_batch():
    model = FlightDelayModel("logreg").fit_frame(labelled_df(seed=21))
    batch = labelled_df(seed=22)
    batch.loc[0:9, "ArrDel15"] = None          # 10 cancelled flights: scored, unlabelled
    batch.loc[0:9, "Cancelled"] = 1.0
    batch.loc[10, "Origin"] = None             # quarantined: not scored
    return model, batch


def test_counts_and_alert_rates():
    model, batch = _model_and_batch()
    m, p = evaluate_batch(model, batch)
    assert m["scored_rows"] == int(usable_features_mask(batch).sum()) == len(p) == 599
    assert m["labelled_rows"] == int(training_row_mask(batch).sum()) == int(p["labelled"].sum()) == 589
    assert m["flagged_rows"] == int(p["flagged"].sum())
    assert m["flagged_labelled_rows"] == int((p["flagged"] & p["labelled"]).sum())
    assert m["realised_alert_rate_scored"] == pytest.approx(m["flagged_rows"] / 599)
    assert m["realised_alert_rate_labelled"] == pytest.approx(m["flagged_labelled_rows"] / 589)
    assert m["target_alert_rate"] == model.metadata["train_positive_rate"]
    assert m["k"] == round(m["pi_train"] * 599)


def test_metrics_match_saved_predictions():
    model, batch = _model_and_batch()
    m, p = evaluate_batch(model, batch)
    lab = p[p["labelled"]]
    y = lab["label"].astype(int).to_numpy()
    f = lab["flagged"].to_numpy()
    tp = int((f & (y == 1)).sum())
    assert m["precision"] == pytest.approx(tp / f.sum())
    assert m["recall"] == pytest.approx(tp / (y == 1).sum())
    assert m["average_precision"] == average_precision_score(y, lab["score"].to_numpy())
    assert p.loc[~p["labelled"], "label"].isna().all()
    assert list(p["row_id"]) == list(batch.index[usable_features_mask(batch)])


def test_threshold_and_flags_do_not_depend_on_evaluation_labels():
    model, batch = _model_and_batch()
    m1, p1 = evaluate_batch(model, batch)
    flipped = batch.copy()
    lab = flipped["ArrDel15"].notna()
    flipped.loc[lab, "ArrDel15"] = 1.0 - flipped.loc[lab, "ArrDel15"]
    m2, p2 = evaluate_batch(model, flipped)
    assert m1["tau"] == m2["tau"] and m1["k"] == m2["k"]
    pd.testing.assert_series_equal(p1["flagged"], p2["flagged"])
    assert m1["precision"] != m2["precision"]     # labels only affect the metrics


def test_ties_can_flag_more_than_k_and_are_reported():
    t = alert_threshold(np.array([0.5, 0.5, 0.5, 0.5, 0.1]), 0.2)
    flagged = np.array([0.5, 0.5, 0.5, 0.5, 0.1]) >= t["tau"]
    assert t["k"] == 1 and flagged.sum() == 4
