"""Registry aliases, promotion, rollback and bundle rebuild (D-038, D-039, D-042)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from flightdelay import registry as reg
from flightdelay.models import FlightDelayModel
from flightdelay.serving import frozen_threshold, load_bundle
from flightdelay.tracking import log_run, setup_mlflow
from tests.test_models import labelled_df

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning", "ignore::FutureWarning",
                                        "ignore::UserWarning")
NAME = "flightdelay_test"


@pytest.fixture
def versions(tmp_path):
    setup_mlflow("test_registry", tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
                 artifact_root=tmp_path / "artifacts")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("month,zip_sha256\n2019-01,abc\n")
    out = []
    for seed, month in ((31, "2019-02"), (32, "2019-03")):
        m = FlightDelayModel("lightgbm").fit_frame(
            labelled_df(n=600, seed=seed), {"train_from": "2019-01", "train_to": "2019-01",
                                            "train_months": 1})
        thr = frozen_threshold(m, m.predict_proba(labelled_df(n=300, seed=seed + 10)).to_numpy(),
                               month)
        run_id = log_run(m, f"v{seed}", {"fit_seconds": 1.0}, manifest_path=manifest)
        v = reg.register_version(run_id, NAME, {**reg.threshold_tags(thr), "decision": "D-037"})
        out.append((v, m, thr))
    return out


def test_promote_rollback_and_rebuild_bundle(versions, tmp_path):
    (v1, m1, t1), (v2, m2, t2) = versions
    assert reg.promote(NAME, v1) == {"champion": v1}
    assert reg.promote(NAME, v2) == {"champion": v2, "previous": v1}
    b = load_bundle(reg.export_bundle(NAME, tmp_path / "bundle", event="promote"))
    assert b.meta["model_version"] == v2 and b.threshold == t2["tau"]
    batch = labelled_df(n=200, seed=40)
    np.testing.assert_array_equal(b.model.predict_proba(batch).to_numpy(),
                                  m2.predict_proba(batch).to_numpy())
    assert reg.rollback(NAME) == {"champion": v1, "previous": v2}
    b = load_bundle(reg.export_bundle(NAME, tmp_path / "bundle", event="rollback"))
    assert b.meta["model_version"] == v1 and b.threshold == t1["tau"]
    assert b.meta["threshold"]["calibration_month"] == "2019-02"
    assert b.meta["bundle_event"] == "rollback"
    np.testing.assert_array_equal(b.model.predict_proba(batch).to_numpy(),
                                  m1.predict_proba(batch).to_numpy())


def test_rollback_needs_a_previous_version(versions):
    reg.promote(NAME, versions[0][0])
    with pytest.raises(RuntimeError, match="previous"):
        reg.rollback(NAME)


def test_threshold_tags_round_trip_and_missing_tags():
    thr = {"tau": 0.19, "k": 5, "scored_rows": 20, "pi_train": 0.2, "calibration_month": "2024-11",
           "train_from": "2023-11", "train_to": "2024-10", "labels_used": False, "rule": "D-038"}
    assert reg.threshold_from_tags(reg.threshold_tags(thr)) == thr
    with pytest.raises(KeyError):
        reg.threshold_from_tags({"threshold_tau": json.dumps(0.1)})


def test_tag_from_bundle_backfills_threshold(versions, tmp_path):
    from flightdelay.serving import write_bundle
    v1, m1, t1 = versions[0]
    b = write_bundle(m1, tmp_path / "b", t1, {"registered_model": NAME, "model_version": v1})
    from mlflow import MlflowClient
    client = MlflowClient()
    for k in reg.THRESHOLD_TAG_KEYS:
        client.delete_model_version_tag(NAME, v1, f"threshold_{k}")
    assert reg._tag_from_bundle(NAME, b) == v1
    assert reg.threshold_from_tags(client.get_model_version(NAME, v1).tags) == t1
