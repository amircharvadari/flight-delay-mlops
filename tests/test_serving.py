"""M5 serving: bundle, registry round trip, API contract (D-018, D-025, D-037, D-038, D-039)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from flightdelay.api import MAX_FLIGHTS, create_app, to_batch
from flightdelay.models import FlightDelayModel
from flightdelay.serving import frozen_threshold, load_bundle, write_bundle
from tests.test_models import labelled_df

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning", "ignore::FutureWarning",
                                        "ignore::UserWarning")


@pytest.fixture(scope="module")
def model():
    return FlightDelayModel("lightgbm").fit_frame(
        labelled_df(n=800, seed=21), {"train_from": "2019-01", "train_to": "2019-01",
                                      "train_months": 1})


@pytest.fixture(scope="module")
def bundle_dir(model, tmp_path_factory):
    batch = labelled_df(n=400, seed=22, month=2)
    scores = model.predict_proba(batch).to_numpy()
    thr = frozen_threshold(model, scores, "2019-02")
    return write_bundle(model, tmp_path_factory.mktemp("serving") / "champion", thr,
                        {"model_version": "7", "champion": {"decision": "D-037"}})


@pytest.fixture(scope="module")
def client(bundle_dir):
    with TestClient(create_app(bundle_dir)) as c:
        yield c


def _vocab_flight(model, **over):
    v = model.vocab
    f = {"flight_id": "f1", "flight_date": "2019-03-04", "reporting_airline": v["airline"][0],
         "origin": v["origin"][0], "dest": v["dest"][-1], "crs_dep_time": 830,
         "crs_arr_time": 1045, "crs_elapsed_time": 135.0, "distance": 700.0}
    return {**f, **over}


# --- bundle ---------------------------------------------------------------------------

def test_bundle_round_trip_gives_identical_scores(model, bundle_dir):
    b = load_bundle(bundle_dir)
    batch = labelled_df(n=300, seed=23, month=3)
    np.testing.assert_array_equal(b.model.predict_proba(batch).to_numpy(),
                                  model.predict_proba(batch).to_numpy())
    assert b.meta["model_type"] == "lightgbm" and b.meta["model_version"] == "7"
    assert b.threshold == b.meta["threshold"]["tau"]


def test_bundle_is_readable_by_any_user(bundle_dir):
    """I-24: the bundle is mounted into a container that runs as a non-root user."""
    paths = [bundle_dir, *bundle_dir.rglob("*")]
    assert (bundle_dir / "serving.json") in paths
    for p in paths:
        mode = p.stat().st_mode & 0o777
        assert mode == (0o755 if p.is_dir() else 0o644), (p, oct(mode))


def test_frozen_threshold_is_the_d034_rule(model):
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    thr = frozen_threshold(model, scores, "2019-02")
    k = max(1, round(model.metadata["train_positive_rate"] * len(scores)))
    assert thr["k"] == k and thr["tau"] == np.sort(scores)[::-1][k - 1]
    assert thr["calibration_month"] == "2019-02" and thr["rule"].startswith("D-038")
    assert thr["labels_used"] is False
    assert (thr["train_from"], thr["train_to"]) == ("2019-01", "2019-01")


def test_incomplete_or_tampered_bundle_is_refused(bundle_dir, tmp_path):
    import shutil
    broken = tmp_path / "broken"
    shutil.copytree(bundle_dir, broken)
    (broken / "serving.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_bundle(broken)
    tampered = tmp_path / "tampered"
    shutil.copytree(bundle_dir, tampered)
    meta = json.loads((tampered / "serving.json").read_text())
    meta["vocabulary_sha256"] = "0" * 64
    (tampered / "serving.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="vocabulary"):
        load_bundle(tampered)


def test_app_does_not_start_without_a_bundle(tmp_path):
    with pytest.raises(FileNotFoundError), TestClient(create_app(tmp_path / "none")):
        pass


def test_registry_alias_round_trip(model, tmp_path):
    import mlflow

    from flightdelay.run_export import download_alias, register_champion
    from flightdelay.tracking import log_run, setup_mlflow
    setup_mlflow("test_serving", tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
                 artifact_root=tmp_path / "artifacts")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("month,zip_sha256\n2019-01,abc\n")
    run_id = log_run(model, "champion_test", {"fit_seconds": 1.0}, manifest_path=manifest)
    v1 = register_champion(run_id, "flightdelay_test", "champion", {"decision": "D-037"})
    run_id2 = log_run(model, "champion_test_2", {"fit_seconds": 1.0}, manifest_path=manifest)
    v2 = register_champion(run_id2, "flightdelay_test", "champion", {"decision": "D-037"})
    client = mlflow.MlflowClient()
    assert (v1, v2) == ("1", "2")
    assert str(client.get_model_version_by_alias("flightdelay_test", "champion").version) == "2"
    assert client.get_model_version("flightdelay_test", "2").tags["decision"] == "D-037"
    src = download_alias("flightdelay_test", "champion", tmp_path / "dl")
    out = write_bundle(model, tmp_path / "b", frozen_threshold(model, np.array([0.1, 0.9]), "x"),
                       {"model_version": v2}, model_src=src)
    batch = labelled_df(n=200, seed=24)
    np.testing.assert_array_equal(load_bundle(out).model.predict_proba(batch).to_numpy(),
                                  model.predict_proba(batch).to_numpy())


# --- API ------------------------------------------------------------------------------

def test_health_and_model_metadata(client):
    h = client.get("/health").json()
    assert h == {"status": "ok", "model_type": "lightgbm", "model_version": "7"}
    m = client.get("/model").json()
    assert m["threshold"]["rule"].startswith("D-038") and "params" not in m
    assert m["champion"]["decision"] == "D-037"


def test_predict_matches_batch_scoring_no_training_serving_skew(client, model):
    """API scores equal the model's scores on the same rows in processed format."""
    batch = labelled_df(n=50, seed=25, month=4)
    flights = [{"flight_id": str(i), "flight_date": str(r.FlightDate.date()),
                "reporting_airline": str(r.Reporting_Airline), "origin": str(r.Origin),
                "dest": str(r.Dest), "crs_dep_time": int(r.CRSDepTime),
                "crs_arr_time": int(r.CRSArrTime),
                "crs_elapsed_time": None if pd.isna(r.CRSElapsedTime) else float(r.CRSElapsedTime),
                "distance": float(r.Distance)}
               for i, r in enumerate(batch.itertuples())]
    res = client.post("/predict", json={"flights": flights})
    assert res.status_code == 200
    body = res.json()
    api = np.array([p["delay_probability"] for p in body["predictions"]])
    np.testing.assert_allclose(api, model.predict_proba(batch).to_numpy(), rtol=0, atol=1e-12)
    for p in body["predictions"]:
        assert p["delay_alert"] == (p["delay_probability"] >= body["threshold"])


def test_unseen_category_is_scored_and_reported(client, model):
    res = client.post("/predict", json={"flights": [_vocab_flight(model, origin="ZZZ")]})
    assert res.status_code == 200
    p = res.json()["predictions"][0]
    assert p["unknown_categories"] == ["origin"] and 0.0 <= p["delay_probability"] <= 1.0


@pytest.mark.parametrize("duration,used", [(135.0, True), (None, False), (0.0, False),
                                           (5000.0, False)])
def test_missing_or_implausible_duration_is_reported(client, model, duration, used):
    res = client.post("/predict", json={"flights": [_vocab_flight(model, crs_elapsed_time=duration)]})
    assert res.status_code == 200
    assert res.json()["predictions"][0]["crs_elapsed_time_used"] is used


@pytest.mark.parametrize("field", ["ArrDel15", "DepDelay", "arr_delay", "cancelled"])
def test_outcome_or_unknown_fields_are_rejected(client, model, field):
    """D-003: no post-departure or outcome information can enter a request."""
    res = client.post("/predict", json={"flights": [_vocab_flight(model, **{field: 1})]})
    assert res.status_code == 422


@pytest.mark.parametrize("over", [
    {"crs_dep_time": 2401}, {"crs_dep_time": -1}, {"crs_arr_time": 1275},
    {"distance": 0}, {"distance": 6001}, {"origin": "atl"}, {"reporting_airline": "DLX"},
    {"flight_date": "2024-02-30"}, {"crs_dep_time": None},
])
def test_invalid_values_are_rejected(client, model, over):
    assert client.post("/predict", json={"flights": [_vocab_flight(model, **over)]}).status_code == 422


def test_batch_size_limits(client, model):
    assert client.post("/predict", json={"flights": []}).status_code == 422
    too_many = [_vocab_flight(model)] * (MAX_FLIGHTS + 1)
    assert client.post("/predict", json={"flights": too_many}).status_code == 422


def test_request_rows_are_usable_rows_of_the_validation_contract(model):
    """Every request the API accepts is a usable row under validate.usable_features_mask."""
    from flightdelay.api import Flight
    from flightdelay.validate import usable_features_mask
    edge = [Flight(**_vocab_flight(model, crs_dep_time=0, crs_arr_time=2400, distance=1)),
            Flight(**_vocab_flight(model, crs_dep_time=2359, distance=6000))]
    df = to_batch(edge).assign(Flight_Number_Reporting_Airline=pd.array([1, 2], dtype="Int32"))
    assert bool(usable_features_mask(df).all())
