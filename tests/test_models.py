"""M3 tests: baseline formula, the four model types, guards, window loading, tracking."""

import numpy as np
import pandas as pd
import pytest

from flightdelay.features import FEATURE_COLUMNS, fit_vocabulary, scoring_frame, training_frame
from flightdelay.models import (
    MODEL_TYPES,
    FlightDelayModel,
    HistoricalRateModel,
    check_model_input,
    load_model_config,
)
from flightdelay.training import (
    guard_holdout,
    load_training_window,
    months_between,
    window_vocabulary,
)
from flightdelay.validate import training_row_mask, usable_features_mask
from tests.test_ingest_validate import make_month_df


def labelled_df(n=600, seed=0, year=2019, month=1):
    df = make_month_df(n=n, year=year, month=month)
    rng = np.random.default_rng(seed)
    df["ArrDel15"] = pd.array(rng.binomial(1, 0.2, n).astype("float32"), dtype="float32")
    return df


# --- historical-rate baseline (D-031) ----------------------------------------

def _frame(rows):
    """rows: (airline, origin, dep_hour, y)."""
    X = pd.DataFrame({
        "airline": pd.Categorical([r[0] for r in rows]),
        "origin": pd.Categorical([r[1] for r in rows]),
        "dep_hour": pd.Series([r[2] for r in rows], dtype="int8"),
    })
    return X, pd.Series([r[3] for r in rows], dtype="int8")


def test_baseline_formula_exact_values():
    X, y = _frame([("AA", "JFK", 8, 1)] * 3 + [("AA", "JFK", 8, 0)] + [("DL", "LAX", 9, 0)] * 6)
    m = HistoricalRateModel(prior_strength_airline=100, prior_strength_combination=50).fit(X, y)
    p_g = 3 / 10
    p_aa = (3 + 100 * p_g) / (4 + 100)
    p_dl = (0 + 100 * p_g) / (6 + 100)
    p_combo = (3 + 50 * p_aa) / (4 + 50)
    assert m.p_global_ == pytest.approx(p_g)
    assert m.p_airline_["AA"] == pytest.approx(p_aa)
    assert m.p_airline_["DL"] == pytest.approx(p_dl)
    q, _ = _frame([("AA", "JFK", 8, 0), ("AA", "LAX", 8, 0), ("UA", "JFK", 8, 0)])
    q["airline"] = q["airline"].cat.set_categories(["AA", "DL", "UA"])
    p = m.predict_proba(q)[:, 1]
    assert p[0] == pytest.approx(p_combo)   # seen combination
    assert p[1] == pytest.approx(p_aa)      # unseen combination -> airline rate
    assert p[2] == pytest.approx(p_g)       # unseen airline -> global rate


def test_baseline_missing_categories_fall_back():
    X, y = _frame([("AA", "JFK", 8, 1), ("AA", "JFK", 8, 0), ("DL", "LAX", 9, 0)])
    m = HistoricalRateModel(100, 50).fit(X, y)
    q = pd.DataFrame({"airline": pd.Categorical(["AA", None], categories=["AA", "DL"]),
                      "origin": pd.Categorical([None, "JFK"], categories=["JFK", "LAX"]),
                      "dep_hour": pd.Series([8, 8], dtype="int8")})
    p = m.predict_proba(q)[:, 1]
    assert p[0] == pytest.approx(m.p_airline_["AA"])  # missing origin -> airline rate
    assert p[1] == pytest.approx(m.p_global_)         # missing airline -> global rate


def test_baseline_low_count_is_shrunk_towards_parent():
    X, y = _frame([("AA", "JFK", 8, 1)] + [("AA", "BOS", 9, 0)] * 199)
    m = HistoricalRateModel(100, 50).fit(X, y)
    p_one = m.p_combo_[("AA", "JFK", 8)]
    parent = m.p_airline_["AA"]
    assert parent < p_one < 1.0
    assert p_one == pytest.approx((1 + 50 * parent) / 51)


def test_baseline_prior_strengths_are_fixed_in_config():
    cfg = load_model_config()["historical_rate"]
    assert cfg == {"prior_strength_airline": 100, "prior_strength_combination": 50}


# --- all four model types ------------------------------------------------------

@pytest.mark.parametrize("mt", MODEL_TYPES)
def test_model_fits_and_scores_usable_rows(mt):
    train = labelled_df(seed=1)
    batch = labelled_df(seed=2)
    batch.loc[0, "ArrDel15"] = None          # no label: must still be scored
    batch.loc[0, "Cancelled"] = 1.0
    batch.loc[1, "Origin"] = None            # quarantined: must not be scored
    m = FlightDelayModel(mt).fit_frame(train)
    p = m.predict_proba(batch)
    assert list(p.index) == list(batch.index[usable_features_mask(batch)])
    assert 0 in p.index and 1 not in p.index
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    full = m.predict_proba_all(batch)
    assert len(full) == len(batch) and np.isnan(full.loc[1]) and not np.isnan(full.loc[0])


@pytest.mark.parametrize("mt", MODEL_TYPES)
def test_model_is_deterministic(mt):
    train, batch = labelled_df(seed=3), labelled_df(seed=4)
    a = FlightDelayModel(mt).fit_frame(train).predict_proba(batch)
    b = FlightDelayModel(mt).fit_frame(train).predict_proba(batch)
    pd.testing.assert_series_equal(a, b)


@pytest.mark.parametrize("mt", MODEL_TYPES)
def test_vocabulary_comes_from_training_rows_and_unseen_categories_score(mt):
    train = labelled_df(seed=5)
    m = FlightDelayModel(mt).fit_frame(train)
    assert m.vocab == fit_vocabulary(train)
    batch = labelled_df(seed=6)
    batch["Origin"] = batch["Origin"].cat.add_categories(["XYZ"])
    batch.loc[:9, "Origin"] = "XYZ"
    p = m.predict_proba(batch)
    assert len(p) == len(batch) and np.isfinite(p).all()


def test_categorical_integer_codes_are_rejected():
    df = labelled_df()
    X, y = training_frame(df, fit_vocabulary(df))
    X_codes = X.assign(airline=X["airline"].cat.codes)
    with pytest.raises(TypeError, match="D-026"):
        check_model_input(X_codes)
    for mt in MODEL_TYPES:
        with pytest.raises(TypeError, match="D-026"):
            FlightDelayModel(mt).fit(X_codes, y, fit_vocabulary(df))
    m = FlightDelayModel("xgboost").fit(X, y, fit_vocabulary(df))
    with pytest.raises(TypeError, match="D-026"):
        m.predict_proba_features(X_codes)


def test_unknown_model_type_and_unfitted_model():
    with pytest.raises(ValueError):
        FlightDelayModel("random_forest")
    with pytest.raises(RuntimeError):
        FlightDelayModel("lightgbm").predict_proba(labelled_df())


# --- O-018 guard -----------------------------------------------------------

def test_lightgbm_guard_when_training_has_no_missing_durations():
    train = labelled_df(seed=7)
    assert train["CRSElapsedTime"].notna().all()
    m = FlightDelayModel("lightgbm").fit_frame(train)
    fill = m.metadata["lgbm_duration_fill"]
    assert fill is not None and m.metadata["train_missing_duration"] == 0
    X, _ = training_frame(labelled_df(seed=8), m.vocab)
    nan_pred = m.predict_proba_features(X.assign(crs_elapsed_time=np.nan))
    med_pred = m.predict_proba_features(X.assign(crs_elapsed_time=np.float32(fill)))
    np.testing.assert_allclose(nan_pred, med_pred)


def test_lightgbm_guard_off_when_training_has_missing_durations():
    train = labelled_df(seed=9)
    train.loc[:20, "CRSElapsedTime"] = None
    m = FlightDelayModel("lightgbm").fit_frame(train)
    assert m.metadata["lgbm_duration_fill"] is None
    assert m.metadata["train_missing_duration"] == 21


# --- training window loading (D-010, D-020, D-025) ---------------------------

def _write_months(tmp_path, months):
    for i, m in enumerate(months):
        df = labelled_df(n=300, seed=10 + i, year=int(m[:4]), month=int(m[5:]))
        if i == 1:
            df["Origin"] = df["Origin"].cat.add_categories(["SEA"])
            df.loc[:4, "Origin"] = "SEA"
        df.to_parquet(tmp_path / f"{m[:4]}_{m[5:]}.parquet", index=False)


def test_window_loader_matches_whole_window_processing(tmp_path):
    months = ["2019-01", "2019-02", "2019-03"]
    _write_months(tmp_path, months)
    whole = pd.concat([pd.read_parquet(tmp_path / f"{m[:4]}_{m[5:]}.parquet") for m in months],
                      ignore_index=True)
    assert window_vocabulary(months, tmp_path) == fit_vocabulary(whole)
    X, y, vocab, info = load_training_window(months, tmp_path)
    assert vocab == fit_vocabulary(whole)
    assert len(X) == int(training_row_mask(whole).sum()) == len(y)
    assert list(X.columns) == FEATURE_COLUMNS
    assert info["train_from"] == "2019-01" and info["train_to"] == "2019-03"
    assert info["train_months"] == 3 and info["window_raw_rows"] == len(whole)
    check_model_input(X)


def test_months_between_and_holdout_guard():
    assert months_between("2019-11", "2020-02") == ["2019-11", "2019-12", "2020-01", "2020-02"]
    with pytest.raises(ValueError):
        months_between("2020-02", "2019-11")
    guard_holdout(["2024-12"])
    with pytest.raises(PermissionError, match="D-020"):
        guard_holdout(["2024-12", "2025-01"])
    guard_holdout(["2025-01"], allow_holdout=True)


def test_run_train_refuses_holdout_and_non_future_eval():
    from flightdelay.run_train import main
    with pytest.raises(PermissionError):
        main(["--model", "lightgbm", "--train-from", "2024-01", "--train-to", "2025-01", "--no-mlflow"])
    with pytest.raises(PermissionError):
        main(["--model", "lightgbm", "--train-from", "2024-01", "--train-to", "2024-06",
              "--eval-month", "2025-02", "--no-mlflow"])
    with pytest.raises(ValueError, match="D-007"):
        main(["--model", "lightgbm", "--train-from", "2024-01", "--train-to", "2024-06",
              "--eval-month", "2024-06", "--no-mlflow"])


# --- tracking: logged model reloads and gives identical predictions ---------------

# MLflow's SQLite store triggers a SQLAlchemy deprecation warning inside the
# library (``noload`` loader strategy). It is third-party and does not affect us.
@pytest.mark.filterwarnings("ignore:The ``noload`` loader strategy is deprecated")
def test_mlflow_round_trip(tmp_path):
    import mlflow

    from flightdelay.tracking import log_run, setup_mlflow
    train, batch = labelled_df(seed=11), labelled_df(seed=12)
    batch.loc[3, "Distance"] = None           # one unusable row -> NaN in output
    model = FlightDelayModel("lightgbm").fit_frame(train, {"train_from": "2019-01"})
    setup_mlflow("test_exp", tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
                 artifact_root=tmp_path / "artifacts")
    run_id = log_run(model, "test_run", {"fit_seconds": 1.0})
    run = mlflow.get_run(run_id)
    assert run.data.tags["model_type"] == "lightgbm"
    assert "git_commit" in run.data.tags and "data_manifest_sha256" in run.data.tags
    assert run.data.params["hp_n_estimators"] == "300"
    loaded = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    out = loaded.predict(batch)
    expected = model.predict_proba_all(batch)
    np.testing.assert_allclose(out["delay_probability"].to_numpy(), expected.to_numpy(), equal_nan=True)
    assert np.isnan(out["delay_probability"].iloc[3])


def test_scoring_frame_used_for_prediction_matches_feature_contract():
    train = labelled_df(seed=13)
    m = FlightDelayModel("xgboost").fit_frame(train)
    X = scoring_frame(train, m.vocab)
    np.testing.assert_allclose(m.predict_proba(train).to_numpy(), m.predict_proba_features(X))



def test_git_dirty_includes_untracked_code_files(tmp_path):
    """I-22: a run made with new, uncommitted code must not be recorded as clean."""
    import subprocess

    from flightdelay.tracking import git_info

    def git(*a):
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *a],
                       cwd=tmp_path, check=True, capture_output=True)
    git("init", "-q")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    git("add", ".")
    git("commit", "-q", "-m", "init")
    assert git_info(tmp_path)["git_dirty"] is False
    (tmp_path / "src" / "new_module.py").write_text("y = 2\n")      # untracked code
    info = git_info(tmp_path)
    assert info["git_dirty"] is True and info["git_changed_files"] == 1
    (tmp_path / "notes.txt").write_text("not code")                   # outside CODE_PATHS
    assert git_info(tmp_path)["git_changed_files"] == 1
