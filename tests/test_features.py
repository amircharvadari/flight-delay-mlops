"""M2 feature tests: allowlist, leakage, determinism, dtypes, missing values, masks."""

import json

import numpy as np
import pandas as pd
import pytest

import flightdelay.features as F
from flightdelay.config import load_config
from flightdelay.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    FEATURE_DTYPES,
    FEATURE_SOURCE_COLUMNS,
    FORBIDDEN_COLUMNS,
    MAX_BLOCK_SPEED_MPH,
    build_features,
    fit_vocabulary,
    implausible_duration_mask,
    scoring_frame,
    training_frame,
)
from flightdelay.validate import training_row_mask, usable_features_mask
from tests.test_ingest_validate import make_month_df

POST_FLIGHT_EXTRAS = {
    "DepTime": 905, "DepDelay": 5.0, "DepDel15": 0.0, "TaxiOut": 12.0, "WheelsOff": 917,
    "WheelsOn": 1105, "TaxiIn": 6.0, "ArrTime": 1111, "ArrDelay": 11.0, "AirTime": 108.0,
    "ActualElapsedTime": 126.0, "CarrierDelay": 0.0, "WeatherDelay": 0.0, "NASDelay": 0.0,
    "SecurityDelay": 0.0, "LateAircraftDelay": 0.0, "CancellationCode": "B",
}


@pytest.fixture
def df():
    return make_month_df()


@pytest.fixture
def vocab(df):
    return fit_vocabulary(df)


# --- allowlist and leakage -------------------------------------------------

def test_allowlist_contains_no_forbidden_column():
    assert not set(FEATURE_SOURCE_COLUMNS) & FORBIDDEN_COLUMNS


def test_allowlist_is_within_the_configured_policy():
    allowed = set(load_config()["allowed_feature_columns"])
    assert set(FEATURE_SOURCE_COLUMNS) <= allowed
    assert load_config()["target"] in FORBIDDEN_COLUMNS


def test_output_has_exactly_the_declared_features(df, vocab):
    X = build_features(df, vocab)
    assert list(X.columns) == FEATURE_COLUMNS
    assert not set(X.columns) & FORBIDDEN_COLUMNS
    assert "ArrDel15" not in X.columns and "y" not in X.columns


def test_forbidden_on_allowlist_fails_loudly(df, vocab, monkeypatch):
    monkeypatch.setattr(F, "FEATURE_SOURCE_COLUMNS", FEATURE_SOURCE_COLUMNS + ["DepDelay"])
    with pytest.raises(ValueError, match="forbidden"):
        build_features(df, vocab)


def test_outcome_columns_cannot_influence_features(df, vocab):
    """Leakage test: perfectly predictive outcome columns must change nothing."""
    base = build_features(df, vocab)
    leaky = df.copy()
    for col, val in POST_FLIGHT_EXTRAS.items():
        leaky[col] = val
    leaky["ArrDelay"] = np.where(df["ArrDel15"] == 1, 60.0, -5.0)   # encodes the label
    leaky["Cancelled"] = 1.0 - df["ArrDel15"]                       # also encodes it
    leaky["ArrDel15"] = 1.0 - df["ArrDel15"]                        # flip the target itself
    leaky["Diverted"] = df["ArrDel15"]
    out = build_features(leaky, vocab)
    pd.testing.assert_frame_equal(out, base)


def test_features_work_without_any_outcome_column(df, vocab):
    base = build_features(df, vocab)
    only_allowed = df[FEATURE_SOURCE_COLUMNS]
    pd.testing.assert_frame_equal(build_features(only_allowed, vocab), base)


# --- unexpected and missing columns, dtypes ---------------------------------

def test_unexpected_extra_columns_are_ignored(df, vocab):
    extra = df.assign(Tail_Number="N123", SomethingNew=42)
    pd.testing.assert_frame_equal(build_features(extra, vocab), build_features(df, vocab))


@pytest.mark.parametrize("col", FEATURE_SOURCE_COLUMNS)
def test_missing_source_column_raises(df, vocab, col):
    with pytest.raises(KeyError, match=col):
        build_features(df.drop(columns=[col]), vocab)


def test_output_dtypes_are_fixed(df, vocab):
    X = build_features(df, vocab)
    assert {c: str(t) for c, t in X.dtypes.items()} == FEATURE_DTYPES


def test_non_numeric_time_column_raises(df, vocab):
    bad = df.assign(CRSDepTime=df["CRSDepTime"].astype(str))
    with pytest.raises(TypeError, match="CRSDepTime"):
        build_features(bad, vocab)


def test_string_dates_raise(df, vocab):
    bad = df.assign(FlightDate=df["FlightDate"].dt.strftime("%Y-%m-%d"))
    with pytest.raises(TypeError, match="FlightDate"):
        build_features(bad, vocab)


def test_categorical_input_as_object_gives_same_result(df, vocab):
    obj = df.assign(Origin=df["Origin"].astype(str), Dest=df["Dest"].astype(str))
    pd.testing.assert_frame_equal(build_features(obj, vocab), build_features(df, vocab))


# --- values ----------------------------------------------------------------

def test_time_and_calendar_values():
    df = make_month_df(n=4)
    df["FlightDate"] = pd.to_datetime(["2019-01-07", "2019-01-08", "2019-01-12", "2019-01-31"])
    df["CRSDepTime"] = pd.array([0, 1259, 2359, 2400], dtype="Int16")
    df["CRSArrTime"] = pd.array([5, 1300, 15, 100], dtype="Int16")
    X = build_features(df, fit_vocabulary(df))
    assert list(X["dep_hour"]) == [0, 12, 23, 0]
    assert list(X["dep_minute_of_day"]) == [0, 779, 1439, 0]
    assert list(X["arr_hour"]) == [0, 13, 0, 1]
    assert list(X["day_of_week"]) == [0, 1, 5, 3]          # 2019-01-07 was a Monday
    assert list(X["day_of_month"]) == [7, 8, 12, 31]
    assert set(X["month"]) == {1}


# --- missing values --------------------------------------------------------

def test_missing_crs_elapsed_time_is_kept_as_nan(df, vocab):
    df.loc[0, "CRSElapsedTime"] = None
    X = build_features(df, vocab)
    assert np.isnan(X.loc[0, "crs_elapsed_time"])
    assert X.drop(columns="crs_elapsed_time").notna().all().all()


def test_unselected_quarantined_row_raises(df, vocab):
    df.loc[0, "Origin"] = None
    with pytest.raises(ValueError, match="usable_features_mask"):
        build_features(df, vocab)


def test_unseen_category_becomes_missing_not_new_code(df, vocab):
    df["Origin"] = df["Origin"].cat.add_categories(["XYZ"])
    df.loc[0, "Origin"] = "XYZ"
    X = build_features(df, vocab)
    assert pd.isna(X.loc[0, "origin"])
    assert list(X["origin"].cat.categories) == vocab["origin"]


# --- determinism -----------------------------------------------------------

def test_deterministic_and_row_independent(df, vocab):
    a = build_features(df, vocab)
    b = build_features(df, vocab)
    pd.testing.assert_frame_equal(a, b)
    shuffled = df.sample(frac=1.0, random_state=0)
    pd.testing.assert_frame_equal(build_features(shuffled, vocab).loc[df.index], a)
    subset = df.iloc[::7]
    pd.testing.assert_frame_equal(build_features(subset, vocab), a.loc[subset.index])


def test_category_codes_do_not_depend_on_batch_content(df, vocab):
    only_lax = df[df["Origin"] == "LAX"]
    codes_full = build_features(df, vocab).loc[only_lax.index, "origin"].cat.codes
    codes_part = build_features(only_lax, vocab)["origin"].cat.codes
    pd.testing.assert_series_equal(codes_part, codes_full)


def test_vocabulary_is_sorted_json_serialisable_and_training_only(df):
    df["Origin"] = df["Origin"].cat.add_categories(["ZZZ"])
    df.loc[0, "Origin"] = "ZZZ"           # appears only on a row with no label
    df.loc[0, "ArrDel15"] = None
    df.loc[0, "Cancelled"] = 1.0
    v = fit_vocabulary(df)
    assert set(v) == set(CATEGORICAL_FEATURES)
    assert all(lst == sorted(lst) for lst in v.values())
    assert "ZZZ" not in v["origin"]
    assert json.loads(json.dumps(v)) == v


# --- masks wired correctly (D-024) ------------------------------------------

def test_scoring_frame_keeps_rows_without_labels(df, vocab):
    df.loc[0, "ArrDel15"] = None
    df.loc[0, "Cancelled"] = 1.0
    df.loc[1, "ArrDel15"] = 2.0           # invalid label, features fine
    X = scoring_frame(df, vocab)
    assert len(X) == 200
    assert {0, 1} <= set(X.index)


def test_scoring_frame_drops_only_quarantined_rows(df, vocab):
    df.loc[0, "Distance"] = None
    df.loc[1, "ArrDel15"] = None
    X = scoring_frame(df, vocab)
    assert list(X.index) == list(df.index[usable_features_mask(df)])
    assert 0 not in X.index and 1 in X.index


def test_training_frame_uses_training_mask_and_valid_labels(df, vocab):
    df.loc[0, "Distance"] = None          # quarantined
    df.loc[1, "ArrDel15"] = None          # no label
    df.loc[1, "Cancelled"] = 1.0
    df.loc[2, "ArrDel15"] = 2.0           # invalid label
    X, y = training_frame(df, vocab)
    assert list(X.index) == list(df.index[training_row_mask(df)])
    assert list(y.index) == list(X.index)
    assert not {0, 1, 2} & set(X.index)
    assert set(y.unique()) <= {0, 1} and str(y.dtype) == "int8"
    assert len(X) == 197



def test_scheduled_duration_comes_from_crs_elapsed_time_not_clock_times():
    """D-027: CRS times are LOCAL clock times, so their difference is wrong for flights
    that cross time zones or midnight. The duration feature must be the BTS
    scheduled elapsed time, copied unchanged."""
    df = make_month_df(n=3)
    # westbound JFK->LAX: clock difference 3h15, scheduled duration 6h15
    # eastbound overnight: departs 23:30, arrives 05:30 next day, 6h00 (clock wraps)
    # westbound, arrival clock earlier than departure clock on the same day: 55 min
    df["CRSDepTime"] = pd.array([800, 2330, 700], dtype="Int16")
    df["CRSArrTime"] = pd.array([1115, 530, 655], dtype="Int16")
    df["CRSElapsedTime"] = pd.array([375.0, 360.0, 55.0], dtype="float32")
    X = build_features(df, fit_vocabulary(df))
    assert list(X["crs_elapsed_time"]) == [375.0, 360.0, 55.0]
    assert "ActualElapsedTime" in FORBIDDEN_COLUMNS and "AirTime" in FORBIDDEN_COLUMNS



# ---------------------------------------------------------------------------
# D-028: implausible scheduled durations -> missing value, row kept
# ---------------------------------------------------------------------------

def _routes(pairs):
    """pairs: list of (distance_miles, scheduled_minutes)."""
    df = make_month_df(n=len(pairs))
    df["Distance"] = pd.array([d for d, _ in pairs], dtype="float32")
    df["CRSElapsedTime"] = pd.array([e for _, e in pairs], dtype="float32")
    return df


@pytest.mark.parametrize("distance,minutes,case", [
    (699.0, 1.0, "too short: 699 mi in 1 min (observed in the data)"),
    (1476.0, 5.0, "too short: 1476 mi in 5 min (observed)"),
    (184.0, 727.0, "too long: LGA-BOS 184 mi in 727 min (observed)"),
    (284.0, 948.0, "too long: MSP-GFK 284 mi in 948 min (observed)"),
    (1990.0, 1312.0, "too long: JFK-SLC 1990 mi in 1312 min (observed)"),
])
def test_implausible_duration_is_set_to_missing(distance, minutes, case):
    df = _routes([(distance, minutes)])
    assert bool(implausible_duration_mask(df).iloc[0]), case
    X = build_features(df, fit_vocabulary(df))
    assert np.isnan(X["crs_elapsed_time"].iloc[0]), case


@pytest.mark.parametrize("distance,minutes,case", [
    (2475.0, 375.0, "JFK-LAX westbound, normal"),
    (2475.0, 328.0, "LAX-JFK eastbound, normal"),
    (31.0, 40.0, "very short hop: low block speed but plausible"),
    (4502.0, 620.0, "long Hawaii route"),
    (500.0, 360.0, "slow but plausible (long schedule padding)"),
])
def test_plausible_duration_is_kept(distance, minutes, case):
    df = _routes([(distance, minutes)])
    assert not bool(implausible_duration_mask(df).iloc[0]), case
    X = build_features(df, fit_vocabulary(df))
    assert X["crs_elapsed_time"].iloc[0] == minutes, case


def test_duration_thresholds_are_inclusive_boundaries():
    # exactly at the speed limit: plausible; just above: implausible
    at_speed = MAX_BLOCK_SPEED_MPH * 60.0 / 60.0          # distance covered in 60 min
    # exactly at the long limit for 300 mi: 180 + 300/150*60 = 300 min
    df = _routes([(at_speed, 60.0), (at_speed + 1.0, 60.0), (300.0, 300.0), (300.0, 301.0)])
    assert list(implausible_duration_mask(df)) == [False, True, False, True]


def test_implausible_duration_keeps_row_and_other_features(df, vocab):
    base = build_features(df, vocab)
    df.loc[0, "CRSElapsedTime"] = 1.0                       # implausible for 500 mi
    X = build_features(df, vocab)
    assert len(X) == len(base) and np.isnan(X.loc[0, "crs_elapsed_time"])
    pd.testing.assert_frame_equal(X.drop(columns="crs_elapsed_time"),
                                  base.drop(columns="crs_elapsed_time"))
    assert len(scoring_frame(df, vocab)) == 200
    assert len(training_frame(df, vocab)[0]) == 200


def test_implausible_duration_does_not_change_validation(df):
    from flightdelay.validate import validate_month
    df.loc[0, "CRSElapsedTime"] = 1.0
    v = validate_month(df, "2019-01")
    assert v.status == "pass" and v.quarantined_rows == 0


def test_missing_duration_is_not_flagged(df):
    df.loc[0, "CRSElapsedTime"] = None
    assert not bool(implausible_duration_mask(df).iloc[0])



@pytest.mark.parametrize("bad", [-77.0, -1.0, 0.0, 0.5, 1601.0, 2000.0])
def test_impossible_duration_becomes_missing_and_row_is_scored(df, vocab, bad):
    """D-030: values that used to quarantine the row now only blank the feature."""
    df.loc[0, "CRSElapsedTime"] = bad
    assert bool(implausible_duration_mask(df).iloc[0])
    Xs = scoring_frame(df, vocab)
    X, _ = training_frame(df, vocab)
    assert 0 in Xs.index and 0 in X.index
    assert np.isnan(Xs.loc[0, "crs_elapsed_time"]) and np.isnan(X.loc[0, "crs_elapsed_time"])


def test_duration_rule_has_a_single_definition():
    import flightdelay.features as feat
    import flightdelay.validate as val
    assert feat.implausible_duration_mask is val.implausible_duration_mask
