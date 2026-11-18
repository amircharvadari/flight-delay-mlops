"""Ingestion and validation tests on a small synthetic fixture."""

import calendar
import io
import zipfile

import pandas as pd
import pytest

from flightdelay.config import load_config
from flightdelay.ingest import KEEP_COLUMNS, convert_month
from flightdelay.validate import (
    QUARANTINE_FAIL_PCT,
    REQUIRED_NOT_NULL,
    schema,
    training_row_mask,
    usable_features_mask,
    validate_month,
)


def make_month_df(n=200, year=2019, month=1) -> pd.DataFrame:
    # Cover every day of the month, so a clean fixture validates as "pass".
    # (An earlier version covered only 28 days, so every fixture batch already
    # carried a missing-days warning and "warn" assertions proved little.)
    days = pd.date_range(f"{year}-{month:02d}-01", periods=calendar.monthrange(year, month)[1], freq="D")
    df = pd.DataFrame(
        {
            "FlightDate": [days[i % len(days)] for i in range(n)],
            "Reporting_Airline": pd.Categorical(["AA", "DL"][0::1] * 0 + ["AA" if i % 2 else "DL" for i in range(n)]),
            "Flight_Number_Reporting_Airline": pd.array(range(100, 100 + n), dtype="Int32"),
            "Origin": pd.Categorical(["JFK" if i % 3 else "LAX" for i in range(n)]),
            "Dest": pd.Categorical(["ORD" if i % 3 else "ATL" for i in range(n)]),
            "CRSDepTime": pd.array([800 + (i % 12) * 100 for i in range(n)], dtype="Int16"),
            "CRSArrTime": pd.array([1000 + (i % 12) * 100 for i in range(n)], dtype="Int16"),
            "CRSElapsedTime": pd.array([120.0] * n, dtype="float32"),
            "Distance": pd.array([500.0] * n, dtype="float32"),
            "ArrDel15": pd.array([float(i % 5 == 0) for i in range(n)], dtype="float32"),
            "Cancelled": pd.array([0.0] * n, dtype="float32"),
            "Diverted": pd.array([0.0] * n, dtype="float32"),
            "DivReachedDest": pd.array([None] * n, dtype="float32"),
        }
    )
    return df


def write_fixture_zip(tmp_path, df, year=2019, month=1):
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    buf = io.StringIO()
    extra = df.copy()
    extra["DepTime"] = 900  # a post-departure column that must be dropped on ingest
    extra.to_csv(buf, index=False)
    name = f"On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
    with zipfile.ZipFile(raw / name, "w") as z:
        z.writestr("data.csv", buf.getvalue())
        z.writestr("readme.html", "<html></html>")
    return raw


def test_convert_month_keeps_only_selected_columns(tmp_path):
    df = make_month_df()
    raw = write_fixture_zip(tmp_path, df)
    res = convert_month(raw, tmp_path / "out", 2019, 1)
    got = pd.read_parquet(res.parquet_path)
    assert list(got.columns) == KEEP_COLUMNS
    assert "DepTime" not in got.columns
    assert res.rows == len(df)
    assert len(res.zip_sha256) == 64


def test_validate_pass_and_counts():
    df = make_month_df()
    v = validate_month(df, "2019-01")
    assert v.status == "pass", (v.errors, v.warnings)
    assert v.usable_feature_rows == 200
    assert v.training_rows == 200
    assert v.rows == 200
    assert v.excluded_rows == 0
    assert v.delay_rate_pct == pytest.approx(20.0, abs=0.5)


def test_validate_fails_on_duplicates_and_wrong_month():
    df = make_month_df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # duplicate
    v = validate_month(df, "2019-02")  # every row is outside 2019-02
    assert v.status == "fail"
    assert any("duplicate" in e for e in v.errors)
    assert any("outside" in e for e in v.errors)


def test_validate_excludes_invalid_labels_and_warns_on_new_airline():
    df = make_month_df()
    df.loc[:9, "ArrDel15"] = None
    df.loc[:9, "Cancelled"] = 1.0
    v = validate_month(df, "2019-01", known_airlines={"AA"})
    assert v.excluded_rows == 10
    assert v.labeled_rows == 190
    assert "DL" in v.new_airlines
    assert v.status == "warn"


def test_impossible_duration_does_not_quarantine_the_row():
    """D-030 (revises D-023): -77 min was observed in the real 2020-01 file.
    Only the duration feature becomes missing; the row stays usable."""
    df = make_month_df()
    df.loc[0, "CRSElapsedTime"] = -77.0
    v = validate_month(df, "2019-01")
    assert v.status == "pass", (v.errors, v.warnings)
    assert v.quarantined_rows == 0
    assert v.duration_set_nan_rows == 1
    assert v.usable_feature_rows == 200 and v.training_rows == 200


@pytest.mark.parametrize("bad", [-77.0, 0.0, 0.5, 1601.0, 5000.0])
def test_out_of_range_duration_is_counted_not_quarantined(bad):
    df = make_month_df()
    df.loc[0, "CRSElapsedTime"] = bad
    v = validate_month(df, "2019-01")
    assert v.quarantined_rows == 0 and v.duration_set_nan_rows == 1
    assert bool(usable_features_mask(df).iloc[0])


@pytest.mark.parametrize("col,bad", [
    ("Distance", -1.0), ("Distance", 0.0), ("Distance", 7000.0),
    ("CRSDepTime", 2401), ("CRSArrTime", 2500),
])
def test_out_of_range_required_feature_still_quarantines_the_row(col, bad):
    df = make_month_df()
    df.loc[0, col] = bad
    v = validate_month(df, "2019-01")
    assert v.status == "warn" and v.quarantined_rows == 1
    assert not bool(usable_features_mask(df).iloc[0])


def test_duration_count_excludes_quarantined_rows():
    df = make_month_df()
    df.loc[0, "CRSElapsedTime"] = -5.0
    df.loc[0, "Origin"] = None            # row quarantined for another reason
    v = validate_month(df, "2019-01")
    assert v.quarantined_rows == 1 and v.duration_set_nan_rows == 0


def test_many_out_of_range_rows_fail_the_batch():
    df = make_month_df()
    df.loc[: len(df) // 10, "Distance"] = -1.0
    v = validate_month(df, "2019-01")
    assert v.status == "fail"


def test_missing_flight_number_is_quarantined():
    df = make_month_df()
    df.loc[0, "Flight_Number_Reporting_Airline"] = None  # observed in the real 2024-08 file
    v = validate_month(df, "2019-01")
    assert v.status == "warn"
    assert v.quarantined_rows == 1
def test_missing_origin_is_quarantined_not_fatal():
    df = make_month_df()

    df.loc[0, "Origin"] = None

    v = validate_month(df, "2019-01")

    assert v.status == "warn"
    assert v.quarantined_rows == 1


# ---------------------------------------------------------------------------
# D-023 / D-024: row-level quarantine of missing required values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("col", REQUIRED_NOT_NULL)
def test_null_in_any_required_column_is_quarantined_not_fatal(col):
    df = make_month_df()
    df.loc[5, col] = None
    v = validate_month(df, "2019-01")
    assert v.errors == [], v.errors
    assert v.status == "warn"
    assert v.quarantined_rows == 1
    assert v.usable_feature_rows == 199
    assert v.training_rows == 199
    assert not bool(usable_features_mask(df).iloc[5])
    assert not bool(training_row_mask(df).iloc[5])


def test_schema_nullability_matches_quarantine_policy():
    """The schema and the quarantine policy must never silently diverge.

    - Every REQUIRED_NOT_NULL column must be nullable in the schema; otherwise a
      single null fails the batch before row_quarantine_mask() can act.
    - The only strict (non-nullable) columns are the structural flags. Making any
      other column strict, or adding a new strict one, must be a conscious change
      to this test.
    - Every configured feature column is either quarantined when null or is a
      column where null is legitimate.
    """
    nullable = {name: col.nullable for name, col in schema.columns.items()}
    for col in REQUIRED_NOT_NULL:
        assert col in nullable, f"{col} missing from schema"
        assert nullable[col], f"{col} is required-not-null but not nullable in the schema"
    strict = {name for name, is_nullable in nullable.items() if not is_nullable}
    assert strict == {"Cancelled", "Diverted"}
    legitimately_nullable_features = {"CRSElapsedTime"}
    features = set(load_config()["allowed_feature_columns"])
    assert features <= set(REQUIRED_NOT_NULL) | legitimately_nullable_features


def test_missing_flightdate_quarantined_but_wrong_month_date_still_fails():
    df = make_month_df()
    df.loc[0, "FlightDate"] = pd.NaT
    v = validate_month(df, "2019-01")
    assert v.errors == [] and v.status == "warn" and v.quarantined_rows == 1

    df.loc[1, "FlightDate"] = pd.Timestamp("2019-02-03")
    v = validate_month(df, "2019-01")
    assert v.status == "fail"
    assert any("1 rows outside 2019-01" in e for e in v.errors)


# ---------------------------------------------------------------------------
# Structural checks that must stay strict
# ---------------------------------------------------------------------------

def test_wrong_dtype_still_fails_the_batch():
    df = make_month_df()
    df["Origin"] = df["Origin"].astype(str)  # object instead of category
    v = validate_month(df, "2019-01")
    assert v.status == "fail"
    assert any("schema" in e for e in v.errors)


def test_missing_required_column_still_fails_the_batch():
    df = make_month_df().drop(columns=["Dest"])
    v = validate_month(df, "2019-01")
    assert v.status == "fail"
    assert any("missing columns" in e for e in v.errors)


def test_invalid_cancelled_flag_still_fails_the_batch():
    df = make_month_df()
    df.loc[0, "Cancelled"] = 2.0
    v = validate_month(df, "2019-01")
    assert v.status == "fail"


def test_quarantine_threshold_boundary():
    df = make_month_df(n=1000)
    df["Flight_Number_Reporting_Airline"] = pd.array(range(1000, 2000), dtype="Int32")
    at_limit = int(len(df) * QUARANTINE_FAIL_PCT / 100)  # 5 rows = exactly 0.5%
    df.loc[: at_limit - 1, "Distance"] = None
    assert validate_month(df, "2019-01").status == "warn"
    df.loc[at_limit, "Distance"] = None  # 0.6%
    assert validate_month(df, "2019-01").status == "fail"


# ---------------------------------------------------------------------------
# D-013: invalid labels are excluded and reported, not batch-fatal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [2.0, -1.0, 0.5])
def test_invalid_label_is_excluded_not_fatal(bad):
    df = make_month_df()
    df.loc[3, "ArrDel15"] = bad
    v = validate_month(df, "2019-01")
    assert v.errors == [], v.errors
    assert v.status == "warn"
    assert v.excluded_rows == 1
    assert v.invalid_label_rows == 1
    assert v.labeled_rows == 199
    assert any("invalid label value" in w for w in v.warnings)
    # never a training label
    mask = training_row_mask(df)
    assert not bool(mask.iloc[3])
    assert set(df.loc[mask, "ArrDel15"].unique()) <= {0.0, 1.0}


def test_delay_rate_ignores_invalid_labels():
    df = make_month_df()
    expected = df["ArrDel15"].iloc[1:].mean() * 100
    df.loc[0, "ArrDel15"] = 2.0
    v = validate_month(df, "2019-01")
    assert v.delay_rate_pct == pytest.approx(expected, abs=0.01)


def test_missing_label_on_normal_flight_keeps_d013_warning():
    df = make_month_df()
    df.loc[0, "ArrDel15"] = None  # not cancelled, not diverted
    v = validate_month(df, "2019-01")
    assert v.status == "warn"
    assert v.excluded_rows == 1 and v.invalid_label_rows == 0
    assert any("lack a label but are not cancelled/diverted" in w for w in v.warnings)


def test_training_row_mask_combines_quarantine_and_label_rules():
    df = make_month_df()
    df.loc[0, "Origin"] = None        # quarantined, label valid
    df.loc[1, "ArrDel15"] = None      # label missing
    df.loc[1, "Cancelled"] = 1.0
    df.loc[2, "ArrDel15"] = 2.0       # label invalid
    mask = training_row_mask(df)
    assert list(mask.iloc[:4]) == [False, False, False, True]
    assert int(mask.sum()) == 197



# ---------------------------------------------------------------------------
# D-024: two masks — features (scoring, drift) vs training (needs a label)
# ---------------------------------------------------------------------------

def test_row_without_label_is_usable_for_features_but_not_for_training():
    df = make_month_df()
    df.loc[0, "ArrDel15"] = None      # e.g. a flight that was later cancelled
    df.loc[0, "Cancelled"] = 1.0
    assert bool(usable_features_mask(df).iloc[0])
    assert not bool(training_row_mask(df).iloc[0])
    v = validate_month(df, "2019-01")
    assert v.usable_feature_rows == 200
    assert v.training_rows == 199


def test_usable_features_mask_does_not_depend_on_post_flight_columns():
    """Scoring happens before departure: the feature mask must not read
    Cancelled, Diverted or ArrDel15 (they would leak the outcome)."""
    df = make_month_df()
    df.loc[0, "Origin"] = None        # one genuinely quarantined row
    expected = usable_features_mask(df)

    changed = df.copy()
    changed["ArrDel15"] = pd.array([None] * len(df), dtype="float32")
    changed["Cancelled"] = pd.array([1.0] * len(df), dtype="float32")
    changed["Diverted"] = pd.array([1.0] * len(df), dtype="float32")
    pd.testing.assert_series_equal(usable_features_mask(changed), expected)

    dropped = df.drop(columns=["ArrDel15", "Cancelled", "Diverted", "DivReachedDest"])
    pd.testing.assert_series_equal(usable_features_mask(dropped), expected)


def test_training_mask_is_subset_of_feature_mask():
    df = make_month_df()
    df.loc[0, "Distance"] = None
    df.loc[1, "ArrDel15"] = 2.0
    feat, train = usable_features_mask(df), training_row_mask(df)
    assert not bool((train & ~feat).any())
    assert int(feat.sum()) == 199 and int(train.sum()) == 198



def test_merge_validation_report_drops_stale_columns():
    from flightdelay.run_ingest import merge_validation_report
    old = pd.DataFrame({"month": ["2019-01", "2019-02"], "status": ["pass", "pass"],
                        "usable_rows": [10, 20]})
    new = pd.DataFrame({"month": ["2019-02"], "status": ["warn"], "training_rows": [19]})
    m = merge_validation_report(old, new)
    assert list(m.columns) == ["month", "status", "training_rows"]
    assert list(m["month"]) == ["2019-01", "2019-02"]
    assert m.loc[m.month == "2019-02", "status"].item() == "warn"
    assert pd.isna(m.loc[m.month == "2019-01", "training_rows"].item())
