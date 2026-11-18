"""Feature engineering (M2, D-003, D-024, P-005).

Contract
--------
build_features(df, vocab) -> pd.DataFrame
    Input:  a processed monthly batch (the Parquet schema from ingest.py) whose
            rows have ALREADY been selected with a validation mask, and a
            category vocabulary fitted on training rows.
    Output: one row per input row, same index, exactly FEATURE_COLUMNS in that
            order, with fixed dtypes (FEATURE_DTYPES).

Guarantees
----------
- Allowlist: only FEATURE_SOURCE_COLUMNS are read. The function first selects
  those columns and derives every feature from that subset, so a column that
  is not on the allowlist cannot influence the output, whatever it contains.
- No outcome information: FORBIDDEN_COLUMNS (post-departure and outcome
  fields) can never be on the allowlist; this is checked at call time.
- Deterministic and row-independent: no randomness, and a row's features do
  not depend on any other row in the batch. Categorical codes come from the
  fitted vocabulary, never from the batch, so "ATL" has the same code in every
  month. A category not in the vocabulary becomes missing (NaN).
- Fails loudly: a missing allowlisted column, a non-numeric numeric column,
  or an unusable (quarantined) row raises an error instead of producing
  silent NaNs. Use scoring_frame() / training_frame() to select rows.
- Impossible or implausible scheduled durations (D-028, D-030; rule in
  validate.implausible_duration_mask) are set to missing in the
  crs_elapsed_time feature only. The row and its other features are kept.

Row selection (D-024)
---------------------
- scoring_frame():  usable_features_mask  -> scoring a batch with labels hidden
                    and feature-drift monitoring. Rows without a label stay in.
- training_frame(): training_row_mask     -> training, validation, test,
                    labelled performance monitoring.
- fit_vocabulary(): built from training rows only.
"""

from __future__ import annotations

import pandas as pd

from flightdelay.validate import (  # noqa: F401  (duration rule re-exported for callers)
    MAX_BLOCK_SPEED_MPH,
    implausible_duration_mask,
    training_row_mask,
    usable_features_mask,
)

# Raw columns the feature code is allowed to read. Must be a subset of the
# allowed_feature_columns policy in configs/base.yaml (checked by a test).
FEATURE_SOURCE_COLUMNS = [
    "FlightDate",
    "Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "CRSArrTime",
    "CRSElapsedTime",
    "Distance",
]

# Post-departure or outcome columns of the BTS table (D-003). None of them may
# ever be read by the feature code. ArrDel15 is the target, never a feature.
FORBIDDEN_COLUMNS = frozenset({
    "DepTime", "DepDelay", "DepDelayMinutes", "DepDel15", "DepartureDelayGroups",
    "TaxiOut", "WheelsOff", "WheelsOn", "TaxiIn",
    "ArrTime", "ArrDelay", "ArrDelayMinutes", "ArrDel15", "ArrivalDelayGroups",
    "Cancelled", "CancellationCode", "Diverted",
    "ActualElapsedTime", "AirTime",
    "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay",
    "FirstDepTime", "TotalAddGTime", "LongestAddGTime",
    "DivAirportLandings", "DivReachedDest", "DivActualElapsedTime", "DivArrDelay",
    "DivDistance",
})

CATEGORICAL_FEATURES = ["airline", "origin", "dest"]
CATEGORICAL_SOURCES = {"airline": "Reporting_Airline", "origin": "Origin", "dest": "Dest"}

FEATURE_DTYPES = {
    "month": "int8",
    "day_of_week": "int8",          # Monday = 0
    "day_of_month": "int8",
    "dep_hour": "int8",             # scheduled, local time, 0-23
    "dep_minute_of_day": "int16",   # scheduled, 0-1439
    "arr_hour": "int8",
    "arr_minute_of_day": "int16",
    "crs_elapsed_time": "float32",  # NaN allowed (legitimately missing in source)
    "distance": "float32",
    "airline": "category",
    "origin": "category",
    "dest": "category",
}
FEATURE_COLUMNS = list(FEATURE_DTYPES)
NUMERIC_SOURCES = ["CRSDepTime", "CRSArrTime", "CRSElapsedTime", "Distance"]


def _check_allowlist() -> None:
    leaked = set(FEATURE_SOURCE_COLUMNS) & FORBIDDEN_COLUMNS
    if leaked:
        raise ValueError(f"forbidden columns on the feature allowlist: {sorted(leaked)}")


def fit_vocabulary(df: pd.DataFrame) -> dict[str, list[str]]:
    """Sorted category lists per categorical feature, from TRAINING rows only.

    The result is a plain dict of lists, so it can be stored as JSON next to a
    model and reused at scoring time.
    """
    train = df.loc[training_row_mask(df)]
    return {
        feat: sorted(train[src].dropna().astype(str).unique().tolist())
        for feat, src in CATEGORICAL_SOURCES.items()
    }


def _hhmm_to_hour_and_minute(hhmm: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Scheduled hhmm (0-2400) -> hour 0-23 and minute of day 0-1439. 2400 = 00:00."""
    v = hhmm.astype("int32")
    hour = (v // 100) % 24
    minute_of_day = hour * 60 + (v % 100)
    return hour, minute_of_day


def build_features(df: pd.DataFrame, vocab: dict[str, list[str]]) -> pd.DataFrame:
    """Build the model features from already-selected rows. See module docstring."""
    _check_allowlist()
    missing = [c for c in FEATURE_SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"missing feature source columns: {missing}")
    missing_vocab = [f for f in CATEGORICAL_FEATURES if f not in vocab]
    if missing_vocab:
        raise KeyError(f"vocabulary has no entry for: {missing_vocab}")

    src = df[FEATURE_SOURCE_COLUMNS]  # from here on, nothing else is visible
    for col in NUMERIC_SOURCES:
        if not pd.api.types.is_numeric_dtype(src[col]):
            raise TypeError(f"{col} must be numeric, got {src[col].dtype}")
    if not pd.api.types.is_datetime64_any_dtype(src["FlightDate"]):
        raise TypeError(f"FlightDate must be datetime, got {src['FlightDate'].dtype}")

    required = [c for c in FEATURE_SOURCE_COLUMNS if c != "CRSElapsedTime"]
    if bool(src[required].isna().any().any()):
        raise ValueError(
            "rows with missing required feature values; select rows with "
            "scoring_frame() or training_frame() (usable_features_mask, D-024)"
        )

    dep_hour, dep_min = _hhmm_to_hour_and_minute(src["CRSDepTime"])
    arr_hour, arr_min = _hhmm_to_hour_and_minute(src["CRSArrTime"])
    out = pd.DataFrame(
        {
            "month": src["FlightDate"].dt.month,
            "day_of_week": src["FlightDate"].dt.day_of_week,
            "day_of_month": src["FlightDate"].dt.day,
            "dep_hour": dep_hour,
            "dep_minute_of_day": dep_min,
            "arr_hour": arr_hour,
            "arr_minute_of_day": arr_min,
            "crs_elapsed_time": src["CRSElapsedTime"].mask(implausible_duration_mask(src)),
            "distance": src["Distance"],
        },
        index=src.index,
    )
    for feat, col in CATEGORICAL_SOURCES.items():
        values = src[col].astype(str)
        known = values.isin(vocab[feat])
        # Unseen categories become missing explicitly (pandas would otherwise
        # deprecate/raise on values outside the category list).
        out[feat] = pd.Categorical(values.where(known, None), categories=vocab[feat])
    return out[FEATURE_COLUMNS].astype(FEATURE_DTYPES)


def scoring_frame(df: pd.DataFrame, vocab: dict[str, list[str]]) -> pd.DataFrame:
    """Features for every row with usable features, whether or not it has a label.

    For scoring a batch with labels hidden and for feature-drift monitoring
    (D-024). Never filters on Cancelled, Diverted or ArrDel15.
    """
    return build_features(df.loc[usable_features_mask(df)], vocab)


def training_frame(df: pd.DataFrame, vocab: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.Series]:
    """Features and labels for rows with usable features AND a valid label (D-024)."""
    rows = df.loc[training_row_mask(df)]
    X = build_features(rows, vocab)
    y = rows["ArrDel15"].astype("int8").rename("y")
    return X, y
