"""Batch validation (D-008 input side, D-013, D-023, D-024).

Each monthly batch is checked before it is used. The outcome is one of:
- pass:  all checks hold, no warnings
- warn:  batch is usable, but something was flagged (quarantined rows, invalid
         or unexplained labels, new or missing airline categories, missing days)
- fail:  a structural check is broken, or too many rows are quarantined

Two layers with separate responsibilities (D-024):

1. Structural layer (batch level, can fail the batch):
   - all required columns present
   - every column has the expected dtype (Pandera schema)
   - Cancelled / Diverted contain only 0 or 1 (non-feature flags; a bad value
     here indicates a format problem, not a single bad flight)
   - no non-null FlightDate outside the batch month
   - no duplicate flights on the natural key
   - quarantined share not above QUARANTINE_FAIL_PCT

2. Row level (never fails the batch on its own):
   - QUARANTINE (D-023): a row with a missing required feature value or an
     out-of-range feature value. Implemented in row_quarantine_mask().
   - LABEL EXCLUSION (D-013): a row whose ArrDel15 is missing or not in {0, 1}.
   Both kinds of row are counted and reported here.

Important: this module does NOT remove rows from the Parquet files. The
processed Parquet layer keeps every source row, so the data stays traceable to
the raw zip. Downstream steps select rows with one of two masks (D-024):

- usable_features_mask(): the row's pre-departure features can be given to the
  model. Used for scoring a batch (labels hidden) and for feature-drift
  monitoring. It must never read Cancelled, Diverted or ArrDel15, because those
  are only known after the flight (leakage).
- training_row_mask(): usable features AND a valid label. Used for training,
  validation and test, labelled performance monitoring, shadow evaluation and
  prevalence monitoring.

Population rule: a drift reference and the batch it is compared with must be
selected with the same mask, otherwise the comparison measures the mask and not
the data (for example the 41.5% cancellations of 2020-04).
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field

import pandas as pd
import pandera.pandas as pa

from flightdelay.ingest import KEEP_COLUMNS

KEY = ["FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline",
       "Origin", "Dest", "CRSDepTime"]

QUARANTINE_FAIL_PCT = 0.5  # D-023: fail the batch above this share of quarantined rows

VALID_LABELS = [0.0, 1.0]

# Feature columns whose missing value makes a row unusable (D-023).
# Every column listed here MUST be nullable in `schema`, otherwise a single null
# would fail the batch before the row can be quarantined. This is enforced by
# tests/test_ingest_validate.py::test_schema_nullability_matches_quarantine_policy.
REQUIRED_NOT_NULL = ["FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline",
                     "Origin", "Dest", "CRSDepTime", "CRSArrTime", "Distance"]


def row_quarantine_mask(df: pd.DataFrame) -> pd.Series:
    """True for rows that are unusable: missing or out-of-range feature values."""
    ok = (
        df["CRSDepTime"].between(0, 2400)
        & df["CRSArrTime"].between(0, 2400)
        & (df["CRSElapsedTime"].between(1, 1600) | df["CRSElapsedTime"].isna())
        & df["Distance"].between(1, 6000)
    )
    for col in REQUIRED_NOT_NULL:
        ok &= df[col].notna()
    return ~ok.fillna(False).astype(bool)


def usable_features_mask(df: pd.DataFrame) -> pd.Series:
    """True for rows whose features can be given to the model (D-024).

    Not quarantined (D-023). Independent of label availability, so it also keeps
    flights that were later cancelled or diverted: at prediction time that is
    not yet known. Uses feature columns only.
    """
    return ~row_quarantine_mask(df)


def valid_label_mask(df: pd.DataFrame) -> pd.Series:
    """True for rows whose ArrDel15 is exactly 0 or 1 (D-013)."""
    return df["ArrDel15"].isin(VALID_LABELS)


def training_row_mask(df: pd.DataFrame) -> pd.Series:
    """True for rows that may be used where a label is needed (D-024).

    Usable features AND a valid label: training, validation and test,
    labelled performance monitoring, shadow evaluation, prevalence monitoring.
    """
    return usable_features_mask(df) & valid_label_mask(df)


# Structural schema. Dtypes are checked for every column. Nullability:
# - nullable=True for REQUIRED_NOT_NULL columns: nulls are a row-level problem,
#   handled by row_quarantine_mask() (D-023)
# - nullable=True for CRSElapsedTime and DivReachedDest: nulls are legitimate
# - ArrDel15: dtype only; value validity is a row-level problem (D-013)
# - Cancelled / Diverted: strict, structural flags (see module docstring)
schema = pa.DataFrameSchema(
    {
        "FlightDate": pa.Column("datetime64[ns]", nullable=True),
        "Reporting_Airline": pa.Column("category", nullable=True),
        "Flight_Number_Reporting_Airline": pa.Column("Int32", nullable=True),
        "Origin": pa.Column("category", nullable=True),
        "Dest": pa.Column("category", nullable=True),
        "CRSDepTime": pa.Column("Int16", nullable=True),
        "CRSArrTime": pa.Column("Int16", nullable=True),
        "CRSElapsedTime": pa.Column("float32", nullable=True),
        "Distance": pa.Column("float32", nullable=True),
        "ArrDel15": pa.Column("float32", nullable=True),
        "Cancelled": pa.Column("float32", pa.Check.isin(VALID_LABELS)),
        "Diverted": pa.Column("float32", pa.Check.isin(VALID_LABELS)),
        "DivReachedDest": pa.Column("float32", pa.Check.isin(VALID_LABELS), nullable=True),
    },
    strict=False,
)


@dataclass
class ValidationResult:
    month: str
    status: str                      # pass / warn / fail
    rows: int
    labeled_rows: int                # rows with a valid label (0 or 1)
    excluded_rows: int               # missing OR invalid label (D-013)
    excluded_pct: float
    invalid_label_rows: int          # subset of excluded_rows: non-null but not 0/1
    quarantined_rows: int            # missing or out-of-range feature values (D-023)
    quarantined_pct: float
    usable_feature_rows: int         # rows passing usable_features_mask() (scoring, drift)
    training_rows: int               # rows passing training_row_mask() (needs a label)
    delay_rate_pct: float            # among rows with a valid label
    duplicates: int
    days_expected: int
    days_present: int
    new_airlines: list[str] = field(default_factory=list)
    gone_airlines: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_month(
    df: pd.DataFrame, month: str, known_airlines: set[str] | None = None
) -> ValidationResult:
    year, mon = int(month[:4]), int(month[5:7])
    errors: list[str] = []
    warnings: list[str] = []

    # --- Structural layer -------------------------------------------------
    missing_cols = [c for c in KEEP_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"missing columns: {missing_cols}")
    else:
        try:
            schema.validate(df, lazy=True)
        except pa.errors.SchemaErrors as e:
            errors.append(f"schema: {len(e.failure_cases)} failing checks")

    days_expected = calendar.monthrange(year, mon)[1]
    days_present = 0
    if "FlightDate" in df:
        dates = df["FlightDate"]
        days_present = int(dates.dropna().dt.date.nunique())
        # Missing dates are a row-level problem (quarantined below).
        # A non-null date in another month is a structural error.
        wrong_month = dates.notna() & ~((dates.dt.year == year) & (dates.dt.month == mon))
        if bool(wrong_month.any()):
            errors.append(f"{int(wrong_month.sum())} rows outside {month}")
    if days_present < days_expected:
        warnings.append(f"only {days_present}/{days_expected} days present")

    dup = int(df.duplicated(KEY).sum()) if not missing_cols else -1
    if dup > 0:
        errors.append(f"{dup} duplicate flights on the natural key")

    # --- Row level: quarantine (D-023) -------------------------------------
    quarantined = 0
    usable_feat = 0
    training = 0
    if not missing_cols:
        qmask = row_quarantine_mask(df)
        quarantined = int(qmask.sum())
        q_pct = quarantined / len(df) * 100 if len(df) else 0.0
        if q_pct > QUARANTINE_FAIL_PCT:
            errors.append(f"{quarantined} rows ({q_pct:.2f}%) quarantined, above {QUARANTINE_FAIL_PCT}%")
        elif quarantined:
            warnings.append(f"{quarantined} rows quarantined (missing or out-of-range feature values)")
        usable_feat = int(usable_features_mask(df).sum())
        training = int(training_row_mask(df).sum())

    # --- Row level: label exclusion (D-013) --------------------------------
    label = df["ArrDel15"] if "ArrDel15" in df else pd.Series(dtype="float32")
    valid_label = label.isin(VALID_LABELS)
    excluded = int((~valid_label).sum())
    labeled = int(valid_label.sum())
    invalid_nonnull = int((label.notna() & ~valid_label).sum())
    if invalid_nonnull:
        warnings.append(f"{invalid_nonnull} rows have an invalid label value (not 0/1); excluded")
    # D-013 evidence: every excluded row should be cancelled or diverted.
    if {"Cancelled", "Diverted"} <= set(df.columns):
        unexplained = int((~valid_label & (df["Cancelled"] != 1) & (df["Diverted"] != 1)).sum())
        if unexplained:
            warnings.append(f"{unexplained} rows lack a label but are not cancelled/diverted")

    # --- Category drift (soft) ---------------------------------------------
    airlines = (
        set(df["Reporting_Airline"].dropna().astype(str).unique())
        if "Reporting_Airline" in df
        else set()
    )
    new_a: list[str] = []
    gone_a: list[str] = []
    if known_airlines is not None:
        new_a = sorted(airlines - known_airlines)
        gone_a = sorted(known_airlines - airlines)
        if new_a:
            warnings.append(f"new airline codes: {new_a}")
        if gone_a:
            warnings.append(f"airline codes no longer present: {gone_a}")

    status = "fail" if errors else ("warn" if warnings else "pass")
    return ValidationResult(
        month=month,
        status=status,
        rows=len(df),
        labeled_rows=labeled,
        excluded_rows=excluded,
        excluded_pct=round(excluded / len(df) * 100, 2) if len(df) else 0.0,
        invalid_label_rows=invalid_nonnull,
        quarantined_rows=quarantined,
        quarantined_pct=round(quarantined / len(df) * 100, 4) if len(df) else 0.0,
        usable_feature_rows=usable_feat,
        training_rows=training,
        delay_rate_pct=round(float(label[valid_label].mean()) * 100, 2) if labeled else 0.0,
        duplicates=max(dup, 0),
        days_expected=days_expected,
        days_present=days_present,
        new_airlines=new_a,
        gone_airlines=gone_a,
        errors=errors,
        warnings=warnings,
    )
