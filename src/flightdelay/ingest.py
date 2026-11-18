"""Ingestion: raw monthly BTS zip -> column-reduced Parquet + manifest entry.

Design (D-010, D-013, P-003):
- The original zip files are never modified; they are the raw layer.
- Only the columns the project needs are read, with compact dtypes.
- Every converted file gets a manifest row: SHA-256 of the zip, row count,
  Parquet path and creation time. The manifest is the data-version record.
"""

from __future__ import annotations

import hashlib
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

# Columns kept in the processed layer. Features (D-003) + target + exclusion flags (D-013).
KEEP_COLUMNS = [
    "FlightDate",
    "Reporting_Airline",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "CRSArrTime",
    "CRSElapsedTime",
    "Distance",
    "ArrDel15",
    "Cancelled",
    "Diverted",
    "DivReachedDest",
]

DTYPES = {
    "Reporting_Airline": "category",
    "Origin": "category",
    "Dest": "category",
    "Flight_Number_Reporting_Airline": "Int32",
    "CRSDepTime": "Int16",
    "CRSArrTime": "Int16",
    "CRSElapsedTime": "float32",
    "Distance": "float32",
    "ArrDel15": "float32",
    "Cancelled": "float32",
    "Diverted": "float32",
    "DivReachedDest": "float32",
}


@dataclass
class IngestResult:
    month: str
    zip_name: str
    zip_sha256: str
    zip_bytes: int
    rows: int
    parquet_path: str
    parquet_bytes: int
    ingested_at: str


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def raw_zip_path(raw_dir: Path, year: int, month: int) -> Path:
    name = f"On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
    return raw_dir / name


def convert_month(raw_dir: Path, out_dir: Path, year: int, month: int) -> IngestResult:
    """Convert one raw monthly zip to a Parquet file and return its manifest row."""
    zpath = raw_zip_path(raw_dir, year, month)
    with zipfile.ZipFile(zpath) as z:
        csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"{zpath.name}: expected exactly one CSV, found {csv_names}")
        with z.open(csv_names[0]) as f:
            df = pd.read_csv(f, usecols=KEEP_COLUMNS, dtype=DTYPES, parse_dates=["FlightDate"])
    df = df[KEEP_COLUMNS]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{year}_{month:02d}.parquet"
    df.to_parquet(out, index=False)
    return IngestResult(
        month=f"{year}-{month:02d}",
        zip_name=zpath.name,
        zip_sha256=sha256_of(zpath),
        zip_bytes=zpath.stat().st_size,
        rows=len(df),
        parquet_path=str(out),
        parquet_bytes=out.stat().st_size,
        ingested_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def append_manifest(manifest_path: Path, results: list[IngestResult]) -> None:
    rows = pd.DataFrame([asdict(r) for r in results])
    if manifest_path.exists() and manifest_path.stat().st_size > 0:
        old = pd.read_csv(manifest_path)
        rows = pd.concat([old[~old["month"].isin(rows["month"])], rows], ignore_index=True)
        rows = rows.sort_values("month")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(manifest_path, index=False)
