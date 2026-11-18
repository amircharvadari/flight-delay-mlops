"""CLI: convert raw months to Parquet, validate them, write manifest and reports.

Usage:
    python -m flightdelay.run_ingest --from 2019-01 --to 2019-12
    python -m flightdelay.run_ingest --from 2019-01 --to 2025-12 --validate-only

--validate-only re-validates the existing Parquet files without re-reading the
raw zips and without touching the manifest (used after a validation-rule change).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import pandas as pd

from flightdelay.config import PROJECT_ROOT, load_config, set_seed
from flightdelay.ingest import append_manifest, convert_month
from flightdelay.validate import validate_month


def merge_validation_report(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Replace re-validated months and keep the others.

    The result has exactly the columns of the CURRENT validator (`new`). Columns
    that the validator no longer produces are dropped, so a renamed or removed
    field cannot survive as a stale column. Months that were not re-validated
    get NaN in columns added since they were last validated.
    """
    kept = old[~old["month"].isin(new["month"])]
    merged = pd.concat([kept, new], ignore_index=True)
    return merged[list(new.columns)].sort_values("month").reset_index(drop=True)


def month_range(first: str, last: str) -> list[tuple[int, int]]:
    out = []
    y, m = int(first[:4]), int(first[5:7])
    while f"{y}-{m:02d}" <= last:
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--from", dest="first", required=True)
    ap.add_argument("--to", dest="last", required=True)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    set_seed(cfg["project"]["seed"])
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_dir"]
    out_dir = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    manifest = PROJECT_ROOT / cfg["paths"]["manifest"]
    vreport_path = PROJECT_ROOT / "data" / "validation_report.csv"

    ingest_rows, val_rows = [], []
    known_airlines: set[str] | None = None
    if vreport_path.exists() and vreport_path.stat().st_size > 0:
        prev = pd.read_csv(vreport_path)
        # Seed the airline set from the month right before the first processed month,
        # so partial re-runs compare against the correct neighbour.
        before = prev[prev["month"] < args.first]
        if len(before):
            known_airlines = set(str(before.iloc[-1]["airlines"]).split())

    for year, mon in month_range(args.first, args.last):
        month = f"{year}-{mon:02d}"
        if args.validate_only:
            df = pd.read_parquet(out_dir / f"{year}_{mon:02d}.parquet")
        else:
            res = convert_month(raw_dir, out_dir, year, mon)
            ingest_rows.append(res)
            df = pd.read_parquet(res.parquet_path)
        v = validate_month(df, month, known_airlines)
        known_airlines = set(df["Reporting_Airline"].dropna().astype(str).unique())
        row = asdict(v)
        row["airlines"] = " ".join(sorted(known_airlines))
        for k in ("new_airlines", "gone_airlines", "errors", "warnings"):
            row[k] = "; ".join(map(str, row[k]))
        val_rows.append(row)
        print(f"{month}: rows={v.rows} status={v.status} "
              f"excluded={v.excluded_pct}% delay={v.delay_rate_pct}%"
              + (f" | {v.errors or v.warnings}" if v.status != "pass" else ""))

    if ingest_rows:
        append_manifest(manifest, ingest_rows)
    vnew = pd.DataFrame(val_rows)
    if vreport_path.exists() and vreport_path.stat().st_size > 0:
        vnew = merge_validation_report(pd.read_csv(vreport_path), vnew)
    vnew.to_csv(vreport_path, index=False)
    failed = [r for r in val_rows if r["status"] == "fail"]
    print(f"done: {len(val_rows)} months, {len(failed)} failed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
