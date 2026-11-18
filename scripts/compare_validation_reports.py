"""Compare two validation reports month by month (evidence for I-13 and later rule changes).

Usage:
    python scripts/compare_validation_reports.py BEFORE.csv AFTER.csv \
        [--normalize-warning "NEW TEXT=>OLD TEXT"] [--strict]

What it checks, for months present in both files:
- numeric columns: equal within --atol (default 0, i.e. exact), NaN equal to NaN
- text columns (status, airlines, errors, warnings, ...): exact after treating
  missing as the empty string; warnings optionally after wording normalisation
- columns present in only one file are listed, not compared

Exit code: 0 if no compared column differs, 1 if something differs and --strict.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

TEXT_COLUMNS = ["status", "airlines", "new_airlines", "gone_airlines", "errors", "warnings"]


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).set_index("month").sort_index()
    for c in TEXT_COLUMNS:
        if c in df:
            df[c] = df[c].astype("string").fillna("")
    return df


def compare(before: pd.DataFrame, after: pd.DataFrame, atol: float = 0.0,
            replacements: list[tuple[str, str]] | None = None) -> dict:
    months = before.index.intersection(after.index)
    only_months = sorted(set(before.index) ^ set(after.index))
    shared = [c for c in before.columns if c in after.columns]
    result = {
        "months_compared": len(months),
        "months_in_one_file_only": only_months,
        "columns_only_before": sorted(set(before.columns) - set(after.columns)),
        "columns_only_after": sorted(set(after.columns) - set(before.columns)),
        "differences": {},
    }
    b, a = before.loc[months], after.loc[months]
    for c in shared:
        if c in TEXT_COLUMNS:
            av, bv = a[c], b[c]
            if c == "warnings" and replacements:
                for new, old in replacements:
                    av = av.str.replace(new, old, regex=False)
            diff = av != bv
        else:
            diff = ~np.isclose(a[c].astype(float), b[c].astype(float), atol=atol, rtol=0, equal_nan=True)
            diff = pd.Series(diff, index=months)
        if bool(diff.any()):
            result["differences"][c] = list(months[diff.to_numpy()])
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--atol", type=float, default=0.0)
    ap.add_argument("--normalize-warning", action="append", default=[],
                    help='wording change to ignore in warnings, as "NEW=>OLD"')
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args(argv)

    reps = [tuple(x.split("=>", 1)) for x in args.normalize_warning]
    before, after = load(args.before), load(args.after)
    r = compare(before, after, args.atol, reps)

    print(f"months compared: {r['months_compared']}")
    print(f"months in one file only: {r['months_in_one_file_only'] or 'none'}")
    print(f"columns only in BEFORE: {r['columns_only_before'] or 'none'}")
    print(f"columns only in AFTER:  {r['columns_only_after'] or 'none'}")
    print(f"status counts BEFORE: {before['status'].value_counts().to_dict()}")
    print(f"status counts AFTER:  {after['status'].value_counts().to_dict()}")
    if not r["differences"]:
        print("RESULT: no differences in compared columns")
        return 0
    for c, ms in r["differences"].items():
        print(f"DIFF {c}: {len(ms)} months -> {', '.join(ms[:12])}{' ...' if len(ms) > 12 else ''}")
    return 1 if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
