"""Run the feature pipeline on real processed batches and record evidence (M2).

Read-only: Parquet files are only read. Holdout months (2025, D-020) are refused,
because statistics such as unseen-category rates would reveal the holdout.

Vocabulary: fitted on the training rows of the initial training period (2019).

Usage:
    python scripts/check_features.py --from 2019-01 --to 2024-12 --out report/evidence/feature_check_2019_2024.csv
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from flightdelay.config import PROJECT_ROOT, is_holdout_month, load_config
from flightdelay.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    FEATURE_DTYPES,
    fit_vocabulary,
    implausible_duration_mask,
    scoring_frame,
    training_frame,
)
from flightdelay.run_ingest import month_range


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="first", required=True)
    ap.add_argument("--to", dest="last", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    cfg = load_config()
    months = [f"{y}-{m:02d}" for y, m in month_range(args.first, args.last)]
    blocked = [m for m in months if is_holdout_month(m, cfg)]
    if blocked:
        print(f"refused: holdout months requested {blocked} (D-020)")
        return 2

    proc = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    vreport = pd.read_csv(PROJECT_ROOT / "data" / "validation_report.csv").set_index("month")

    t0, t1 = cfg["periods"]["initial_training"]
    train_months = [f"{y}-{m:02d}" for y, m in month_range(t0, t1)]
    base = pd.concat([pd.read_parquet(proc / f"{m[:4]}_{m[5:]}.parquet") for m in train_months],
                     ignore_index=True)
    vocab = fit_vocabulary(base)
    del base
    print("vocabulary sizes:", {k: len(v) for k, v in vocab.items()})

    rows, problems = [], []
    for m in months:
        df = pd.read_parquet(proc / f"{m[:4]}_{m[5:]}.parquet")
        Xs = scoring_frame(df, vocab)
        usable = df.loc[Xs.index]
        Xt, y = training_frame(df, vocab)
        r = {
            "month": m,
            "rows": len(df),
            "scoring_rows": len(Xs),
            "training_rows": len(Xt),
            "scoring_matches_report": len(Xs) == int(vreport.loc[m, "usable_feature_rows"]),
            "training_matches_report": len(Xt) == int(vreport.loc[m, "training_rows"]),
            "columns_ok": list(Xs.columns) == FEATURE_COLUMNS,
            "dtypes_ok": {c: str(t) for c, t in Xs.dtypes.items()} == FEATURE_DTYPES,
            "label_values": " ".join(map(str, sorted(y.unique()))),
            "crs_elapsed_nan": int(Xs["crs_elapsed_time"].isna().sum()),
            "duration_missing_in_source": int(usable["CRSElapsedTime"].isna().sum()),
            "duration_set_nan_d028": int(implausible_duration_mask(usable).sum()),
            "other_numeric_nan": int(Xs.drop(columns=["crs_elapsed_time", *CATEGORICAL_FEATURES]).isna().sum().sum()),
            "scoring_mem_MB": round(Xs.memory_usage(deep=True).sum() / 1e6, 1),
        }
        for f in CATEGORICAL_FEATURES:
            r[f"unseen_{f}_pct"] = round(Xs[f].isna().mean() * 100, 3)
        if r["crs_elapsed_nan"] != r["duration_missing_in_source"] + r["duration_set_nan_d028"]:
            problems.append(m)
        if not (r["scoring_matches_report"] and r["training_matches_report"]
                and r["columns_ok"] and r["dtypes_ok"] and r["other_numeric_nan"] == 0
                and r["label_values"] in ("0 1", "0", "1")):
            problems.append(m)
        rows.append(r)
        print(f"{m}: scoring={r['scoring_rows']} training={r['training_rows']} "
              f"unseen(airline/origin/dest)%={r['unseen_airline_pct']}/{r['unseen_origin_pct']}/{r['unseen_dest_pct']}")

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"done: {len(rows)} months, problems: {problems or 'none'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
