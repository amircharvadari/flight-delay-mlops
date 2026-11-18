"""Load a training window month by month (D-010) and guard the holdout (D-020).

Two passes over the months keep peak memory low:
1. vocabulary: category values seen in the TRAINING rows of the window (D-025);
2. features: each month is turned into a compact feature frame and only those
   frames are kept. The raw month frames are discarded immediately.
The vocabulary equals features.fit_vocabulary() on the whole window (tested).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from flightdelay.config import is_holdout_month, load_config
from flightdelay.features import CATEGORICAL_SOURCES, training_frame
from flightdelay.validate import training_row_mask


def months_between(first: str, last: str) -> list[str]:
    out, y, m = [], int(first[:4]), int(first[5:7])
    while f"{y}-{m:02d}" <= last:
        out.append(f"{y}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    if not out:
        raise ValueError(f"empty month range {first}..{last}")
    return out


def guard_holdout(months: list[str], allow_holdout: bool = False, config: dict | None = None) -> None:
    cfg = config or load_config()
    blocked = [m for m in months if is_holdout_month(m, cfg)]
    if blocked and not allow_holdout:
        raise PermissionError(f"holdout months requested without allow_holdout: {blocked} (D-020)")


def month_path(processed_dir: Path, month: str) -> Path:
    return Path(processed_dir) / f"{month[:4]}_{month[5:]}.parquet"


def window_vocabulary(months: list[str], processed_dir: Path) -> dict[str, list[str]]:
    values: dict[str, set[str]] = {f: set() for f in CATEGORICAL_SOURCES}
    for m in months:
        df = pd.read_parquet(month_path(processed_dir, m))
        tr = df.loc[training_row_mask(df)]
        for feat, src in CATEGORICAL_SOURCES.items():
            values[feat].update(tr[src].dropna().astype(str).unique().tolist())
    return {f: sorted(v) for f, v in values.items()}


def load_training_window(months: list[str], processed_dir: Path,
                         vocab: dict[str, list[str]] | None = None
                         ) -> tuple[pd.DataFrame, pd.Series, dict[str, list[str]], dict[str, Any]]:
    """Feature frame X, labels y, vocabulary and window info for the given months."""
    vocab = vocab or window_vocabulary(months, processed_dir)
    Xs, ys, raw_rows = [], [], 0
    for m in months:
        df = pd.read_parquet(month_path(processed_dir, m))
        raw_rows += len(df)
        X, y = training_frame(df, vocab)
        Xs.append(X)
        ys.append(y)
        del df
    X = pd.concat(Xs, ignore_index=True)
    y = pd.concat(ys, ignore_index=True)
    info = {"train_from": months[0], "train_to": months[-1], "train_months": len(months),
            "window_raw_rows": raw_rows}
    return X, y, vocab, info
