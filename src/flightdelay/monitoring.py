"""Monitoring signals (M6; D-008, D-024, D-043).

All functions are pure: they take feature frames, labels or scores and return
numbers. Nothing here decides about retraining; that is policy.py (D-012).

Populations (D-024): feature drift uses usable rows (scoring_frame); prevalence
and performance use labelled rows (training_row_mask).

Feature drift: Population Stability Index (PSI)
    PSI = sum over bins (cur_share - ref_share) * ln(cur_share / ref_share)
Numeric features: bins are the quantiles of the reference sample, so every
reference bin holds about the same share. Missing values get their own bin.
Categorical features: one bin per category of the model's vocabulary plus one
for missing/unseen. Shares below psi_floor are raised to psi_floor so the
logarithm is defined. Reference month (O-007): the same calendar month inside
the serving model's own training window.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from flightdelay.features import CATEGORICAL_FEATURES
from flightdelay.grid import add_months

MISSING = "__missing__"


def _shares(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    return counts / total if total else np.zeros_like(counts, dtype="float64")


def psi_from_shares(ref: np.ndarray, cur: np.ndarray, floor: float) -> float:
    r = np.maximum(np.asarray(ref, dtype="float64"), floor)
    c = np.maximum(np.asarray(cur, dtype="float64"), floor)
    return float(np.sum((c - r) * np.log(c / r)))


def numeric_bin_edges(ref: pd.Series, bins: int) -> np.ndarray:
    """Inner quantile edges of the non-missing reference values (unique)."""
    v = ref.dropna().to_numpy(dtype="float64")
    if len(v) == 0:
        return np.array([])
    return np.unique(np.quantile(v, np.linspace(0, 1, bins + 1)[1:-1]))


def numeric_counts(x: pd.Series, edges: np.ndarray) -> np.ndarray:
    v = x.to_numpy(dtype="float64")
    miss = np.isnan(v)
    idx = np.searchsorted(edges, v[~miss], side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1).astype("float64")
    return np.append(counts, miss.sum())


def categorical_counts(x: pd.Series, categories: list[str]) -> np.ndarray:
    s = x.astype("object").where(x.notna(), MISSING)
    vc = s.value_counts()
    return np.array([vc.get(c, 0) for c in [*categories, MISSING]], dtype="float64")


def feature_psi(ref_X: pd.DataFrame, cur_X: pd.DataFrame, features: list[str],
                vocab: dict[str, list[str]], bins: int, floor: float) -> dict[str, float]:
    out = {}
    for f in features:
        if f in CATEGORICAL_FEATURES:
            r, c = categorical_counts(ref_X[f], vocab[f]), categorical_counts(cur_X[f], vocab[f])
        else:
            edges = numeric_bin_edges(ref_X[f], bins)
            r, c = numeric_counts(ref_X[f], edges), numeric_counts(cur_X[f], edges)
        out[f] = psi_from_shares(_shares(r), _shares(c), floor)
    return out


def drift_level(psi: float, warning: float, alert: float) -> str:
    return "alert" if psi >= alert else "warning" if psi >= warning else "ok"


def reference_month(window_months: list[str], month: str) -> str:
    """Same calendar month inside the training window (O-007, D-043)."""
    same = [m for m in window_months if m[5:7] == month[5:7]]
    if not same:
        raise ValueError(f"training window {window_months[0]}..{window_months[-1]} has no "
                         f"month {month[5:7]}")
    return same[-1]


def unseen_category_share(df: pd.DataFrame, vocab: dict[str, list[str]]) -> float:
    """Share of rows with an airline, origin or destination not in the vocabulary."""
    from flightdelay.features import CATEGORICAL_SOURCES
    unseen = np.zeros(len(df), dtype=bool)
    for feat, col in CATEGORICAL_SOURCES.items():
        unseen |= ~df[col].astype(str).isin(vocab[feat]).to_numpy()
    return float(unseen.mean()) if len(df) else 0.0


def feature_drift_signal(ref_X: pd.DataFrame, cur_X: pd.DataFrame, vocab: dict[str, list[str]],
                         cfg: dict[str, Any]) -> dict[str, Any]:
    fd = cfg["feature_drift"]
    psi = feature_psi(ref_X, cur_X, fd["features"], vocab, fd["numeric_bins"], fd["psi_floor"])
    levels = {f: drift_level(v, fd["warning"], fd["alert"]) for f, v in psi.items()}
    worst = max(psi, key=psi.get)
    return {"psi": psi, "levels": levels, "max_psi": psi[worst], "max_psi_feature": worst,
            "n_alert": sum(lv == "alert" for lv in levels.values()),
            "n_warning": sum(lv == "warning" for lv in levels.values())}


def prevalence_signal(prev_t: float, prev_last_year: float | None,
                      cfg: dict[str, Any]) -> dict[str, Any]:
    """Prevalence vs the same month one year earlier (signal only, D-043)."""
    if prev_last_year is None or np.isnan(prev_last_year):
        return {"prevalence": prev_t, "prevalence_last_year": None, "abs_diff": None,
                "alert": False}
    diff = prev_t - prev_last_year
    return {"prevalence": prev_t, "prevalence_last_year": prev_last_year, "abs_diff": diff,
            "alert": bool(abs(diff) > cfg["prevalence"]["alert_abs_diff"])}


def volume_signal(rows: int, usable_rows: int, labelled_rows: int,
                  rows_last_year: int | None) -> dict[str, Any]:
    """Volume and quality (O-016, D-043): reported only."""
    return {
        "rows": rows,
        "unscorable_rows": rows - usable_rows,
        "unscorable_share": (rows - usable_rows) / rows if rows else 0.0,
        "labelled_share_of_scored": labelled_rows / usable_rows if usable_rows else 0.0,
        "volume_ratio_vs_last_year": rows / rows_last_year if rows_last_year else None,
    }


def concept_drift_inference(triggered: bool, drift: dict[str, Any] | None) -> str:
    """D-008: concept drift is inferred, never measured directly."""
    if not triggered:
        return "no_degradation"
    if drift is not None and drift["n_alert"] > 0:
        return "degradation_with_feature_drift"
    return "suspected_concept_drift"


def same_month_last_year(month: str) -> str:
    return add_months(month, -12)
