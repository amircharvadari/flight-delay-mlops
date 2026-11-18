"""Selection rule for the E1+E2 grid (D-022, D-033, D-035).

Options are (candidate model, window). For each option, over the decision points:
mean and spread of Average Precision, mean wall-clock fit time, mean peak RSS.

Rule, in order (all values fixed in configs/grid.yaml before the grid runs):
1. best = highest mean AP. Eligible: mean AP >= best - ap_margin (0.005).
2. Among eligible options, t_min = smallest mean fit time. Time-tied: mean fit
   time <= t_min * (1 + time_tie_tolerance) (10%).
3. Among time-tied options, m_min = smallest mean peak RSS. Memory-tied:
   mean peak RSS <= m_min * (1 + memory_tie_tolerance) (10%).
4. Among memory-tied options: highest mean AP; final deterministic tie-break:
   alphabetical (model, window).
So the best option wins unless a clearly cheaper one is within the AP margin.
Selection is refused unless every option has a result at every decision point.

Baseline acceptance check (D-036), applied AFTER the candidate is selected and
without changing that selection:
    diff = mean AP of the selected candidate - mean AP of the best reference
           (best of historical_rate and logreg over all four windows)
    diff <  0          -> "candidate_below_reference": no boosted candidate
                          demonstrated an improvement over the reference
    0 <= diff < margin -> "improvement_not_demonstrated": higher, but by less
                          than the 0.005 AP margin (owner-approved)
    diff >= margin     -> "candidate_better"
Only "candidate_better" allows presenting the candidate as superior to the
references.
The margin is a practical improvement threshold, not statistical significance;
the paired statistical analysis is separate. The E3 champion is a separate, documented owner decision (D-036).
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def summarise(results: pd.DataFrame, candidates: list[str]) -> pd.DataFrame:
    ok = results[(results["status"] == "ok") & results["model"].isin(candidates)]
    g = ok.groupby(["model", "window"])
    s = pd.DataFrame({
        "n_points": g["average_precision"].count(),
        "mean_ap": g["average_precision"].mean(),
        "std_ap": g["average_precision"].std(ddof=1),
        "min_ap": g["average_precision"].min(),
        "mean_fit_seconds": g["fit_seconds"].mean(),
        "mean_peak_rss_mb": g["peak_rss_mb"].mean(),
        "mean_roc_auc": g["roc_auc"].mean(),
        "mean_precision": g["precision"].mean(),
        "mean_recall": g["recall"].mean(),
        "mean_f1": g["f1"].mean(),
    }).reset_index()
    best_per_point = ok.loc[ok.groupby("decision_point")["average_precision"].idxmax(), ["model", "window"]]
    wins = best_per_point.value_counts().rename("points_won").reset_index()
    return s.merge(wins, on=["model", "window"], how="left").fillna({"points_won": 0})


def acceptance_check(results: pd.DataFrame, cfg: dict[str, Any], expected_points: list[str],
                     candidate_mean_ap: float) -> dict[str, Any]:
    refs = summarise(results, cfg["references"])
    expected = len(cfg["references"]) * len(cfg["windows"])
    complete = (len(refs) == expected) and bool((refs["n_points"] == len(expected_points)).all())
    if refs.empty:
        return {
            "outcome": "references_missing",
            "statement": "No reference results: the acceptance check could not be performed.",
            "references_complete": False, "best_reference_model": None,
            "best_reference_window": None, "best_reference_mean_ap": None,
            "candidate_minus_reference_ap": None,
            "candidate_may_be_presented_as_superior": False, "reference_table": [],
        }
    best = refs.sort_values(["mean_ap", "model", "window"], ascending=[False, True, True]).iloc[0]
    diff = candidate_mean_ap - float(best["mean_ap"])
    margin = cfg["selection"]["ap_margin"]
    if diff < 0:
        outcome = "candidate_below_reference"
        statement = ("No boosted candidate demonstrated an improvement over the reference: the selected "
                     "candidate's mean AP is below the best reference.")
    elif diff < margin:
        outcome = "improvement_not_demonstrated"
        statement = (f"The selected candidate's mean AP exceeds the best reference by less than the "
                     f"{margin} AP margin; an improvement is not demonstrated.")
    else:
        outcome = "candidate_better"
        statement = (f"The selected candidate exceeds the best reference by at least {margin} mean AP.")
    return {
        "outcome": outcome, "statement": statement, "references_complete": complete,
        "best_reference_model": best["model"], "best_reference_window": best["window"],
        "best_reference_mean_ap": float(best["mean_ap"]), "candidate_minus_reference_ap": diff,
        "candidate_may_be_presented_as_superior": outcome == "candidate_better",
        "reference_table": refs.sort_values("mean_ap", ascending=False).to_dict(orient="records"),
    }


def select(results: pd.DataFrame, cfg: dict[str, Any], expected_points: list[str],
           allow_incomplete: bool = False) -> dict[str, Any]:
    sel = cfg["selection"]
    cands = cfg["candidates"]
    s = summarise(results, cands)
    expected_options = len(cands) * len(cfg["windows"])
    complete = (len(s) == expected_options) and bool((s["n_points"] == len(expected_points)).all())
    if not complete and not allow_incomplete:
        raise ValueError("grid incomplete: every candidate option needs a result at every decision point")

    best = s["mean_ap"].max()
    s["eligible"] = s["mean_ap"] >= best - sel["ap_margin"]
    el = s[s["eligible"]]
    t_min = el["mean_fit_seconds"].min()
    s["time_tied"] = s["eligible"] & (s["mean_fit_seconds"] <= t_min * (1 + sel["time_tie_tolerance"]))
    tt = s[s["time_tied"]]
    m_min = tt["mean_peak_rss_mb"].min()
    s["memory_tied"] = s["time_tied"] & (s["mean_peak_rss_mb"] <= m_min * (1 + sel["memory_tie_tolerance"]))
    final = s[s["memory_tied"]].sort_values(["mean_ap", "model", "window"], ascending=[False, True, True])
    chosen = final.iloc[0]
    s["selected"] = (s["model"] == chosen["model"]) & (s["window"] == chosen["window"])
    acc = acceptance_check(results, cfg, expected_points, float(chosen["mean_ap"]))
    if not acc["references_complete"] and not allow_incomplete:
        raise ValueError("grid incomplete: every reference option needs a result at every decision point")
    complete = complete and acc["references_complete"]
    return {
        "selected_model": chosen["model"], "selected_window": chosen["window"],
        "best_mean_ap": float(best), "selected_mean_ap": float(chosen["mean_ap"]),
        "complete": complete, "is_decision": complete,
        "rule": {k: sel[k] for k in ("ap_margin", "time_tie_tolerance", "memory_tie_tolerance")},
        "acceptance": acc,
        "table": s.sort_values("mean_ap", ascending=False).to_dict(orient="records"),
    }
