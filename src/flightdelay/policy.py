"""Retraining, promotion and rollback decisions (M6; D-012, D-040 to D-044).

Pure decision logic, no data and no training: the replay engine (M7) scores the
batches, passes the metrics in, and executes the returned actions. This keeps
the time order testable on its own.

Monthly cycle for month t, called after the labels of t are revealed:
 1. Initial model without a baseline: its AUC on t becomes the baseline
    (D-040); no trigger in that month.
 2. Rollback check (D-042): if the previous model was in shadow on t and
    AP(previous) >= AP(serving) + margin, the previous model serves from t+1.
 3. Promotion check (D-041): if a candidate was in shadow on t and
    AP(candidate) >= AP(serving) + margin, the candidate serves from t+1
    (= its decision month + 2) and the old model is watched on t+1.
    Otherwise the candidate is discarded.
 4. Trigger (D-040, D-044): only if no model change was decided in 2.-3. and
    no candidate is in shadow. Policy never_retrained: never; triggered:
    AUC(t) < baseline - delta; scheduled: always. A trigger trains a
    candidate on the 12 months up to and including t (labels <= t only) and
    puts it in shadow on t+1. If t+1 is in the 2025 holdout, nothing is trained.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

from flightdelay.config import is_holdout_month, load_config
from flightdelay.grid import add_months

POLICIES = ("never_retrained", "triggered", "scheduled")


@dataclass(frozen=True)
class ModelRef:
    model_id: str
    train_from: str
    train_to: str
    baseline_auc: float | None = None


@dataclass
class PolicyState:
    serving: ModelRef
    candidate: ModelRef | None = None
    candidate_shadow_month: str | None = None
    previous: ModelRef | None = None
    watch_month: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class Observation:
    """Metrics on the labelled rows of month t (same rows for every model)."""
    month: str
    serving_auc: float
    serving_ap: float
    candidate_auc: float | None = None
    candidate_ap: float | None = None
    previous_ap: float | None = None


def window_for(train_to: str, months: int) -> tuple[str, str]:
    return add_months(train_to, -(months - 1)), train_to


def models_to_score(state: PolicyState, month: str) -> dict[str, ModelRef]:
    """Which models score batch `month` (serving officially, others in shadow)."""
    out = {"serving": state.serving}
    if state.candidate is not None and state.candidate_shadow_month == month:
        out["candidate"] = state.candidate
    if state.previous is not None and state.watch_month == month:
        out["previous"] = state.previous
    return out


def _check_month(month: str, cfg: dict[str, Any]) -> None:
    if is_holdout_month(month, cfg):
        raise ValueError(f"{month} is in the locked 2025 holdout (D-020)")


def step(state: PolicyState, obs: Observation, policy: str, mcfg: dict[str, Any],
         window_months: int = 12, base_cfg: dict[str, Any] | None = None
         ) -> tuple[PolicyState, list[dict[str, Any]]]:
    """Apply the monthly decision for month obs.month. Returns (new state, actions)."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    cfg = base_cfg or load_config()
    t = obs.month
    _check_month(t, cfg)
    s = replace(state, events=list(state.events))
    actions: list[dict[str, Any]] = []

    def log(event: str, **kw: Any) -> None:
        s.events.append({"month": t, "event": event, "serving": s.serving.model_id, **kw})

    changed = False
    # 1. baseline of the initial model
    if s.serving.baseline_auc is None:
        s.serving = replace(s.serving, baseline_auc=obs.serving_auc)
        log("baseline_set", baseline_auc=obs.serving_auc)
        changed = True

    # 2. rollback check
    if s.previous is not None and s.watch_month == t:
        margin = mcfg["rollback"]["margin"]
        if obs.previous_ap is None:
            raise ValueError("previous model was not scored in its watch month")
        if obs.previous_ap >= obs.serving_ap + margin:
            log("rollback", rolled_back=s.serving.model_id, to=s.previous.model_id,
                serving_ap=obs.serving_ap, previous_ap=obs.previous_ap,
                serves_from=add_months(t, 1))
            s.serving, s.previous = s.previous, s.serving
            changed = True
        else:
            log("rollback_check_passed", serving_ap=obs.serving_ap, previous_ap=obs.previous_ap)
        s.watch_month = None

    # 3. promotion check
    if s.candidate is not None and s.candidate_shadow_month == t:
        margin = mcfg["promotion"]["margin"]
        if obs.candidate_ap is None or obs.candidate_auc is None:
            raise ValueError("candidate was not scored in its shadow month")
        if obs.candidate_ap >= obs.serving_ap + margin:
            promoted = replace(s.candidate, baseline_auc=obs.candidate_auc)
            log("promote", candidate=promoted.model_id, serving_ap=obs.serving_ap,
                candidate_ap=obs.candidate_ap, serves_from=add_months(t, 1),
                train_to=promoted.train_to)
            s.previous, s.serving = s.serving, promoted
            s.watch_month = add_months(t, 1)
            changed = True
        else:
            log("discard", candidate=s.candidate.model_id, serving_ap=obs.serving_ap,
                candidate_ap=obs.candidate_ap)
        s.candidate, s.candidate_shadow_month = None, None

    # 4. trigger
    if not changed and s.candidate is None:
        delta = mcfg["trigger"]["delta"]
        degraded = obs.serving_auc < s.serving.baseline_auc - delta
        fire = {"never_retrained": False, "triggered": degraded, "scheduled": True}[policy]
        if degraded:
            log("degradation", serving_auc=obs.serving_auc, baseline_auc=s.serving.baseline_auc,
                delta=delta)
        if fire:
            shadow = add_months(t, 1)
            if is_holdout_month(shadow, cfg):
                log("trigger_skipped_end_of_replay", reason="next month is in the locked holdout")
            else:
                train_from, train_to = window_for(t, window_months)
                cand = ModelRef(model_id=f"{policy}_{train_to}", train_from=train_from,
                                train_to=train_to)
                s.candidate, s.candidate_shadow_month = cand, shadow
                log("train_candidate", candidate=cand.model_id, train_from=train_from,
                    train_to=train_to, shadow_month=shadow, degraded=degraded)
                actions.append({"type": "train_candidate", "model_id": cand.model_id,
                                "train_from": train_from, "train_to": train_to,
                                "shadow_month": shadow})
    return s, actions


def check_time_order(events: list[dict[str, Any]], cfg: dict[str, Any] | None = None) -> None:
    """Raise if any event breaks the D-012 time order or touches the holdout."""
    cfg = cfg or load_config()
    trained: dict[str, dict[str, Any]] = {}
    for e in events:
        # No decision, training window or shadow month may lie in the holdout.
        # serves_from may be 2025-01 for a change decided in 2024-12 (no 2025 data used).
        for key in ("month", "train_from", "train_to", "shadow_month"):
            if key in e and is_holdout_month(e[key], cfg):
                raise AssertionError(f"holdout month in event {e}")
        if e["event"] == "train_candidate":
            if e["train_to"] != e["month"]:
                raise AssertionError(f"candidate uses data after its decision month: {e}")
            if e["shadow_month"] != add_months(e["train_to"], 1):
                raise AssertionError(f"shadow month is not t+1: {e}")
            trained[e["candidate"]] = e
        if e["event"] in ("promote", "discard"):
            tr = trained[e["candidate"]]
            if e["month"] != tr["shadow_month"]:
                raise AssertionError(f"decision not in the shadow month: {e}")
        if e["event"] == "promote" and e["serves_from"] != add_months(trained[e["candidate"]]["train_to"], 2):
            raise AssertionError(f"promoted model does not serve from t+2: {e}")
        if e["event"] == "rollback" and e["serves_from"] != add_months(e["month"], 1):
            raise AssertionError(f"rollback does not take effect next month: {e}")


def is_finite(x: float | None) -> bool:
    return x is not None and math.isfinite(x)
