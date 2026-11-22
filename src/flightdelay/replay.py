"""E3 replay engine (M7; D-004, D-012, D-017, D-020, D-037, D-040 to D-044).

Replays the months of configs/monitoring.yaml (2020-01 to 2024-12) for the
three policies of D-044. For every month t, in time order:
  1. the models chosen by policy.models_to_score() score batch t
     (serving officially; candidate or previous model in shadow mode);
  2. the labels of t are revealed and every scoring model is evaluated on the
     same labelled rows (AP, ROC AUC, and P/R/F1 at the D-034 batch threshold);
  3. monitoring signals are computed (D-043);
  4. policy.step() decides; a candidate is trained at once on the 12 months up
     to and including t (data <= t only).
The serving model's frozen alert threshold (D-038) is the D-034 threshold of
its calibration month, the month after its training window, and its realised
alert rate, precision and recall with that frozen threshold are reported too.

Caches (resume after an interruption, shared by the three policies):
  models/<key>.pkl + .json   fitted model of a training window + fit statistics
  scores/<key>/<month>.parquet + .json   row scores (float32) and metrics
  months/<month>.json + labels/<month>.parquet   policy-independent data signals
A re-run starts each policy from the beginning and takes everything it can
from the caches, so the decisions are recomputed and must come out the same.
identity.json pins code, configs and data; a resume with any change is refused.
No model is trained or scored on a month of the 2025 holdout.
"""
from __future__ import annotations

import gc
import json
import pickle
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from flightdelay import monitoring as mon
from flightdelay.config import PROJECT_ROOT, is_holdout_month, load_config
from flightdelay.evaluation import evaluate_batch
from flightdelay.features import scoring_frame
from flightdelay.grid import add_months, atomic_write_bytes, atomic_write_json, atomic_write_parquet
from flightdelay.models import FlightDelayModel, load_model_config
from flightdelay.policy import (
    ModelRef,
    Observation,
    PolicyState,
    check_time_order,
    models_to_score,
    step,
)
from flightdelay.tracking import peak_rss_mb
from flightdelay.training import load_training_window, month_path, months_between
from flightdelay.validate import training_row_mask, usable_features_mask

MONITORING_CONFIG = PROJECT_ROOT / "configs" / "monitoring.yaml"
REPLAY_CONFIG_FILES = [PROJECT_ROOT / "configs" / n
                       for n in ("base.yaml", "models.yaml", "serving.yaml", "monitoring.yaml")]
METRIC_KEYS = ("average_precision", "roc_auc", "precision", "recall", "f1", "tau", "k",
               "scored_rows", "labelled_rows", "prevalence", "flagged_rows")


def load_monitoring_config(path: str | Path = MONITORING_CONFIG) -> dict[str, Any]:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


class Replay:
    def __init__(self, processed_dir: Path, out_dir: Path, model_type: str = "lightgbm",
                 window_months: int = 12, mon_cfg: dict | None = None,
                 model_cfg: dict | None = None, base_cfg: dict | None = None,
                 log=print):
        self.proc, self.out = Path(processed_dir), Path(out_dir)
        self.model_type, self.window = model_type, window_months
        self.mcfg = mon_cfg or load_monitoring_config()
        self.model_cfg = model_cfg or load_model_config()
        self.cfg = base_cfg or load_config()
        self.log = log
        self._models: OrderedDict[str, FlightDelayModel] = OrderedDict()
        self._batches: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self.fits_this_run = 0

    # -- guards and keys -------------------------------------------------------------
    def _guard(self, *months: str) -> None:
        bad = [m for m in months if is_holdout_month(m, self.cfg)]
        if bad:
            raise ValueError(f"locked 2025 holdout months requested: {bad} (D-020)")

    def key(self, ref: ModelRef) -> str:
        return f"{self.model_type}_{ref.train_from}_{ref.train_to}"

    # -- data ------------------------------------------------------------------------
    def batch(self, month: str) -> pd.DataFrame:
        self._guard(month)
        if month not in self._batches:
            self._batches[month] = pd.read_parquet(month_path(self.proc, month))
            while len(self._batches) > 3:
                self._batches.popitem(last=False)
        return self._batches[month]

    def month_info(self, month: str) -> dict[str, Any]:
        """Policy-independent signals of a month (D-043), cached."""
        path = self.out / "months" / f"{month}.json"
        if (info := _json(path)) is not None:
            return info
        df = self.batch(month)
        usable, lab = usable_features_mask(df), training_row_mask(df)
        prev = float(df.loc[lab, "ArrDel15"].mean()) if lab.any() else float("nan")
        last = mon.same_month_last_year(month)
        prev_last, rows_last = None, None
        if month_path(self.proc, last).exists() and not is_holdout_month(last, self.cfg):
            d = pd.read_parquet(month_path(self.proc, last))
            lab_l = training_row_mask(d)
            prev_last, rows_last = float(d.loc[lab_l, "ArrDel15"].mean()), len(d)
        info = {"month": month,
                **mon.prevalence_signal(prev, prev_last, self.mcfg),
                **mon.volume_signal(len(df), int(usable.sum()), int(lab.sum()), rows_last)}
        labels = pd.DataFrame({"row_id": np.flatnonzero(usable.to_numpy()).astype("int64")})
        y = df.loc[usable, "ArrDel15"].to_numpy(dtype="float64")
        labels["label"] = pd.array(np.where(training_row_mask(df)[usable].to_numpy(), y, np.nan),
                                   dtype="Float64").astype("Int8")
        atomic_write_parquet(self.out / "labels" / f"{month}.parquet", labels)
        atomic_write_json(path, info)
        return info

    # -- models ----------------------------------------------------------------------
    def model(self, ref: ModelRef) -> FlightDelayModel:
        key = self.key(ref)
        if key in self._models:
            self._models.move_to_end(key)
            return self._models[key]
        pkl = self.out / "models" / f"{key}.pkl"
        if pkl.exists():
            m = pickle.loads(pkl.read_bytes())
        else:
            months = months_between(ref.train_from, ref.train_to)
            self._guard(*months)
            if len(months) != self.window:
                raise ValueError(f"window {ref.train_from}..{ref.train_to} is not {self.window} months")
            t0 = time.perf_counter()
            X, y, vocab, window = load_training_window(months, self.proc)
            t_load = time.perf_counter() - t0
            t1 = time.perf_counter()
            m = FlightDelayModel(self.model_type, self.model_cfg).fit(X, y, vocab, window)
            t_fit = time.perf_counter() - t1
            del X, y
            gc.collect()
            atomic_write_bytes(pkl, pickle.dumps(m, protocol=5))
            atomic_write_json(pkl.with_suffix(".json"), {
                "key": key, "train_from": ref.train_from, "train_to": ref.train_to,
                "train_rows": m.metadata["train_rows"], "fit_seconds": t_fit,
                "load_seconds": t_load, "peak_rss_mb": peak_rss_mb()})
            self.fits_this_run += 1
            self.log(f"  fitted {key}: {t_fit:.0f} s")
        self._models[key] = m
        while len(self._models) > 4:
            self._models.popitem(last=False)
        return m

    def fit_seconds(self, ref: ModelRef) -> float:
        info = _json(self.out / "models" / f"{self.key(ref)}.json")
        return float(info["fit_seconds"]) if info else 0.0

    # -- scoring ---------------------------------------------------------------------
    def score(self, ref: ModelRef, month: str) -> dict[str, Any]:
        """Metrics and monitoring of one model on one month (cached)."""
        if ref.train_to >= month:
            raise AssertionError(f"{self.key(ref)} would score {month}, inside its training data")
        key = self.key(ref)
        mpath = self.out / "scores" / key / f"{month}.json"
        if (res := _json(mpath)) is not None:
            return res
        model, df = self.model(ref), self.batch(month)
        metrics, preds = evaluate_batch(model, df)
        window = months_between(ref.train_from, ref.train_to)
        ref_month = mon.reference_month(window, month)
        drift = mon.feature_drift_signal(scoring_frame(self.batch(ref_month), model.vocab),
                                         scoring_frame(df, model.vocab), model.vocab, self.mcfg)
        res = {"model": key, "month": month, **{k: metrics[k] for k in METRIC_KEYS},
               "pi_train": metrics["pi_train"], "reference_month": ref_month,
               "max_psi": drift["max_psi"], "max_psi_feature": drift["max_psi_feature"],
               "n_psi_alert": drift["n_alert"], "n_psi_warning": drift["n_warning"],
               "psi": drift["psi"],
               "unseen_category_share": mon.unseen_category_share(df[usable_features_mask(df)],
                                                                  model.vocab)}
        atomic_write_parquet(mpath.with_suffix(".parquet"), pd.DataFrame({
            "row_id": preds["row_id"], "score": preds["score"].astype("float32")}))
        atomic_write_json(mpath, res)
        return res

    def frozen_stats(self, ref: ModelRef, month: str) -> dict[str, Any]:
        """Realised alerts with the serving model's frozen threshold (D-038)."""
        cal = add_months(ref.train_to, 1)
        tau = self.score(ref, cal)["tau"]
        key = self.key(ref)
        s = pd.read_parquet(self.out / "scores" / key / f"{month}.parquet")
        lab = pd.read_parquet(self.out / "labels" / f"{month}.parquet")
        j = s.merge(lab, on="row_id", how="left")
        flagged = j["score"].to_numpy(dtype="float64") >= tau
        has = j["label"].notna().to_numpy()
        y = j["label"].to_numpy(dtype="float64", na_value=np.nan)[has] == 1
        f = flagged[has]
        tp = int((f & y).sum())
        return {"frozen_tau": tau, "calibration_month": cal,
                "frozen_alert_rate": float(flagged.mean()) if len(flagged) else 0.0,
                "frozen_precision": tp / int(f.sum()) if f.sum() else 0.0,
                "frozen_recall": tp / int(y.sum()) if y.sum() else 0.0}

    # -- policy run ------------------------------------------------------------------
    def run_policy(self, policy: str, months: list[str],
                   initial: tuple[str, str]) -> dict[str, Any]:
        self._guard(*months, *months_between(*initial))
        state = PolicyState(serving=ModelRef(f"{policy}_initial", *initial))
        rows: list[dict[str, Any]] = []
        refs: dict[str, ModelRef] = {self.key(state.serving): state.serving}
        for m in months:
            info = self.month_info(m)
            roles = {r: self.score(ref, m) for r, ref in models_to_score(state, m).items()}
            srv = roles["serving"]
            obs = Observation(m, srv["roc_auc"], srv["average_precision"],
                              candidate_auc=roles.get("candidate", {}).get("roc_auc"),
                              candidate_ap=roles.get("candidate", {}).get("average_precision"),
                              previous_ap=roles.get("previous", {}).get("average_precision"))
            serving_ref = state.serving
            n_events = len(state.events)
            state, actions = step(state, obs, policy, self.mcfg, self.window, self.cfg)
            for a in actions:                                   # train at decision time t
                cref = ModelRef(a["model_id"], a["train_from"], a["train_to"])
                self.model(cref)
                refs[self.key(cref)] = cref
            new_events = state.events[n_events:]
            degraded = any(e["event"] == "degradation" for e in new_events)
            rows.append({
                "policy": policy, "month": m, "serving_model": self.key(serving_ref),
                "serving_train_to": serving_ref.train_to,
                "model_age_months": len(months_between(serving_ref.train_to, m)) - 1,
                "baseline_auc": (state.serving.baseline_auc
                                 if state.serving.model_id == serving_ref.model_id
                                 else serving_ref.baseline_auc),
                **{k: srv[k] for k in ("average_precision", "roc_auc", "precision", "recall", "f1",
                                       "prevalence", "scored_rows", "labelled_rows")},
                **self.frozen_stats(serving_ref, m),
                "prevalence_last_year": info["prevalence_last_year"],
                "prevalence_alert": info["alert"],
                "volume_ratio_vs_last_year": info["volume_ratio_vs_last_year"],
                "unscorable_share": info["unscorable_share"],
                "labelled_share_of_scored": info["labelled_share_of_scored"],
                "max_psi": srv["max_psi"], "max_psi_feature": srv["max_psi_feature"],
                "n_psi_alert": srv["n_psi_alert"],
                "unseen_category_share": srv["unseen_category_share"],
                "degradation": degraded,
                "concept_inference": mon.concept_drift_inference(
                    degraded, {"n_alert": srv["n_psi_alert"]}),
                "shadow_candidate_ap": roles.get("candidate", {}).get("average_precision"),
                "shadow_previous_ap": roles.get("previous", {}).get("average_precision"),
                "events": ";".join(e["event"] for e in new_events),
            })
            self.log(f"{policy} {m}: AP {srv['average_precision']:.4f} AUC {srv['roc_auc']:.4f} "
                     f"[{rows[-1]['events'] or '-'}]")
        check_time_order(state.events, self.cfg)
        trained = [e for e in state.events if e["event"] == "train_candidate"]
        fit_s = self.fit_seconds(ModelRef("", *initial)) + sum(
            self.fit_seconds(ModelRef("", e["train_from"], e["train_to"])) for e in trained)
        monthly = pd.DataFrame(rows)
        summary = {
            "policy": policy, "months": len(months), "first_month": months[0],
            "last_month": months[-1],
            "mean_ap": float(monthly["average_precision"].mean()),
            "median_ap": float(monthly["average_precision"].median()),
            "mean_auc": float(monthly["roc_auc"].mean()),
            "mean_f1": float(monthly["f1"].mean()),
            "mean_frozen_alert_rate": float(monthly["frozen_alert_rate"].mean()),
            "n_trained": len(trained),
            "n_promoted": sum(e["event"] == "promote" for e in state.events),
            "n_discarded": sum(e["event"] == "discard" for e in state.events),
            "n_rollbacks": sum(e["event"] == "rollback" for e in state.events),
            "n_degradation_months": int(monthly["degradation"].sum()),
            "total_fit_seconds": fit_s,
            "mean_model_age_months": float(monthly["model_age_months"].mean()),
        }
        pdir = self.out / "policies" / policy
        atomic_write_parquet(pdir / "monthly.parquet", monthly)
        atomic_write_bytes(pdir / "monthly.csv", monthly.to_csv(index=False).encode())
        atomic_write_json(pdir / "events.json", state.events)
        atomic_write_json(pdir / "summary.json", summary)
        return {"summary": summary, "monthly": monthly, "events": state.events}


def compare(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Month-by-month AP differences between policies (same months, same rows)."""
    ap = pd.DataFrame({p: r["monthly"].set_index("month")["average_precision"]
                       for p, r in results.items()})
    out = {}
    pairs = [("triggered", "never_retrained"), ("scheduled", "never_retrained"),
             ("scheduled", "triggered")]
    for a, b in pairs:
        if a in ap and b in ap:
            d = ap[a] - ap[b]
            out[f"{a}_minus_{b}"] = {"mean_ap_difference": float(d.mean()),
                                     "months_better": int((d > 0).sum()),
                                     "months_worse": int((d < 0).sum()),
                                     "months_equal": int((d == 0).sum())}
    return out
