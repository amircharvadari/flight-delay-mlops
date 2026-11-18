"""Evaluation of one model on one batch (D-016, D-024, D-034).

One function for every model, decision point, replay month and the holdout.

Threshold (D-034, an evaluation policy, not an API threshold):
    pi_train = training-window delay rate (model metadata, training labels only)
    S        = scores of all usable-feature rows of the batch (no labels read)
    k        = max(1, round(pi_train * N)),  tau = k-th largest score in S
    flagged  = score >= tau
Only afterwards are the batch labels used, on labelled rows only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from flightdelay.features import scoring_frame
from flightdelay.validate import valid_label_mask


def alert_threshold(scores: np.ndarray, pi_train: float) -> dict[str, float]:
    """Batch-level alert threshold from unlabelled scores (D-034). No labels involved."""
    s = np.asarray(scores, dtype="float64")
    n = len(s)
    if n == 0:
        raise ValueError("cannot set a threshold on an empty batch")
    if not 0.0 <= pi_train <= 1.0:
        raise ValueError(f"pi_train must be a rate in [0, 1], got {pi_train}")
    k = max(1, round(pi_train * n))
    tau = float(np.sort(s)[::-1][k - 1])
    return {"pi_train": float(pi_train), "scored_rows": n, "k": k, "tau": tau}


def evaluate_batch(model, batch: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Score a batch, apply the D-034 threshold, compute metrics on labelled rows.

    Returns (metrics, predictions). `predictions` has one row per scored
    (usable-feature) row, keyed by row_id = row position in the processed
    Parquet month file, so predictions of different models on the same month
    can be joined row by row for paired analysis.
    """
    X = scoring_frame(batch, model.vocab)
    scores = model.predict_proba_features(X)
    thr = alert_threshold(scores, float(model.metadata["train_positive_rate"]))
    flagged = scores >= thr["tau"]

    labelled = valid_label_mask(batch).loc[X.index].to_numpy()
    y = batch.loc[X.index, "ArrDel15"].to_numpy(dtype="float64")
    y_l, s_l, f_l = y[labelled].astype("int8"), scores[labelled], flagged[labelled]

    tp = int((f_l & (y_l == 1)).sum())
    n_flag_l = int(f_l.sum())
    n_pos = int((y_l == 1).sum())
    precision = tp / n_flag_l if n_flag_l else 0.0
    recall = tp / n_pos if n_pos else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    two_classes = 0 < n_pos < len(y_l)

    metrics = {
        **thr,
        "target_alert_rate": thr["pi_train"],
        "flagged_rows": int(flagged.sum()),
        "realised_alert_rate_scored": float(flagged.mean()),
        "labelled_rows": int(labelled.sum()),
        "flagged_labelled_rows": n_flag_l,
        "realised_alert_rate_labelled": n_flag_l / int(labelled.sum()) if labelled.any() else 0.0,
        "prevalence": float(y_l.mean()) if len(y_l) else float("nan"),
        "average_precision": float(average_precision_score(y_l, s_l)) if two_classes else float("nan"),
        "roc_auc": float(roc_auc_score(y_l, s_l)) if two_classes else float("nan"),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    predictions = pd.DataFrame({
        "row_id": X.index.to_numpy(dtype="int64"),
        "score": scores.astype("float64"),
        "label": pd.array(np.where(labelled, y, np.nan), dtype="Float64").astype("Int8"),
        "labelled": labelled,
        "flagged": flagged,
    })
    return metrics, predictions
