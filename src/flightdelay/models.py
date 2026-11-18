"""Model wrapper for the four models of the project (M3, D-026, D-028, D-031).

One class, FlightDelayModel, for every model type:
    historical_rate  non-ML reference: smoothed historical delay rates (D-031)
    logreg           reference: logistic regression with its own preprocessing (D-026)
    lightgbm         candidate (C-002)
    xgboost          candidate (C-002)

The wrapper owns everything needed to score a raw processed batch:
the vocabulary fitted on its training window (D-025), the model-specific
preprocessing, the LightGBM missing-duration guard (O-018) and metadata.
Training, the grid, the replay and the API all use the same object.

Configurations come from configs/models.yaml and are fixed (D-022).
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from flightdelay.config import PROJECT_ROOT
from flightdelay.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    build_features,
    scoring_frame,
    training_frame,
)
from flightdelay.validate import training_row_mask, usable_features_mask

MODEL_TYPES = ("historical_rate", "logreg", "lightgbm", "xgboost")
MODELS_CONFIG = PROJECT_ROOT / "configs" / "models.yaml"


def load_model_config(path: str | Path = MODELS_CONFIG) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_model_input(X: pd.DataFrame) -> None:
    """D-026 guard: categorical features must reach a model as category dtype,
    never as integer codes (which tree models would split as ordered numbers)."""
    if list(X.columns) != FEATURE_COLUMNS:
        raise ValueError("model input must have exactly FEATURE_COLUMNS in order")
    for col in CATEGORICAL_FEATURES:
        if not isinstance(X[col].dtype, pd.CategoricalDtype):
            raise TypeError(f"{col} must be categorical, got {X[col].dtype} (D-026)")


# ---------------------------------------------------------------------------
# Historical-rate baseline (D-031)
# ---------------------------------------------------------------------------

class HistoricalRateModel:
    """Smoothed delay rate per airline x origin x scheduled departure hour.

    p_global  = s / n over the whole training window
    p_airline = (s_a + k_a * p_global)  / (n_a + k_a)
    p_combo   = (s_c + k_c * p_airline) / (n_c + k_c)

    Low counts: a group with few flights is pulled towards its parent rate;
    with n = k both get equal weight. Unseen or missing values: an airline
    not seen in training (or missing after the vocabulary) gets p_global; an
    airline x origin x hour combination not seen in training (or with a
    missing origin) gets its airline rate. No label of the scored batch is used.
    """

    KEYS = ("airline", "origin", "dep_hour")

    def __init__(self, prior_strength_airline: float, prior_strength_combination: float):
        self.k_a = float(prior_strength_airline)
        self.k_c = float(prior_strength_combination)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> HistoricalRateModel:
        frame = pd.DataFrame({
            "airline": X["airline"].astype("object"),
            "origin": X["origin"].astype("object"),
            "dep_hour": X["dep_hour"].astype("int64"),
            "y": y.astype("float64").to_numpy(),
        })
        self.p_global_ = float(frame["y"].mean())
        a = frame.dropna(subset=["airline"]).groupby("airline")["y"].agg(["sum", "count"])
        self.p_airline_ = (a["sum"] + self.k_a * self.p_global_) / (a["count"] + self.k_a)
        c = frame.dropna(subset=["airline", "origin"]).groupby(list(self.KEYS))["y"].agg(["sum", "count"])
        parent = self.p_airline_.reindex(c.index.get_level_values("airline")).to_numpy()
        self.p_combo_ = (c["sum"] + self.k_c * parent) / (c["count"] + self.k_c)
        self.n_combinations_ = len(c)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        airline = X["airline"].astype("object")
        idx = pd.MultiIndex.from_arrays(
            [airline, X["origin"].astype("object"), X["dep_hour"].astype("int64")], names=list(self.KEYS)
        )
        p = self.p_combo_.reindex(idx).to_numpy(dtype="float64")
        p_air = self.p_airline_.reindex(airline).to_numpy(dtype="float64")
        p = np.where(np.isnan(p), p_air, p)
        p = np.where(np.isnan(p), self.p_global_, p)
        return np.column_stack([1.0 - p, p])


# ---------------------------------------------------------------------------
# Logistic regression preprocessing (D-026, D-028)
# ---------------------------------------------------------------------------

LOGREG_NOMINAL = ["airline", "origin", "dest", "month", "day_of_week", "dep_hour", "arr_hour"]
LOGREG_CONTINUOUS = ["distance", "crs_elapsed_time"]


def make_logreg(params: dict[str, Any], seed: int):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import MissingIndicator, SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    pre = ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown="ignore", dtype=np.float32), LOGREG_NOMINAL),
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), LOGREG_CONTINUOUS),
        ("missing", MissingIndicator(features="all"), ["crs_elapsed_time"]),
    ])
    return Pipeline([("pre", pre), ("model", LogisticRegression(random_state=seed, **params))])


# ---------------------------------------------------------------------------
# The wrapper
# ---------------------------------------------------------------------------

class FlightDelayModel:
    def __init__(self, model_type: str, config: dict[str, Any] | None = None):
        if model_type not in MODEL_TYPES:
            raise ValueError(f"unknown model type {model_type!r}; expected one of {MODEL_TYPES}")
        cfg = config or load_model_config()
        self.model_type = model_type
        self.seed = int(cfg["seed"])
        self.params = dict(cfg[model_type])
        self.vocab: dict[str, list[str]] | None = None
        self.estimator: Any = None
        self.metadata: dict[str, Any] = {}

    # -- training -----------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series, vocab: dict[str, list[str]],
            window: dict[str, Any] | None = None) -> FlightDelayModel:
        """Fit on a feature frame built with `vocab` (from training rows only)."""
        check_model_input(X)
        self.vocab = {k: list(v) for k, v in vocab.items()}
        n_missing_duration = int(X["crs_elapsed_time"].isna().sum())
        self.lgbm_duration_fill_: float | None = None
        Xf = X

        if self.model_type == "historical_rate":
            self.estimator = HistoricalRateModel(**self.params).fit(X, y)
        elif self.model_type == "logreg":
            self.estimator = make_logreg(self.params, self.seed)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.estimator.fit(X, y)
            self.metadata["fit_warnings"] = sorted({str(w.message)[:160] for w in caught})
        elif self.model_type == "lightgbm":
            import lightgbm as lgb
            if n_missing_duration == 0:
                # O-018 guard: without missing values in training, LightGBM would
                # treat a missing duration at prediction time like 0 minutes.
                self.lgbm_duration_fill_ = float(X["crs_elapsed_time"].median())
            self.estimator = lgb.LGBMClassifier(**self.params).fit(Xf, y)
        elif self.model_type == "xgboost":
            import xgboost as xgb
            self.estimator = xgb.XGBClassifier(**self.params).fit(Xf, y)

        self.metadata.update({
            "model_type": self.model_type,
            "params": self.params,
            "seed": self.seed,
            "feature_columns": list(FEATURE_COLUMNS),
            "train_rows": len(X),
            "train_positive_rate": float(np.mean(y)),
            "train_missing_duration": n_missing_duration,
            "lgbm_duration_fill": self.lgbm_duration_fill_,
            "vocabulary_sizes": {k: len(v) for k, v in self.vocab.items()},
            "cpu_count": os.cpu_count(),
            **(window or {}),
        })
        return self

    def fit_frame(self, df: pd.DataFrame, window: dict[str, Any] | None = None) -> FlightDelayModel:
        """Convenience: fit directly on a processed batch (small data, tests)."""
        from flightdelay.features import fit_vocabulary
        vocab = fit_vocabulary(df)
        X, y = training_frame(df, vocab)
        return self.fit(X, y, vocab, window)

    # -- prediction ---------------------------------------------------------
    def _require_fitted(self) -> None:
        if self.estimator is None or self.vocab is None:
            raise RuntimeError("model is not fitted")

    def predict_proba_features(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of ArrDel15 = 1 for a feature frame built with self.vocab."""
        self._require_fitted()
        check_model_input(X)
        if X.empty:
            return np.empty(0, dtype="float64")
        if self.model_type == "lightgbm" and self.lgbm_duration_fill_ is not None:
            X = X.assign(crs_elapsed_time=X["crs_elapsed_time"].fillna(self.lgbm_duration_fill_))
        return np.asarray(self.estimator.predict_proba(X)[:, 1], dtype="float64")

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        return build_features(df, self.vocab)

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        """Score a processed batch: one probability per row with usable features.

        Rows without a label are scored (D-024). Quarantined rows are not.
        """
        self._require_fitted()
        X = scoring_frame(df, self.vocab)
        return pd.Series(self.predict_proba_features(X), index=X.index, name="delay_probability")

    def predict_proba_all(self, df: pd.DataFrame) -> pd.Series:
        """Like predict_proba, but one value per input row; NaN for unusable rows."""
        return self.predict_proba(df).reindex(df.index)

    def evaluation_frame(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Labelled rows of a batch for evaluation (training_row_mask, D-024)."""
        self._require_fitted()
        return training_frame(df, self.vocab)


def usable_and_training_counts(df: pd.DataFrame) -> dict[str, int]:
    return {"rows": len(df), "usable_feature_rows": int(usable_features_mask(df).sum()),
            "training_rows": int(training_row_mask(df).sum())}
