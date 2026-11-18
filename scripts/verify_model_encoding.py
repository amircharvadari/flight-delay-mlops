"""Verify how LightGBM, XGBoost and logistic regression treat the M2 categorical features.

Also verifies how the three models handle missing crs_elapsed_time values
(D-028), including a model trained on a window with no missing durations.

Mechanics only: no model-quality metrics are computed or reported, and no
holdout month (2025) is read. Real data used: training rows of 2019-07 (fit)
and the 2021-07 batch (contains airline QX, unseen in the 2019 vocabulary).

Usage:
    python scripts/verify_model_encoding.py
"""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from flightdelay.config import PROJECT_ROOT
from flightdelay.features import fit_vocabulary, scoring_frame, training_frame

PROC = PROJECT_ROOT / "data" / "processed"


def stump_test() -> None:
    """A pattern no single ordinal split can learn: positive for codes 1 and 3 only.

    One tree with one split: a nominal (categorical) split can isolate {B, D};
    an ordinal split on the integer codes cannot (accuracy at most 0.75).
    """
    cats = ["A", "B", "C", "D"]
    x = pd.Series(pd.Categorical(np.tile(cats, 250), categories=cats), name="c")
    y = np.isin(x.astype(str), ["B", "D"]).astype(int)
    Xc = x.to_frame()
    Xi = x.cat.codes.astype("int8").to_frame()

    def acc(model, X):
        return float((model.predict(X) == y).mean())

    m = lgb.LGBMClassifier(n_estimators=1, num_leaves=2, min_child_samples=1,
                           min_data_per_group=1, cat_smooth=0, verbose=-1).fit(Xc, y)
    print(f"LightGBM  category dtype, one split: accuracy {acc(m, Xc):.2f}")
    m = lgb.LGBMClassifier(n_estimators=1, num_leaves=2, min_child_samples=1, verbose=-1).fit(Xi, y)
    print(f"LightGBM  integer codes,  one split: accuracy {acc(m, Xi):.2f}  (ordinal)")
    m = xgb.XGBClassifier(n_estimators=1, max_depth=1, learning_rate=1.0, enable_categorical=True,
                          tree_method="hist", max_cat_to_onehot=1).fit(Xc, y)
    print(f"XGBoost   category dtype, one split: accuracy {acc(m, Xc):.2f}")
    m = xgb.XGBClassifier(n_estimators=1, max_depth=1, learning_rate=1.0, tree_method="hist").fit(Xi, y)
    print(f"XGBoost   integer codes,  one split: accuracy {acc(m, Xi):.2f}  (ordinal)")
    # Behaviour changed across XGBoost versions: older versions raised an error for
    # category dtype without enable_categorical. Check what this version does.
    try:
        m = xgb.XGBClassifier(n_estimators=1, max_depth=1, learning_rate=1.0, tree_method="hist",
                              max_cat_to_onehot=1).fit(Xc, y)
        tdf = m.get_booster().trees_to_dataframe()
        nominal = "Category" in tdf and bool(tdf["Category"].notna().any())
        print(f"XGBoost   category dtype WITHOUT enable_categorical: accepted, one split accuracy "
              f"{acc(m, Xc):.2f}, set-based split: {nominal}, "
              f"enable_categorical param = {m.get_params().get('enable_categorical')}")
    except ValueError as e:
        print("XGBoost   category dtype WITHOUT enable_categorical: ValueError ->", str(e)[:90])


def real_data_test() -> None:
    fit_df = pd.read_parquet(PROC / "2019_07.parquet")
    vocab = fit_vocabulary(fit_df)
    X, y = training_frame(fit_df, vocab)
    X, y = X.sample(200_000, random_state=0), y.loc[X.sample(200_000, random_state=0).index]
    later = pd.read_parquet(PROC / "2021_07.parquet")
    Xs = scoring_frame(later, vocab)
    print(f"\nreal data: fit rows {len(X)}, scoring rows {len(Xs)}, "
          f"unseen airline rows in scoring batch {int(Xs['airline'].isna().sum())}")

    m = lgb.LGBMClassifier(n_estimators=20, verbose=-1, random_state=0).fit(X, y)
    splits = m.booster_.trees_to_dataframe()
    cat_splits = splits[splits["decision_type"] == "=="]["split_feature"].unique().tolist()
    print("LightGBM  categorical features recorded:", m.booster_.pandas_categorical is not None,
          "| features split with '==' (nominal):", sorted(cat_splits))
    p = m.predict_proba(Xs)[:, 1]
    print("LightGBM  predicts batch with unseen categories:", bool(np.isfinite(p).all()), len(p))

    m = xgb.XGBClassifier(n_estimators=20, tree_method="hist", enable_categorical=True,
                          random_state=0).fit(X, y)
    tdf = m.get_booster().trees_to_dataframe()
    xcat = tdf[tdf["Category"].notna()]["Feature"].unique().tolist() if "Category" in tdf else []
    print("XGBoost   features with categorical (set-based) splits:", sorted(xcat))
    p = m.predict_proba(Xs)[:, 1]
    print("XGBoost   predicts batch with unseen categories:", bool(np.isfinite(p).all()), len(p))

    try:
        LogisticRegression(max_iter=50).fit(X, y)
        print("LogReg    raw feature frame: accepted (would be WRONG)")
    except (ValueError, TypeError) as e:
        print("LogReg    raw feature frame: rejected ->", type(e).__name__, str(e)[:70])

    nominal = ["airline", "origin", "dest", "month", "day_of_week", "dep_hour", "arr_hour"]
    continuous = ["distance", "crs_elapsed_time"]
    pre = ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=None), nominal),
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), continuous),
    ])
    lr = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=300))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr.fit(X, y)
    n_out = lr.named_steps["pre"].transform(X.head(5)).shape[1]
    p = lr.predict_proba(Xs)[:, 1]
    print(f"LogReg    with one-hot pipeline: fitted, {n_out} input columns after encoding, "
          f"predicts unseen categories: {bool(np.isfinite(p).all())}")


def logreg_pipeline() -> Pipeline:
    """Reference preprocessing for logistic regression (D-026, D-028)."""
    nominal = ["airline", "origin", "dest", "month", "day_of_week", "dep_hour", "arr_hour"]
    continuous = ["distance", "crs_elapsed_time"]
    pre = ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown="ignore"), nominal),
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), continuous),
        # features="all": the indicator column exists even if the training window
        # has no missing durations, so the model input width never changes.
        ("missing", MissingIndicator(features="all"), ["crs_elapsed_time"]),
    ])
    return Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=300))])


def missing_duration_test() -> bool:
    fit_df = pd.read_parquet(PROC / "2019_07.parquet")
    vocab = fit_vocabulary(fit_df)
    X, y = training_frame(fit_df, vocab)
    idx = X.sample(100_000, random_state=1).index
    X, y = X.loc[idx], y.loc[idx]
    clean = X[X["crs_elapsed_time"].notna()]
    yc = y.loc[clean.index]
    probe = clean.head(2000).copy()
    probe_nan = probe.assign(crs_elapsed_time=np.nan)
    probe_zero = probe.assign(crs_elapsed_time=0.0)
    ok = True
    print(f"\nmissing durations: training rows without any missing duration: {len(clean)}")

    m = lgb.LGBMClassifier(n_estimators=30, verbose=-1, random_state=0).fit(clean, yc)
    pn, pz = m.predict_proba(probe_nan)[:, 1], m.predict_proba(probe_zero)[:, 1]
    same = bool(np.allclose(pn, pz))
    print(f"LightGBM  trained WITHOUT missing durations: NaN predicted like 0 minutes: {same}")
    Xm = clean.copy()
    Xm.loc[Xm.sample(frac=0.01, random_state=2).index, "crs_elapsed_time"] = np.nan
    m = lgb.LGBMClassifier(n_estimators=30, verbose=-1, random_state=0).fit(Xm, yc)
    pn, pz = m.predict_proba(probe_nan)[:, 1], m.predict_proba(probe_zero)[:, 1]
    print(f"LightGBM  trained WITH 1% missing durations:  NaN predicted like 0 minutes: "
          f"{bool(np.allclose(pn, pz))}, finite: {bool(np.isfinite(pn).all())}")
    ok &= bool(np.isfinite(pn).all())

    for label, train in [("WITHOUT", clean), ("WITH 1%", Xm)]:
        m = xgb.XGBClassifier(n_estimators=30, tree_method="hist", enable_categorical=True,
                              random_state=0).fit(train, yc)
        pn, pz = m.predict_proba(probe_nan)[:, 1], m.predict_proba(probe_zero)[:, 1]
        print(f"XGBoost   trained {label} missing durations: NaN predicted like 0 minutes: "
              f"{bool(np.allclose(pn, pz))}, finite: {bool(np.isfinite(pn).all())}")
        ok &= bool(np.isfinite(pn).all())

    for label, train in [("WITHOUT", clean), ("WITH 1%", Xm)]:
        lr = logreg_pipeline()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr.fit(train, yc)
        width_fit = lr.named_steps["pre"].transform(train.head(3)).shape[1]
        width_nan = lr.named_steps["pre"].transform(probe_nan.head(3)).shape[1]
        ind = lr.named_steps["pre"].transform(probe_nan.head(3))[:, -1]
        ind = np.asarray(ind.todense()).ravel() if hasattr(ind, "todense") else ind
        pn = lr.predict_proba(probe_nan)[:, 1]
        print(f"LogReg    trained {label} missing durations: same input width {width_fit == width_nan}, "
              f"missing indicator set: {bool((ind == 1).all())}, finite: {bool(np.isfinite(pn).all())}")
        ok &= width_fit == width_nan and bool((ind == 1).all()) and bool(np.isfinite(pn).all())
    return ok


if __name__ == "__main__":
    print("versions: lightgbm", lgb.__version__, "| xgboost", xgb.__version__)
    stump_test()
    real_data_test()
    passed = missing_duration_test()
    print("\nmissing-duration handling checks passed:", passed)
    raise SystemExit(0 if passed else 1)
