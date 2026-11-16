"""Smoke test: every core library imports and can do one tiny operation.

This catches installation problems early (for example missing OpenMP on macOS).
"""

import numpy as np


def test_boosting_libraries_fit_a_tiny_model():
    import lightgbm as lgb
    import xgboost as xgb

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] > 0).astype(int)
    lgb.LGBMClassifier(n_estimators=5, verbose=-1).fit(X, y)
    xgb.XGBClassifier(n_estimators=5).fit(X, y)


def test_other_core_libraries_import():
    import fastapi  # noqa: F401
    import mlflow  # noqa: F401
    import pandera  # noqa: F401
    import pyarrow  # noqa: F401
    import sklearn  # noqa: F401
