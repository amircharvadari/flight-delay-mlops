"""Serving bundle (M5; D-018, D-019, D-037; threshold rule D-038).

A serving bundle is a directory that holds everything the API needs and
nothing else:

    <bundle>/model/         MLflow pyfunc model (FlightDelayPyfunc), copied
                            from the model registry (alias "champion")
    <bundle>/serving.json   metadata: model version, training window,
                            vocabulary hash, alert threshold and its rule,
                            the champion decision. Written LAST, atomically:
                            a bundle without serving.json is incomplete and
                            is refused by load_bundle().

Alert threshold for single requests (D-038). D-034 sets a threshold
per batch from the batch's own scores; a single request has no batch. The
exporter therefore applies the D-034 rule once, to the first month after the
training window (labels are not used), and freezes the resulting tau in the
bundle. The API flags a flight when its probability >= tau. The probability
itself is always returned.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from flightdelay.config import PROJECT_ROOT
from flightdelay.evaluation import alert_threshold
from flightdelay.grid import atomic_write_json

SERVING_CONFIG = PROJECT_ROOT / "configs" / "serving.yaml"
SERVING_DIR_ENV = "FLIGHTDELAY_SERVING_DIR"
DEFAULT_BUNDLE_DIR = PROJECT_ROOT / "artifacts" / "serving" / "champion"
BUNDLE_FORMAT = 1
THRESHOLD_RULE = (
    "D-038: D-034 alert threshold (k = max(1, round(pi_train * N)), tau = k-th largest "
    "score) computed once on the unlabelled scores of the first month after the training "
    "window and frozen; a flight is flagged when its probability >= tau"
)


def load_serving_config(path: str | Path = SERVING_CONFIG) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def vocabulary_sha256(vocab: dict[str, list[str]]) -> str:
    return hashlib.sha256(json.dumps(vocab, sort_keys=True).encode()).hexdigest()


def frozen_threshold(model, scores, month: str) -> dict[str, Any]:
    """D-034 rule applied once to a month's unlabelled scores (D-038).

    Only scores are passed in, so labels of the calibration month cannot be used.
    """
    thr = alert_threshold(scores, float(model.metadata["train_positive_rate"]))
    return {**thr, "calibration_month": month, "labels_used": False,
            "train_from": model.metadata.get("train_from"),
            "train_to": model.metadata.get("train_to"), "rule": THRESHOLD_RULE}


def write_bundle(model, out_dir: Path, threshold: dict[str, Any], provenance: dict[str, Any],
                 model_src: Path | None = None) -> Path:
    """Write a serving bundle. model_src: an existing pyfunc model directory
    (downloaded from the registry); if None, the model is saved from `model`."""
    import mlflow.pyfunc

    from flightdelay.tracking import FlightDelayPyfunc

    out_dir = Path(out_dir)
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    if model_src is not None:
        shutil.copytree(model_src, tmp / "model")
    else:
        mlflow.pyfunc.save_model(str(tmp / "model"), python_model=FlightDelayPyfunc(model),
                                 pip_requirements=["pandas", "scikit-learn", "lightgbm", "xgboost"])
    meta = {
        "bundle_format": BUNDLE_FORMAT,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_type": model.model_type,
        "train_from": model.metadata.get("train_from"),
        "train_to": model.metadata.get("train_to"),
        "train_months": model.metadata.get("train_months"),
        "train_rows": model.metadata.get("train_rows"),
        "train_positive_rate": model.metadata.get("train_positive_rate"),
        "lgbm_duration_fill": model.metadata.get("lgbm_duration_fill"),
        "params": model.params,
        "vocabulary_sizes": {k: len(v) for k, v in model.vocab.items()},
        "vocabulary_sha256": vocabulary_sha256(model.vocab),
        "threshold": threshold,
        **provenance,
    }
    atomic_write_json(tmp / "serving.json", meta)  # last: marks the bundle complete
    old = out_dir.with_name(out_dir.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if out_dir.exists():
        out_dir.rename(old)
    tmp.rename(out_dir)
    shutil.rmtree(old, ignore_errors=True)
    return out_dir


@dataclass
class Bundle:
    model: Any                 # FlightDelayModel
    meta: dict[str, Any]
    path: Path

    @property
    def threshold(self) -> float:
        return float(self.meta["threshold"]["tau"])


def load_bundle(path: str | Path) -> Bundle:
    """Load and check a serving bundle. Raises if it is incomplete or inconsistent."""
    import mlflow.pyfunc

    path = Path(path)
    meta_path = path / "serving.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"no serving.json in {path}: bundle missing or incomplete")
    meta = json.loads(meta_path.read_text())
    if meta.get("bundle_format") != BUNDLE_FORMAT:
        raise ValueError(f"unsupported bundle format {meta.get('bundle_format')!r}")
    model = mlflow.pyfunc.load_model(str(path / "model")).unwrap_python_model().model
    if vocabulary_sha256(model.vocab) != meta["vocabulary_sha256"]:
        raise ValueError("bundle model vocabulary does not match serving.json")
    if model.model_type != meta["model_type"]:
        raise ValueError("bundle model type does not match serving.json")
    return Bundle(model=model, meta=meta, path=path)


def score_frame(model, df: pd.DataFrame) -> pd.Series:
    """Probabilities for rows in processed-batch format (already validated)."""
    X = model.features(df)
    return pd.Series(model.predict_proba_features(X), index=df.index, name="delay_probability")
