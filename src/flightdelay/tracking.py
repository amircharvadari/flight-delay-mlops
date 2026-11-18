"""Experiment tracking with MLflow (D-019, M3).

Local SQLite backend in the project folder (mlflow.db, artifacts in
mlartifacts/), both git-ignored. Both locations can be overridden with the
environment variables MLFLOW_TRACKING_URI and FLIGHTDELAY_ARTIFACT_ROOT (needed
where SQLite cannot lock files, e.g. a folder shared into a virtual machine). Every run records what is needed to trace a
result back to code, data and configuration: git commit (and whether the
working tree was dirty), SHA-256 of the data manifest and of the model config,
the training window, row counts, timing, peak memory and the model itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from flightdelay.config import PROJECT_ROOT

DEFAULT_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "mlartifacts"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifest.csv"
MANIFEST_MISSING = "missing"


def peak_rss_mb() -> float:
    """Peak resident memory of this process. ru_maxrss is bytes on macOS, KB on Linux."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / 1e6 if sys.platform == "darwin" else r / 1e3, 1)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# Paths whose content determines a result. Untracked files here also make the
# run "dirty": a run with new, uncommitted code is not reproducible from the commit.
CODE_PATHS = ("src", "configs", "scripts", "tests", "pyproject.toml", "uv.lock")


def git_info(root: Path | None = None) -> dict[str, Any]:
    root = Path(root or PROJECT_ROOT)

    def git(*args: str) -> str:
        out = subprocess.run(["git", "-c", "safe.directory=*", *args], cwd=root,
                             capture_output=True, text=True, check=False)
        return out.stdout.strip()
    changes = git("status", "--porcelain", "--untracked-files=all", "--", *CODE_PATHS)
    return {"git_commit": git("rev-parse", "HEAD") or "unknown",
            "git_dirty": bool(changes),
            "git_changed_files": len(changes.splitlines()) if changes else 0}


def environment_info() -> dict[str, Any]:
    import lightgbm
    import sklearn
    import xgboost
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "pandas": pd.__version__, "sklearn": sklearn.__version__,
            "lightgbm": lightgbm.__version__, "xgboost": xgboost.__version__}


def setup_mlflow(experiment: str, tracking_uri: str | None = None,
                 artifact_root: Path | None = None) -> None:
    import mlflow
    mlflow.set_tracking_uri(tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI)
    if mlflow.get_experiment_by_name(experiment) is None:
        root = Path(artifact_root or os.environ.get("FLIGHTDELAY_ARTIFACT_ROOT") or DEFAULT_ARTIFACT_ROOT)
        root.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(experiment, artifact_location=(root / experiment).as_uri())
    mlflow.set_experiment(experiment)


class FlightDelayPyfunc:
    """MLflow pyfunc wrapper: input = processed batch rows, output = one
    probability per input row (NaN for rows with unusable features)."""

    def __new__(cls, model):
        import mlflow.pyfunc

        class _Wrapper(mlflow.pyfunc.PythonModel):
            def __init__(self, inner):
                self.model = inner

            def predict(self, context, model_input, params=None):
                return self.model.predict_proba_all(model_input).to_frame()

        return _Wrapper(model)


def log_run(model, run_name: str, metrics: dict[str, float], extra: dict[str, Any] | None = None,
            log_model: bool = True, config_paths: list[Path] | None = None,
            manifest_path: Path | None = None) -> str:
    """Log one trained model and its evaluation to the active MLflow experiment.

    The tag data_manifest_sha256 is always written: the SHA-256 of the data
    manifest, or "missing" if the manifest file does not exist. A missing
    manifest is therefore visible in the run instead of silently absent.
    """
    import mlflow

    meta = {**model.metadata, **(extra or {})}
    tags = {**git_info(), **environment_info(), "model_type": model.model_type}
    manifest = Path(manifest_path or DEFAULT_MANIFEST)
    tags["data_manifest_sha256"] = file_sha256(manifest) if manifest.exists() else MANIFEST_MISSING
    for p in config_paths or []:
        tags[f"config_sha256_{Path(p).stem}"] = file_sha256(p)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({k: str(v) for k, v in tags.items()})
        flat = {k: v for k, v in meta.items() if not isinstance(v, (dict, list))}
        mlflow.log_params({k: str(v)[:250] for k, v in flat.items()})
        mlflow.log_params({f"hp_{k}": str(v) for k, v in model.params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "vocabulary.json").write_text(json.dumps(model.vocab, indent=1))
            (Path(tmp) / "metadata.json").write_text(json.dumps(meta, indent=1, default=str))
            mlflow.log_artifacts(tmp, artifact_path="run_info")
        if log_model:
            mlflow.pyfunc.log_model(name="model", python_model=FlightDelayPyfunc(model),
                                    pip_requirements=["pandas", "scikit-learn", "lightgbm", "xgboost"])
        return run.info.run_id
