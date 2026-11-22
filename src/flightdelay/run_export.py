"""Train, register and export the champion serving bundle (M5).

Run on native macOS (uses the project's MLflow database, D-035/I-21):

    uv run python -m flightdelay.run_export

Steps:
1. Train the champion (configs/serving.yaml: lightgbm, 12-month window, D-037)
   with the unchanged configuration in configs/models.yaml on the development
   window, which is the window of the last E1+E2 grid fit (2023-11 to 2024-10).
2. Score the next month (2024-11): average precision is compared with the grid
   result for the same fit (reproducibility, D-009; tolerance still open, O-008).
3. Freeze the alert threshold for single requests (D-038).
4. Log the run with the model to MLflow, register it as a new version of the
   registered model and point the alias "champion" at it.
5. Download the aliased model from the registry into the serving bundle.

The export refuses a dirty working tree (use --allow-dirty only for mechanics
checks) and any month of the 2025 holdout (D-020).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from flightdelay.config import PROJECT_ROOT, load_config, set_seed
from flightdelay.evaluation import evaluate_batch
from flightdelay.grid import add_months
from flightdelay.models import MODELS_CONFIG, FlightDelayModel, load_model_config
from flightdelay.serving import (
    DEFAULT_BUNDLE_DIR,
    SERVING_CONFIG,
    frozen_threshold,
    load_serving_config,
    write_bundle,
)
from flightdelay.tracking import git_info
from flightdelay.training import guard_holdout, load_training_window, month_path, months_between


def register_champion(run_id: str, name: str, alias: str, tags: dict[str, str]) -> str:
    """Register the run's model as a new version and point the alias at it."""
    import mlflow
    from mlflow import MlflowClient

    mv = mlflow.register_model(f"runs:/{run_id}/model", name)
    client = MlflowClient()
    for k, v in tags.items():
        client.set_model_version_tag(name, mv.version, k, v)
    client.set_registered_model_alias(name, alias, mv.version)
    return str(mv.version)


def download_alias(name: str, alias: str, dst: Path) -> Path:
    import mlflow

    return Path(mlflow.artifacts.download_artifacts(f"models:/{name}@{alias}", dst_path=str(dst)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_BUNDLE_DIR))
    ap.add_argument("--experiment", default="M5_serving")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)

    git = git_info()
    if git.get("git_dirty") and not args.allow_dirty:
        print("refusing to export from a dirty working tree (commit first)", file=sys.stderr)
        return 2

    cfg, mcfg, scfg = load_config(), load_model_config(), load_serving_config()
    champ, dev = scfg["champion"], scfg["development_model"]
    months = months_between(add_months(dev["train_to"], -(int(champ["window_months"]) - 1)),
                            dev["train_to"])
    thr_month = dev["threshold_month"]
    if thr_month != add_months(dev["train_to"], 1):
        raise ValueError("threshold month must be the month after the training window")
    guard_holdout([*months, thr_month], False, cfg)
    set_seed(mcfg["seed"])
    proc = PROJECT_ROOT / cfg["paths"]["processed_dir"]

    X, y, vocab, window = load_training_window(months, proc)
    model = FlightDelayModel(champ["model"], mcfg).fit(X, y, vocab, window)
    del X, y
    metrics, preds = evaluate_batch(model, pd.read_parquet(month_path(proc, thr_month)))
    threshold = frozen_threshold(model, preds["score"].to_numpy(), thr_month)
    ap_diff = metrics["average_precision"] - float(dev["grid_reference_ap"])

    from flightdelay.tracking import log_run, setup_mlflow
    setup_mlflow(args.experiment)
    run_id = log_run(model, f"champion_{champ['model']}_{months[0]}_{months[-1]}",
                     {**metrics, "ap_difference_vs_grid": ap_diff},
                     {"threshold_month": thr_month, "champion_decision": champ["decision"]},
                     config_paths=[PROJECT_ROOT / "configs" / "base.yaml", MODELS_CONFIG,
                                   SERVING_CONFIG])
    version = register_champion(run_id, scfg["registered_model_name"], scfg["champion_alias"], {
        "decision": champ["decision"],
        "acceptance_outcome": champ["acceptance_outcome"],
        "rationale": "operational efficiency under D-035; not demonstrated predictive superiority",
    })
    provenance: dict[str, Any] = {
        "registered_model": scfg["registered_model_name"],
        "model_version": version,
        "alias": scfg["champion_alias"],
        "mlflow_run_id": run_id,
        "git_commit": git.get("git_commit"),
        "git_dirty": git.get("git_dirty"),
        "champion": champ,
        "threshold_month_metrics": {k: metrics[k] for k in (
            "average_precision", "roc_auc", "precision", "recall", "f1", "prevalence",
            "scored_rows", "labelled_rows", "realised_alert_rate_scored")},
        "grid_reference_ap": dev["grid_reference_ap"],
        "ap_difference_vs_grid": ap_diff,
    }
    with tempfile.TemporaryDirectory() as tmp:
        src = download_alias(scfg["registered_model_name"], scfg["champion_alias"], Path(tmp))
        out = write_bundle(model, Path(args.out_dir), threshold, provenance, model_src=src)
    print(json.dumps({"bundle": str(out), "model_version": version, "run_id": run_id,
                      "train_window": [months[0], months[-1]], "threshold": threshold["tau"],
                      "threshold_month_ap": metrics["average_precision"],
                      "ap_difference_vs_grid": ap_diff}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
