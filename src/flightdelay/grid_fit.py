"""One grid fit, run in its own process by grid.subprocess_executor (D-035).

Steps: load the training window -> fit -> evaluate on the test month ->
write predictions (atomic) -> write the result json (atomic, last) ->
optionally log an MLflow run without the model file.
Peak memory is this process's own peak (ru_maxrss), so it belongs to one fit.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import pandas as pd

from flightdelay.config import PROJECT_ROOT, load_config, set_seed
from flightdelay.evaluation import evaluate_batch
from flightdelay.grid import (
    CONFIG_FILES,
    atomic_write_json,
    atomic_write_parquet,
    fit_json_path,
    prediction_path,
    sha256_file,
)
from flightdelay.models import FlightDelayModel, load_model_config
from flightdelay.tracking import peak_rss_mb
from flightdelay.training import load_training_window, month_path, months_between


def main(task_path: str) -> int:
    task = json.loads(Path(task_path).read_text())
    out_dir = Path(task["out_dir"])
    key = task["key"]
    base, mcfg = load_config(), load_model_config()
    set_seed(mcfg["seed"])
    # processed_dir can be given in the task (tests use small synthetic months)
    proc = Path(task.get("processed_dir") or PROJECT_ROOT / base["paths"]["processed_dir"])
    months = months_between(task["train_from"], task["train_to"])

    t0 = time.perf_counter()
    X, y, vocab, window = load_training_window(months, proc)
    load_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    model = FlightDelayModel(task["model"], mcfg).fit(X, y, vocab, window)
    fit_s = time.perf_counter() - t1           # wall-clock training time (D-035)
    del X, y

    test_file = month_path(proc, task["test_month"])
    batch = pd.read_parquet(test_file)
    t2 = time.perf_counter()
    metrics, preds = evaluate_batch(model, batch)
    eval_s = time.perf_counter() - t2

    pred_file = prediction_path(out_dir, key)
    atomic_write_parquet(pred_file, preds)
    result = {
        "key": key, "status": "ok",
        **{k: task[k] for k in ("decision_point", "window", "model", "train_from", "train_to", "test_month")},
        "train_months": len(months), "train_rows": model.metadata["train_rows"],
        "train_missing_duration": model.metadata["train_missing_duration"],
        "lgbm_duration_fill": model.metadata["lgbm_duration_fill"],
        **metrics,
        "load_seconds": load_s, "fit_seconds": fit_s, "eval_seconds": eval_s,
        "peak_rss_mb": peak_rss_mb(),
        "test_parquet_sha256": sha256_file(test_file),
        "predictions_sha256": sha256_file(pred_file),
        **task["identity"],
    }
    if task.get("mlflow", True):
        from flightdelay.tracking import log_run, setup_mlflow
        setup_mlflow(task["experiment"])
        run_metrics = {k: v for k, v in result.items()
                       if isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isnan(v)}
        result["mlflow_run_id"] = log_run(
            model, key, run_metrics,
            {"grid_key": key, "test_month": task["test_month"], "decision_point": task["decision_point"]},
            log_model=False, config_paths=CONFIG_FILES)
    atomic_write_json(fit_json_path(out_dir, key), result)   # last: marks the fit as done
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
