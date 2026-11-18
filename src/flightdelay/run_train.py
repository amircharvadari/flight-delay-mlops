"""CLI: train one or all models on a window, optionally evaluate on one later month.

Examples:
    # M3 exit run: four models, one tracked run each
    python -m flightdelay.run_train --model all --train-from 2021-01 --train-to 2021-12 \\
        --eval-month 2022-01 --experiment M3_smoke

    # resource measurement only (used by scripts/benchmark_training.py): no MLflow, no metrics
    python -m flightdelay.run_train --model lightgbm --train-from 2019-01 --train-to 2024-10 \\
        --no-mlflow --json-out result.json

Holdout months (2025) are refused unless --allow-holdout is given (D-020).
No model-selection decision is made here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from flightdelay.config import PROJECT_ROOT, load_config, set_seed
from flightdelay.models import MODEL_TYPES, MODELS_CONFIG, FlightDelayModel, load_model_config
from flightdelay.tracking import peak_rss_mb
from flightdelay.training import guard_holdout, load_training_window, month_path, months_between


def evaluate(model: FlightDelayModel, eval_df) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    X, y = model.evaluation_frame(eval_df)
    p = model.predict_proba_features(X)
    return {"eval_rows": float(len(y)), "eval_prevalence": float(y.mean()),
            "eval_average_precision": float(average_precision_score(y, p)),
            "eval_roc_auc": float(roc_auc_score(y, p))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=[*MODEL_TYPES, "all"])
    ap.add_argument("--train-from", required=True)
    ap.add_argument("--train-to", required=True)
    ap.add_argument("--eval-month", default=None)
    ap.add_argument("--experiment", default="M3_smoke")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--no-log-model", action="store_true")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--allow-holdout", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config()
    mcfg = load_model_config()
    set_seed(mcfg["seed"])
    months = months_between(args.train_from, args.train_to)
    eval_months = [args.eval_month] if args.eval_month else []
    guard_holdout(months + eval_months, args.allow_holdout, cfg)
    if args.eval_month and args.eval_month <= months[-1]:
        raise ValueError("eval month must be later than the training window (D-007)")
    proc = PROJECT_ROOT / cfg["paths"]["processed_dir"]

    t0 = time.perf_counter()
    X, y, vocab, window = load_training_window(months, proc)
    t_load = time.perf_counter() - t0
    eval_df = None
    if args.eval_month:
        import pandas as pd
        eval_df = pd.read_parquet(month_path(proc, args.eval_month))

    results = []
    for mt in (MODEL_TYPES if args.model == "all" else (args.model,)):
        t1 = time.perf_counter()
        model = FlightDelayModel(mt, mcfg).fit(X, y, vocab, window)
        t_fit = time.perf_counter() - t1
        metrics = {"load_seconds": t_load, "fit_seconds": t_fit, "peak_rss_mb": peak_rss_mb(),
                   "train_rows": float(len(X))}
        if eval_df is not None:
            t2 = time.perf_counter()
            metrics.update(evaluate(model, eval_df))
            metrics["eval_seconds"] = time.perf_counter() - t2
            metrics["peak_rss_mb"] = peak_rss_mb()
        run_id = None
        if not args.no_mlflow:
            from flightdelay.tracking import log_run, setup_mlflow
            setup_mlflow(args.experiment)
            name = f"{mt}_{months[0]}_{months[-1]}" + (f"_eval_{args.eval_month}" if args.eval_month else "")
            run_id = log_run(model, name, metrics, {"eval_month": args.eval_month},
                             log_model=not args.no_log_model,
                             config_paths=[PROJECT_ROOT / "configs" / "base.yaml", MODELS_CONFIG])
        row = {"model": mt, **window, **{k: round(v, 4) for k, v in metrics.items()}, "run_id": run_id}
        results.append(row)
        print(json.dumps(row), flush=True)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
