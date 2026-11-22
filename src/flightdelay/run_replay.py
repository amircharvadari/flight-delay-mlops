"""Run the E3 replay (M7). Native macOS, after committing:

    caffeinate -i uv run python -m flightdelay.run_replay --out-dir data/replay/E3

Resumable: run the same command again after an interruption; fitted models and
scores are taken from the caches in the output directory. Refuses a dirty
working tree (--allow-dirty only for mechanics checks) and any 2025 month.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from flightdelay.config import PROJECT_ROOT, load_config
from flightdelay.grid import (
    IdentityMismatch,
    atomic_write_json,
    check_or_write_identity,
    code_fingerprint,
    sha256_file,
)
from flightdelay.policy import POLICIES
from flightdelay.replay import REPLAY_CONFIG_FILES, Replay, compare, load_monitoring_config
from flightdelay.serving import load_serving_config
from flightdelay.tracking import git_info
from flightdelay.training import months_between


def identity(months: list[str], initial: list[str], policies: list[str]) -> dict:
    manifest = PROJECT_ROOT / "data" / "manifest.csv"
    plan = json.dumps({"months": months, "initial": initial, "policies": policies})
    return {**{k: v for k, v in git_info().items() if k != "git_changed_files"},
            "code_sha256": code_fingerprint(),
            "config_sha256": {p.name: sha256_file(p) for p in REPLAY_CONFIG_FILES},
            "data_manifest_sha256": sha256_file(manifest) if manifest.exists() else "missing",
            "plan_sha256": hashlib.sha256(plan.encode()).hexdigest()}


def log_to_mlflow(results: dict, experiment: str) -> None:
    import mlflow

    from flightdelay.tracking import setup_mlflow
    setup_mlflow(experiment)
    for policy, r in results.items():
        with mlflow.start_run(run_name=f"E3_{policy}"):
            mlflow.set_tags({**{k: str(v) for k, v in git_info().items()}, "policy": policy})
            mlflow.log_metrics({k: float(v) for k, v in r["summary"].items()
                                if isinstance(v, (int, float)) and not isinstance(v, bool)})
            for i, row in r["monthly"].reset_index(drop=True).iterrows():
                mlflow.log_metrics({"monthly_ap": row["average_precision"],
                                    "monthly_auc": row["roc_auc"],
                                    "monthly_max_psi": row["max_psi"],
                                    "monthly_prevalence": row["prevalence"]}, step=int(i))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/replay/E3")
    ap.add_argument("--policies", nargs="+", default=list(POLICIES), choices=POLICIES)
    ap.add_argument("--processed-dir", default=None)
    ap.add_argument("--first-month", default=None, help="mechanics checks only")
    ap.add_argument("--last-month", default=None, help="mechanics checks only")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--experiment", default="E3_replay")
    args = ap.parse_args(argv)

    cfg, mcfg, scfg = load_config(), load_monitoring_config(), load_serving_config()
    first, last = mcfg["replay"]["months"]
    months = months_between(args.first_month or first, args.last_month or last)
    initial = mcfg["replay"]["initial_train"]
    ident = identity(months, initial, args.policies)
    if ident.get("git_dirty") and not args.allow_dirty:
        print("refusing to run the replay from a dirty working tree (commit first)", file=sys.stderr)
        return 2
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        check_or_write_identity(out, ident)
    except IdentityMismatch as e:
        print(str(e), file=sys.stderr)
        return 3
    proc = Path(args.processed_dir) if args.processed_dir else PROJECT_ROOT / cfg["paths"]["processed_dir"]
    champ = scfg["champion"]
    replay = Replay(proc, out, champ["model"], int(champ["window_months"]), mcfg, base_cfg=cfg)

    t0 = time.perf_counter()
    results = {p: replay.run_policy(p, months, tuple(initial)) for p in args.policies}
    comparison = compare(results)
    atomic_write_json(out / "comparison.json", {
        "summaries": {p: r["summary"] for p, r in results.items()}, "comparison": comparison,
        "fits_this_run": replay.fits_this_run, "wall_seconds": time.perf_counter() - t0})
    if not args.no_mlflow:
        log_to_mlflow(results, args.experiment)
    print(json.dumps({p: r["summary"] for p, r in results.items()}, indent=1))
    print(json.dumps(comparison, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
