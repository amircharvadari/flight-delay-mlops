"""Run the E3 replay (M7). Native macOS, after committing:

    caffeinate -i uv run python -m flightdelay.run_replay --out-dir data/replay/E3

Resumable: run the same command again after an interruption; fitted models and
scores are taken from the caches in the output directory. Refuses a dirty
working tree (--allow-dirty only for mechanics checks) and any 2025 month.

E4 final evaluation (D-045), once, after the design freeze:

    caffeinate -i uv run python -m flightdelay.run_replay --e4

Gates: clean tree, HEAD == the commit of tag e4-freeze, the E4 output directory
not yet completed. Months, initial window and policies come from
configs/e4.yaml; they cannot be overridden. Every start is appended to
e4_starts.jsonl in the output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
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

E4_CONFIG = PROJECT_ROOT / "configs" / "e4.yaml"


def freeze_commit(tag: str) -> str | None:
    """Commit the freeze tag points at, or None if the tag does not exist."""
    r = subprocess.run(["git", "-c", "safe.directory=*", "rev-parse", "--verify", "--quiet",
                        f"{tag}^{{commit}}"], cwd=PROJECT_ROOT, capture_output=True, text=True,
                       check=False)
    return r.stdout.strip() or None


def e4_gate(ident: dict, out: Path, tag: str) -> str | None:
    """Reason to refuse the E4 run, or None if it may run (D-045)."""
    if ident.get("git_dirty"):
        return "E4 refused: working tree is dirty"
    frozen = freeze_commit(tag)
    if frozen is None:
        return f"E4 refused: freeze tag {tag!r} does not exist"
    if frozen != ident.get("git_commit"):
        return f"E4 refused: HEAD {ident.get('git_commit')} is not the frozen commit {frozen}"
    if (out / "comparison.json").exists():
        return f"E4 refused: {out} already holds a completed E4 run (run once, D-045)"
    return None


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
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--e4", action="store_true", help="final evaluation on 2025 (D-045)")
    args = ap.parse_args(argv)
    if args.e4:
        return main_e4(args)
    args.experiment = args.experiment or "E3_replay"

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


def main_e4(args: argparse.Namespace) -> int:
    import yaml

    if args.first_month or args.last_month or args.allow_dirty:
        print("E4 refused: months and --allow-dirty cannot be overridden", file=sys.stderr)
        return 2
    e4 = yaml.safe_load(E4_CONFIG.read_text())
    cfg, mcfg, scfg = load_config(), load_monitoring_config(), load_serving_config()
    months = months_between(*e4["months"])
    initial, policies = e4["initial_train"], e4["policies"]
    out = Path(args.out_dir if args.out_dir != "data/replay/E3" else "data/replay/E4")
    ident = {**identity(months, initial, policies), "e4_config_sha256": sha256_file(E4_CONFIG),
             "freeze_tag": e4["freeze_tag"]}
    if (reason := e4_gate(ident, out, e4["freeze_tag"])) is not None:
        print(reason, file=sys.stderr)
        return 4
    out.mkdir(parents=True, exist_ok=True)
    try:
        check_or_write_identity(out, ident)
    except IdentityMismatch as e:
        print(str(e), file=sys.stderr)
        return 3
    with open(out / "e4_starts.jsonl", "a") as f:
        f.write(json.dumps({"started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "commit": ident["git_commit"], "tag": e4["freeze_tag"]}) + "\n")
    proc = Path(args.processed_dir) if args.processed_dir else PROJECT_ROOT / cfg["paths"]["processed_dir"]
    champ = scfg["champion"]
    replay = Replay(proc, out, champ["model"], int(champ["window_months"]), mcfg, base_cfg=cfg,
                    unlock_holdout=True)
    t0 = time.perf_counter()
    results = {p: replay.run_policy(p, months, tuple(initial)) for p in policies}
    comparison = compare(results)
    atomic_write_json(out / "comparison.json", {
        "experiment": "E4", "hypotheses": e4["hypotheses"],
        "summaries": {p: r["summary"] for p, r in results.items()}, "comparison": comparison,
        "fits_this_run": replay.fits_this_run, "wall_seconds": time.perf_counter() - t0})
    if not args.no_mlflow:
        log_to_mlflow(results, args.experiment or "E4_final")
    print(json.dumps({p: r["summary"] for p, r in results.items()}, indent=1))
    print(json.dumps(comparison, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
