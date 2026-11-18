"""M3 exit benchmark: training time and peak memory per model and window strategy.

Resources only: no evaluation month, no metrics, no model selection (D-022).
Windows end at the last E1/E2 decision point (2024-10, D-022):
    12, 24, 36 months rolling, and expanding (2019-01 .. 2024-10 = 70 months).
Each (window, model) pair runs in its own process, so the peak memory of one
run is not inflated by another. A run that is killed or exceeds the time limit
is recorded as such: a measured result, not a silent omission (D-014).

Usage (from the project folder):
    uv run python scripts/benchmark_training.py
    uv run python scripts/benchmark_training.py --windows 12 --models historical_rate   # quick check
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from flightdelay.tracking import git_info
from flightdelay.training import months_between

ROOT = Path(__file__).resolve().parents[1]

MODELS = ["historical_rate", "logreg", "lightgbm", "xgboost"]
WINDOWS = ["12", "24", "36", "expanding"]
FIRST_MONTH = "2019-01"


def total_ram_gb() -> float | None:
    try:
        if sys.platform == "darwin":
            return round(int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 1e9, 1)
        with open("/proc/meminfo") as f:
            return round(int(f.readline().split()[1]) / 1e6, 1)
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def window_start(window: str, decision_point: str) -> str:
    if window == "expanding":
        return FIRST_MONTH
    all_months = months_between(FIRST_MONTH, decision_point)
    n = int(window)
    if n > len(all_months):
        raise ValueError(f"{n}-month window does not fit before {decision_point}")
    return all_months[-n]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decision-point", default="2024-10")
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--windows", nargs="+", default=WINDOWS, choices=WINDOWS)
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    tag = f"{platform.system().lower()}_{platform.machine()}"
    out = Path(args.out or ROOT / "report" / "evidence" / f"benchmark_training_{tag}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    machine = {"platform": platform.platform(), "cpu_count": os.cpu_count(), "ram_gb": total_ram_gb(),
               **git_info()}
    print("machine:", machine, flush=True)

    fields = ["window", "model", "train_from", "train_to", "train_months", "status", "returncode",
              "wall_seconds", "load_seconds", "fit_seconds", "peak_rss_mb", "train_rows",
              "platform", "cpu_count", "ram_gb", "git_commit", "git_dirty"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for window in args.windows:
            start = window_start(window, args.decision_point)
            for model in args.models:
                with tempfile.TemporaryDirectory() as tmp:
                    jpath = Path(tmp) / "r.json"
                    cmd = [sys.executable, "-m", "flightdelay.run_train", "--model", model,
                           "--train-from", start, "--train-to", args.decision_point,
                           "--no-mlflow", "--json-out", str(jpath)]
                    t0 = time.perf_counter()
                    try:
                        proc = subprocess.run(cmd, check=False, cwd=ROOT, capture_output=True, text=True,
                                              timeout=args.timeout)
                        rc = proc.returncode
                        status = "ok" if rc == 0 else (
                            "killed (likely out of memory)" if rc in (-9, 137) else f"failed: {proc.stderr.strip().splitlines()[-1][:150] if proc.stderr.strip() else rc}")
                    except subprocess.TimeoutExpired:
                        rc, status = None, f"timeout after {args.timeout} s"
                    wall = time.perf_counter() - t0
                    res = json.loads(jpath.read_text())[0] if jpath.exists() else {}
                row = {"window": window, "model": model, "train_from": start,
                       "train_to": args.decision_point,
                       "train_months": len(months_between(start, args.decision_point)),
                       "status": status, "returncode": rc, "wall_seconds": round(wall, 1),
                       "load_seconds": res.get("load_seconds"), "fit_seconds": res.get("fit_seconds"),
                       "peak_rss_mb": res.get("peak_rss_mb"), "train_rows": res.get("train_rows"),
                       "platform": machine["platform"], "cpu_count": machine["cpu_count"],
                       "ram_gb": machine["ram_gb"], "git_commit": machine["git_commit"],
                       "git_dirty": machine["git_dirty"]}
                w.writerow(row)
                f.flush()
                print(f"{window:>9} {model:16s} {status:30s} wall={row['wall_seconds']:>7}s "
                      f"fit={row['fit_seconds']}s peak={row['peak_rss_mb']} MB rows={row['train_rows']}",
                      flush=True)
    print("written:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
