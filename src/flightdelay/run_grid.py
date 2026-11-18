"""CLI: run (or resume) the E1+E2 grid, and select from its results.

    # full grid (requires a clean git tree)
    python -m flightdelay.run_grid run --out-dir data/grid/E1E2
    # mini grid for mechanics checks
    python -m flightdelay.run_grid run --out-dir data/grid/mini --decision-points 2022-01 2022-04 \\
        --windows 12 --experiment M4_mini_grid --allow-dirty
    # selection (refuses an incomplete grid unless --allow-incomplete, which is never a decision)
    python -m flightdelay.run_grid select --out-dir data/grid/E1E2

Interrupting is safe: run the same command again and it resumes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from flightdelay.config import PROJECT_ROOT
from flightdelay.grid import (
    collect_results,
    decision_points,
    enumerate_fits,
    load_grid_config,
    run_grid,
    run_identity,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--decision-points", nargs="+")
    r.add_argument("--windows", nargs="+")
    r.add_argument("--models", nargs="+")
    r.add_argument("--experiment")
    r.add_argument("--max-fits", type=int)
    r.add_argument("--no-mlflow", action="store_true")
    r.add_argument("--allow-dirty", action="store_true")
    s = sub.add_parser("select")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--decision-points", nargs="+")
    s.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_grid_config()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    if args.cmd == "run":
        fits = enumerate_fits(cfg, args.decision_points, args.windows, args.models)
        identity = run_identity(fits, cfg)
        if identity["git_dirty"] and not args.allow_dirty:
            print("refused: uncommitted code changes. Commit first, or pass --allow-dirty "
                  "(results are then marked dirty and are not decision evidence).")
            return 2
        opts = {"experiment": args.experiment or cfg["experiment"], "mlflow": not args.no_mlflow,
                "fit_timeout_seconds": cfg["fit_timeout_seconds"], "identity": identity}
        summary = run_grid(fits, out_dir, identity, opts, max_fits=args.max_fits)
        print(json.dumps(summary))
        return 0

    from flightdelay.selection import select
    results = collect_results(out_dir)
    points = args.decision_points or decision_points(cfg)
    res = select(results, cfg, points, allow_incomplete=args.allow_incomplete)
    (out_dir / "selection.json").write_text(json.dumps(res, indent=1, default=str))
    acc = {k: v for k, v in res["acceptance"].items() if k != "reference_table"}
    print(json.dumps({**{k: v for k, v in res.items() if k not in ("table", "acceptance")},
                      "acceptance": acc}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
