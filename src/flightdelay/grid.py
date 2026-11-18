"""E1+E2 grid: specification, crash-safe result store, resume, runner (M4, D-035).

Layout of one grid run directory:
    identity.json             what this run is: git commit, dirty flag, config,
                              manifest and spec hashes. Written once; every resume
                              must match it exactly, otherwise the runner refuses.
    fits/<key>.json           one result per fit, written atomically
    predictions/<key>.parquet test-month predictions, written atomically BEFORE the
                              result json; the json records the Parquet SHA-256
    results.csv               regenerated from fits/*.json (never the source of truth)

A fit counts as done only if its json has status "ok", belongs to the same
identity and its prediction file exists with the recorded hash. Anything else
(crash mid-fit, killed process, partial file) is simply run again.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from flightdelay.config import PROJECT_ROOT, is_holdout_month, load_config
from flightdelay.training import months_between

GRID_CONFIG = PROJECT_ROOT / "configs" / "grid.yaml"
CONFIG_FILES = [PROJECT_ROOT / "configs" / "base.yaml", PROJECT_ROOT / "configs" / "models.yaml", GRID_CONFIG]


def load_grid_config(path: str | Path = GRID_CONFIG) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def add_months(month: str, n: int) -> str:
    y, m = int(month[:4]), int(month[5:7]) - 1 + n
    return f"{y + m // 12}-{m % 12 + 1:02d}"


def decision_points(cfg: dict[str, Any]) -> list[str]:
    dp = cfg["decision_points"]
    out, t = [], dp["first"]
    while t <= dp["last"]:
        out.append(t)
        t = add_months(t, dp["step_months"])
    return out


@dataclass(frozen=True)
class FitSpec:
    decision_point: str
    window: str
    model: str
    train_from: str
    train_to: str
    test_month: str

    @property
    def key(self) -> str:
        return f"{self.decision_point}__w{self.window}__{self.model}"


def window_start(window: str, decision_point: str, expanding_start: str) -> str:
    if window == "expanding":
        return expanding_start
    start = add_months(decision_point, -(int(window) - 1))
    if start < expanding_start:
        raise ValueError(f"{window}-month window before {expanding_start} at {decision_point}")
    return start


def enumerate_fits(cfg: dict[str, Any], points: list[str] | None = None,
                   windows: list[str] | None = None, models: list[str] | None = None) -> list[FitSpec]:
    fits = []
    for t in points or decision_points(cfg):
        for w in windows or cfg["windows"]:
            for m in models or cfg["models"]:
                fits.append(FitSpec(t, w, m, window_start(w, t, cfg["expanding_start"]), t, add_months(t, 1)))
    base = load_config()
    for f in fits:
        for month in [*months_between(f.train_from, f.train_to), f.test_month]:
            if is_holdout_month(month, base):
                raise PermissionError(f"grid fit {f.key} touches holdout month {month} (D-020)")
    return fits


# --- crash-safe file helpers ---------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write to a temporary file in the same folder, fsync, then rename over the target.
    A reader sees either the old file or the complete new one, never a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_bytes(path, json.dumps(obj, indent=1, sort_keys=True, default=str).encode())


def atomic_write_parquet(path: Path, df: pd.DataFrame) -> None:
    import io
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression="zstd")
    atomic_write_bytes(path, buf.getvalue())


# --- run identity ----------------------------------------------------------------

def code_fingerprint(root: Path = PROJECT_ROOT) -> str:
    """SHA-256 over the contents of every code file (tracked and untracked) in the
    code paths. Unlike the commit hash, it changes when uncommitted code changes."""
    from flightdelay.tracking import CODE_PATHS
    out = subprocess.run(["git", "-c", "safe.directory=*", "ls-files", "--cached", "--others",
                          "--exclude-standard", "--", *CODE_PATHS],
                         cwd=root, capture_output=True, text=True, check=True).stdout.split()
    h = hashlib.sha256()
    for rel in sorted(set(out)):
        p = root / rel
        if p.is_file():
            h.update(rel.encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def run_identity(spec_fits: list[FitSpec], cfg: dict[str, Any]) -> dict[str, Any]:
    from flightdelay.tracking import git_info
    manifest = PROJECT_ROOT / "data" / "manifest.csv"
    return {
        **{k: v for k, v in git_info().items() if k != "git_changed_files"},
        "code_sha256": code_fingerprint(),
        "config_sha256": {p.name: sha256_file(p) for p in CONFIG_FILES},
        "data_manifest_sha256": sha256_file(manifest) if manifest.exists() else "missing",
        "fit_keys_sha256": hashlib.sha256("\n".join(f.key for f in spec_fits).encode()).hexdigest(),
        "n_fits": len(spec_fits),
    }


class IdentityMismatch(RuntimeError):
    pass


def check_or_write_identity(out_dir: Path, identity: dict[str, Any]) -> None:
    path = out_dir / "identity.json"
    if not path.exists():
        atomic_write_json(path, {**identity, "created": time.strftime("%Y-%m-%dT%H:%M:%S")})
        return
    stored = json.loads(path.read_text())
    diffs = {k: (stored.get(k), v) for k, v in identity.items() if stored.get(k) != v}
    if diffs:
        raise IdentityMismatch(
            f"{out_dir} belongs to a different code/config/data/grid version; refusing to mix "
            f"results. Differences: {sorted(diffs)}. Use a new output directory.")


# --- result store and resume -------------------------------------------------------

def fit_json_path(out_dir: Path, key: str) -> Path:
    return out_dir / "fits" / f"{key}.json"


def prediction_path(out_dir: Path, key: str) -> Path:
    return out_dir / "predictions" / f"{key}.parquet"


def is_done(out_dir: Path, key: str) -> bool:
    j = fit_json_path(out_dir, key)
    if not j.exists():
        return False
    try:
        r = json.loads(j.read_text())
    except json.JSONDecodeError:
        return False
    p = prediction_path(out_dir, key)
    return r.get("status") == "ok" and p.exists() and sha256_file(p) == r.get("predictions_sha256")


def collect_results(out_dir: Path) -> pd.DataFrame:
    rows = []
    for j in sorted((out_dir / "fits").glob("*.json")):
        try:
            rows.append(json.loads(j.read_text()))
        except json.JSONDecodeError:
            continue
    df = pd.DataFrame(rows)
    if len(df):
        atomic_write_bytes(out_dir / "results.csv", df.sort_values("key").to_csv(index=False).encode())
    return df


# --- runner -----------------------------------------------------------------------

Executor = Callable[[FitSpec, Path, dict[str, Any]], dict[str, Any]]


def subprocess_executor(fit: FitSpec, out_dir: Path, opts: dict[str, Any]) -> dict[str, Any]:
    """Run one fit in its own process (D-035): clean per-fit memory, failure isolation."""
    task = out_dir / "tasks" / f"{fit.key}.json"
    atomic_write_json(task, {**asdict(fit), "key": fit.key, "out_dir": str(out_dir), **opts})
    t0 = time.perf_counter()
    try:
        proc = subprocess.run([sys.executable, "-m", "flightdelay.grid_fit", str(task)],
                              check=False, capture_output=True, text=True,
                              timeout=opts["fit_timeout_seconds"])
        rc = proc.returncode
        err = proc.stderr.strip().splitlines()[-1][:300] if proc.stderr.strip() else ""
    except subprocess.TimeoutExpired:
        rc, err = None, f"timeout after {opts['fit_timeout_seconds']} s"
    return {"returncode": rc, "error": err, "process_wall_seconds": time.perf_counter() - t0}


def run_grid(fits: list[FitSpec], out_dir: Path, identity: dict[str, Any], opts: dict[str, Any],
             executor: Executor = subprocess_executor, max_fits: int | None = None,
             log: Callable[[str], None] = print) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    check_or_write_identity(out_dir, identity)
    pending = [f for f in fits if not is_done(out_dir, f.key)]
    log(f"grid: {len(fits)} fits, {len(fits) - len(pending)} already done, {len(pending)} pending")
    started = 0
    for f in pending:
        if max_fits is not None and started >= max_fits:
            break
        started += 1
        info = executor(f, out_dir, opts)
        if is_done(out_dir, f.key):
            r = json.loads(fit_json_path(out_dir, f.key).read_text())
            log(f"  ok      {f.key}: AP={r['average_precision']:.4f} fit={r['fit_seconds']:.1f}s "
                f"peak={r['peak_rss_mb']:.0f}MB")
        else:
            status = "timeout" if info.get("returncode") is None else (
                "killed (likely out of memory)" if info.get("returncode") in (-9, 137) else "failed")
            atomic_write_json(fit_json_path(out_dir, f.key), {
                **asdict(f), "key": f.key, "status": status, **info, **identity})
            log(f"  {status:7s} {f.key}: {info.get('error', '')}")
    collect_results(out_dir)
    done = sum(is_done(out_dir, f.key) for f in fits)
    return {"total": len(fits), "done": done, "started_now": started, "pending": len(fits) - done}
