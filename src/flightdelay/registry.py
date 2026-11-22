"""Model registry operations for serving (M6; D-019, D-039, D-042).

Aliases of the registered model:
    champion    the serving model
    challenger  a candidate in shadow mode
    previous    the model before the last promotion (rollback target)

The frozen alert threshold of each version (D-038) is stored as model-version
tags, so a serving bundle can be rebuilt for any version from the registry
alone; promotion and rollback move aliases and rebuild the bundle, the Docker
image stays the same (D-039).

CLI (native macOS, project MLflow database):
    uv run python -m flightdelay.registry status
    uv run python -m flightdelay.registry rollback
    uv run python -m flightdelay.registry promote --version N
    uv run python -m flightdelay.registry export
    uv run python -m flightdelay.registry tag-from-bundle   # store a bundle's threshold as tags
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from flightdelay.serving import DEFAULT_BUNDLE_DIR, load_serving_config, write_bundle

CHAMPION, CHALLENGER, PREVIOUS = "champion", "challenger", "previous"
THRESHOLD_TAG_KEYS = ("tau", "k", "scored_rows", "pi_train", "calibration_month", "train_from",
                      "train_to", "labels_used", "rule")


def _client():
    from mlflow import MlflowClient
    return MlflowClient()


def aliases(name: str) -> dict[str, str]:
    rm = _client().get_registered_model(name)
    return {a: str(v) for a, v in (rm.aliases or {}).items()}


def threshold_tags(threshold: dict[str, Any]) -> dict[str, str]:
    return {f"threshold_{k}": json.dumps(threshold[k]) for k in THRESHOLD_TAG_KEYS}


def threshold_from_tags(tags: dict[str, str]) -> dict[str, Any]:
    missing = [k for k in THRESHOLD_TAG_KEYS if f"threshold_{k}" not in tags]
    if missing:
        raise KeyError(f"model version has no frozen threshold tags: {missing}")
    return {k: json.loads(tags[f"threshold_{k}"]) for k in THRESHOLD_TAG_KEYS}


def register_version(run_id: str, name: str, tags: dict[str, str]) -> str:
    import mlflow
    mv = mlflow.register_model(f"runs:/{run_id}/model", name)
    client = _client()
    for k, v in tags.items():
        client.set_model_version_tag(name, mv.version, k, v)
    return str(mv.version)


def promote(name: str, version: str) -> dict[str, str]:
    """champion -> previous, version -> champion, challenger removed if it was this version."""
    client, cur = _client(), aliases(name)
    old = cur.get(CHAMPION)
    if old == str(version):
        return aliases(name)
    if old is not None:
        client.set_registered_model_alias(name, PREVIOUS, old)
    client.set_registered_model_alias(name, CHAMPION, str(version))
    if cur.get(CHALLENGER) == str(version):
        client.delete_registered_model_alias(name, CHALLENGER)
    return aliases(name)


def rollback(name: str) -> dict[str, str]:
    """Swap champion and previous (D-042). The rolled-back version stays as previous."""
    client, cur = _client(), aliases(name)
    if PREVIOUS not in cur:
        raise RuntimeError("no previous version to roll back to")
    client.set_registered_model_alias(name, CHAMPION, cur[PREVIOUS])
    client.set_registered_model_alias(name, PREVIOUS, cur[CHAMPION])
    return aliases(name)


def export_bundle(name: str, out_dir: Path, event: str) -> Path:
    """Rebuild the serving bundle for the current champion from the registry."""
    import mlflow

    client = _client()
    mv = client.get_model_version_by_alias(name, CHAMPION)
    threshold = threshold_from_tags(mv.tags)
    run = client.get_run(mv.run_id)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(mlflow.artifacts.download_artifacts(f"models:/{name}@{CHAMPION}", dst_path=tmp))
        model = mlflow.pyfunc.load_model(str(src)).unwrap_python_model().model
        provenance = {
            "registered_model": name, "model_version": str(mv.version), "alias": CHAMPION,
            "mlflow_run_id": mv.run_id, "git_commit": run.data.tags.get("git_commit"),
            "git_dirty": run.data.tags.get("git_dirty"), "bundle_event": event,
            "registry_aliases": aliases(name),
            "champion": {k: mv.tags.get(k) for k in ("decision", "acceptance_outcome", "rationale")},
        }
        return write_bundle(model, out_dir, threshold, provenance, model_src=src)


def _tag_from_bundle(name: str, bundle: Path) -> str:
    meta = json.loads((bundle / "serving.json").read_text())
    if meta.get("registered_model") != name:
        raise ValueError("bundle belongs to another registered model")
    client = _client()
    for k, v in threshold_tags(meta["threshold"]).items():
        client.set_model_version_tag(name, meta["model_version"], k, v)
    return str(meta["model_version"])


def main(argv: list[str] | None = None) -> int:
    from flightdelay.tracking import setup_mlflow

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    for c in ("rollback", "export"):
        sub.add_parser(c).add_argument("--out-dir", default=str(DEFAULT_BUNDLE_DIR))
    p = sub.add_parser("promote")
    p.add_argument("--version", required=True)
    p.add_argument("--out-dir", default=str(DEFAULT_BUNDLE_DIR))
    t = sub.add_parser("tag-from-bundle")
    t.add_argument("--bundle", default=str(DEFAULT_BUNDLE_DIR))
    args = ap.parse_args(argv)

    setup_mlflow("M5_serving")  # sets the tracking URI; registry lives in the same database
    name = load_serving_config()["registered_model_name"]
    if args.cmd == "status":
        print(json.dumps(aliases(name), indent=1))
    elif args.cmd == "tag-from-bundle":
        print(f"threshold tags written to version {_tag_from_bundle(name, Path(args.bundle))}")
    else:
        if args.cmd == "rollback":
            rollback(name)
        elif args.cmd == "promote":
            promote(name, args.version)
        out = export_bundle(name, Path(args.out_dir), event=args.cmd)
        print(json.dumps({"aliases": aliases(name), "bundle": str(out)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
