"""Smoke test against a running API (local uvicorn or the Docker container).

    uv run python scripts/api_smoke.py [--url http://127.0.0.1:8000]

Checks /health, /model and /predict with two example flights, and that an
outcome field in the request is rejected (HTTP 422). Uses only the standard
library. The example flights are illustrative inputs, not data records.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

FLIGHTS = [
    {"flight_id": "example-1", "flight_date": "2024-12-20", "reporting_airline": "DL",
     "origin": "ATL", "dest": "LAX", "crs_dep_time": 1730, "crs_arr_time": 1915,
     "crs_elapsed_time": 285, "distance": 1946},
    {"flight_id": "example-2", "flight_date": "2024-12-21", "reporting_airline": "WN",
     "origin": "DAL", "dest": "HOU", "crs_dep_time": 700, "crs_arr_time": 805,
     "crs_elapsed_time": 65, "distance": 239},
]


def call(url: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    base = ap.parse_args().url.rstrip("/")
    checks = []
    code, health = call(f"{base}/health")
    checks.append(("health 200", code == 200))
    code, meta = call(f"{base}/model")
    checks.append(("model 200", code == 200))
    code, pred = call(f"{base}/predict", {"flights": FLIGHTS})
    ok = code == 200 and len(pred.get("predictions", [])) == len(FLIGHTS)
    checks.append(("predict 200, one result per flight", ok))
    leak = {"flights": [{**FLIGHTS[0], "ArrDel15": 1}]}
    checks.append(("outcome field rejected (422)", call(f"{base}/predict", leak)[0] == 422))
    print(json.dumps({"health": health, "model_version": meta.get("model_version"),
                      "threshold": meta.get("threshold", {}).get("tau"),
                      "predictions": pred.get("predictions")}, indent=1))
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
    return 0 if all(p for _, p in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
