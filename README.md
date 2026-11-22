# Flight delay MLOps pipeline

Final-year capstone, AIN4312, Bahcesehir University.
End-to-end MLOps pipeline for predicting whether a U.S. domestic flight arrives 15 or more minutes late (`ArrDel15`), using the BTS Reporting Carrier On-Time Performance data, 2019-2025.

Design decisions: `PROJECT_DECISIONS.md`. Architecture and plan: `ARCHITECTURE_AND_ROADMAP.md`.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync          # creates .venv from uv.lock
make test        # runs the test suite
```

## Data

Raw files are not stored in git. Download the 84 monthly zip files (2019-01 to 2025-12) from BTS TranStats into `data/raw/bts/`.

## Training (M3)

```bash
# one tracked MLflow run per model (fixed smoke window, no selection)
uv run python -m flightdelay.run_train --model all --train-from 2021-01 --train-to 2021-12 \
    --eval-month 2022-01 --experiment M3_smoke

# training time and peak memory, 4 models x 4 window strategies
uv run python scripts/benchmark_training.py

# both at once
make m3-exit

# browse runs
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Model configurations are fixed in `configs/models.yaml` (D-022, D-031). Holdout months (2025) are refused by the training CLI.

## Serving (M5)

Champion: LightGBM, 12-month window (D-037). Chosen for operational efficiency; the baseline acceptance check found no demonstrated improvement over logistic regression (D-036: `improvement_not_demonstrated`).

```bash
make export-champion   # train, register (MLflow alias "champion"), export artifacts/serving/champion
make serve             # local API on http://127.0.0.1:8000 (docs: /docs)
make smoke             # health, model, predict, outcome-field rejection

make docker-build      # image with code and locked dependencies, no model inside
make docker-run        # mounts artifacts/serving/champion read-only at /app/serving
make smoke
```

Request: `POST /predict` with `{"flights": [{"flight_date": "2024-12-20", "reporting_airline": "DL", "origin": "ATL", "dest": "LAX", "crs_dep_time": 1730, "crs_arr_time": 1915, "crs_elapsed_time": 285, "distance": 1946}]}`. Only pre-departure fields are accepted; any other field is rejected with HTTP 422.

## Monitoring, retraining and rollback (M6)

Thresholds are fixed in `configs/monitoring.yaml` (D-040 to D-044):

- retraining trigger: AUC of the serving model < its baseline AUC - 0.03 (AUC does not depend on the delay rate, so seasonal prevalence changes do not trigger retraining);
- promotion: candidate AP >= serving AP + 0.005 on its shadow month (same rows);
- rollback: previous model AP >= new model AP + 0.005 in the first month after a promotion;
- signals only: feature drift (PSI against the same calendar month in the model's training window), prevalence vs the same month a year earlier, volume and unscorable rows.

```bash
uv run python -m flightdelay.registry status            # aliases champion / previous / challenger
uv run python -m flightdelay.registry rollback          # champion <-> previous, rebuild the bundle
uv run python -m flightdelay.registry promote --version N
```

## E3 replay (M7)

Replays 2020-01 to 2024-12 month by month for three policies (D-044): never retrained, retraining on the AUC trigger, and scheduled monthly retraining, each with shadow evaluation, promotion and rollback (D-012, D-040 to D-042). The 2025 holdout is never touched.

```bash
caffeinate -i uv run python -m flightdelay.run_replay --out-dir data/replay/E3   # resumable
```

Outputs in `data/replay/E3/policies/<policy>/`: `monthly.csv` (performance, frozen-threshold alerts, drift, prevalence and volume signals, events), `events.json`, `summary.json`; cross-policy comparison in `comparison.json`; one MLflow run per policy (experiment `E3_replay`).
