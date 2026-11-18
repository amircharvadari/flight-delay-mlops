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
