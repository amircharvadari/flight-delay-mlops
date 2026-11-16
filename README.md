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
