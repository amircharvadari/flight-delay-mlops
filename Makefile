.PHONY: setup lint test m3-exit

setup:
	uv sync

lint:
	uv run ruff check .

test:
	uv run pytest -q

# M3 exit (run on native macOS after committing M3):
# 1) one tracked MLflow run per model on the fixed smoke window
# 2) training time and peak memory for 4 models x 4 window strategies
m3-exit:
	uv run python -m flightdelay.run_train --model all --train-from 2021-01 --train-to 2021-12 --eval-month 2022-01 --experiment M3_smoke
	uv run python scripts/benchmark_training.py
