.PHONY: setup lint test m3-exit export-champion serve docker-build docker-run smoke

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

# M5 serving (native macOS). export-champion trains, registers and exports the
# champion bundle into artifacts/serving/champion (D-037, D-038, D-039).
export-champion:
	uv run python -m flightdelay.run_export

serve:
	FLIGHTDELAY_SERVING_DIR=artifacts/serving/champion uv run uvicorn flightdelay.api:app --port 8000

docker-build:
	docker build -t flightdelay-api:0.1.0 .

docker-run:
	docker run --rm -p 8000:8000 -v "$(CURDIR)/artifacts/serving/champion:/app/serving:ro" flightdelay-api:0.1.0

smoke:
	uv run python scripts/api_smoke.py
