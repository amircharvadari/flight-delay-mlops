.PHONY: setup lint test

setup:
	uv sync

lint:
	uv run ruff check .

test:
	uv run pytest -q
