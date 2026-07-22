.PHONY: install test lint format check fetch-data train serve dashboard clean

## Setup
install:
	uv sync

## Quality
test:
	uv run pytest

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

check: lint test

## Data
fetch-data:
	uv run python -m gridcast.data.fetch

## Model
train:
	uv run python -m gridcast.data.fetch
	uv run python -m gridcast.features.build
	uv run python -m gridcast.models.train

## Serve
serve:
	uv run uvicorn gridcast.api.app:app --reload

dashboard:
	uv run streamlit run src/gridcast/monitoring/dashboard.py

## Cleanup
clean:
	rm -rf .pytest_cache .ruff_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +
