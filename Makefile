.PHONY: setup benchmark fast test test-all lint clean

PY ?= python

setup:            ## editable install with dev tools (CPU torch)
	$(PY) -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -e ".[dev]"

benchmark:        ## phase 1: 8-model benchmark, feature selection, SHAP (writes results/, reports/figures/)
	churnprop benchmark

fast:             ## same with 3 folds, 100 trees, 10 epochs (~2 min)
	churnprop benchmark --fast

test:             ## unit tests
	$(PY) -m pytest -m "not slow"

test-all:         ## unit + end-to-end
	$(PY) -m pytest

lint:
	ruff check src tests
	ruff format --check src tests

clean:
	rm -rf .pytest_cache models/benchmark
	find . -name __pycache__ -type d -exec rm -rf {} +
