PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
export PYTHONPATH := src

.PHONY: install test lint bench up

install:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -r requirements.txt -r requirements-dev.txt

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests bench

bench:
	$(PYTHON) bench/quant_bench.py
	$(PYTHON) bench/serve_bench.py

up:
	docker compose up --build
