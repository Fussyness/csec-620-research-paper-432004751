# One-command reproducibility. Run `make help` to see targets.
# Uses a plain Python venv + pip (no conda).
.PHONY: help setup reproduce reproduce-scratch quick ci tables test lint clean

VENV := .venv
ifeq ($(OS),Windows_NT)
  PY := $(VENV)/Scripts/python
  SYSPY := python
else
  PY := $(VENV)/bin/python
  SYSPY := python3
endif

help:
	@echo "make setup      - create .venv and install requirements"
	@echo "make reproduce  - rebuild results/ and the paper tables; reuses committed per-seed"
	@echo "                  results (seconds, no data needed), computes any that are missing"
	@echo "make reproduce-scratch - recompute everything from the datasets (hours; GPU)"
	@echo "make quick      - 1-seed, 5-fold run with a BIG 2015 subset, writing results_quick/"
	@echo "make ci         - synthetic-data sanity run, writing results_ci/"
	@echo "make tables     - regenerate report/generated/ from results/"
	@echo "make test       - run tests"
	@echo "make lint       - byte-compile check (stdlib only)"
	@echo "make clean      - remove quick/CI results (keeps the committed results/)"

setup:
	$(SYSPY) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

# The single command a grader runs. Must recreate your reported results.
# Resumes: per-seed results already in results/<exp>/runs/ (committed to the repo)
# are reused when they were made with the same settings (config fingerprint), so
# this only re-merges them into metrics.json and the paper tables. Missing seeds are
# computed (needs the datasets). See README "Reproducing".
reproduce:
	$(PY) -m src.train --config config.yaml
	$(PY) scripts/dedup_threshold.py --config config.yaml
	$(PY) -m src.tables --results results

# Recompute every number from the datasets, ignoring existing results.
# About 14 h on one GPU; see scripts/lovegood/ for the 10-GPU parallel version.
reproduce-scratch:
	$(PY) -m src.train --config config.yaml --fresh
	$(PY) scripts/dedup_threshold.py --config config.yaml --fresh
	$(PY) -m src.tables --results results

quick:
	$(PY) -m src.train --config config.quick.yaml

ci:
	$(PY) -m src.train --config config.ci.yaml

tables:
	$(PY) -m src.tables --results results

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m py_compile src/*.py tests/*.py

clean:
	rm -rf results_quick results_ci
