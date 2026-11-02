# One-command reproducibility. Run `make help` to see targets.
# Uses a plain Python venv + pip (no conda).
.PHONY: help setup reproduce quick ci tables test lint clean

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
	@echo "make reproduce  - full run on Malimg + BIG 2015, writing results/ (THE grading command)"
	@echo "make quick      - 1-seed, 5-fold run with a BIG 2015 subset, writing results_quick/"
	@echo "make ci         - synthetic-data sanity run, writing results_ci/"
	@echo "make tables     - regenerate report/generated/ from results/"
	@echo "make test       - run tests"
	@echo "make lint       - byte-compile check (stdlib only)"
	@echo "make clean      - remove generated results"

setup:
	$(SYSPY) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

# The single command a grader runs. Must recreate your reported results.
reproduce:
	$(PY) -m src.train --config config.yaml
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
	rm -rf results/* results_quick results_ci
