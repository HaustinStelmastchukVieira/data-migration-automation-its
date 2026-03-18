BOOTSTRAP_PYTHON ?= python3
PYTHON=.venv/bin/python
PIP=.venv/bin/pip

.PHONY: setup run test

setup:
	$(BOOTSTRAP_PYTHON) -m venv .venv
	$(PIP) install -r requirements.txt

run:
	$(PYTHON) -m src.run_all

test:
	$(PYTHON) -m pytest
