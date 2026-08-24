SHELL := /bin/sh
PYTHON ?= python3
ROOT := $(CURDIR)/..
export PYTHONPATH := $(CURDIR)/src

ifneq (,$(wildcard $(ROOT)/.env))
include $(ROOT)/.env
export
endif

.PHONY: setup test fetch build smoke run paper clean

setup:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install -q -e .

test:
	PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v

fetch FEED=trimet:
	.venv/bin/python -m gtfsplan.pipeline fetch --feed $(FEED)

build FEED=trimet:
	.venv/bin/python -m gtfsplan.pipeline build --feed $(FEED)

gold FEED=trimet:
	.venv/bin/python -m gtfsplan.pipeline gold --feed $(FEED)

smoke:
	.venv/bin/python -m gtfsplan.pipeline smoke

run:
	.venv/bin/python -m gtfsplan.pipeline run

paper:
	.venv/bin/python -m gtfsplan.pipeline paper

clean:
	.venv/bin/python -m gtfsplan.pipeline clean
