SHELL := /bin/sh
PYTHON ?= python3
ROOT := $(CURDIR)/..
export PYTHONPATH := $(CURDIR)/src

ifneq (,$(wildcard $(ROOT)/.env))
include $(ROOT)/.env
export
endif

# Tunables (defaults shown)
FEED ?= trimet
WALK ?= 300          # metres; inter-stop walking edges for gold generation
N ?=                 # per-feed cap on evaluated gold items (empty = all)
WORKERS ?= 12        # concurrent API calls
ARMS ?= closed_book,schedule_excerpt

.PHONY: setup test fetch build gold smoke run paper clean

setup:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install -q -e .

test:
	PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v

fetch:
	.venv/bin/python -m gtfsplan.pipeline fetch --feed $(FEED)

build:
	.venv/bin/python -m gtfsplan.pipeline build --feed $(FEED)

gold:
	.venv/bin/python -m gtfsplan.pipeline gold --feed $(FEED) --max-walk-m $(WALK)

smoke:
	.venv/bin/python -m gtfsplan.pipeline smoke

run:
	.venv/bin/python -m gtfsplan.pipeline run \
		--feeds trimet,cta,hsl,mta --arms $(ARMS) \
		$(if $(N),--n-per-feed $(N),) --workers $(WORKERS)

paper:
	.venv/bin/python -m gtfsplan.pipeline paper

clean:
	.venv/bin/python -m gtfsplan.pipeline clean
