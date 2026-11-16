# counting-laws — `make` wrappers around tasks.py.
#
# The commands live in tasks.py, not here, because the reference machine for this
# artefact is Windows and `make` is not always installed. Every target below is
# one line, so the two entry points cannot drift apart.
#
#   python tasks.py            # list every task with a description
#   make check                 # everything verifiable without a GPU
#   make paper                 # rebuild main.pdf from the stored results
#   make -n <target>           # print the command instead of running it

PY ?= python
TASKS := $(PY) tasks.py

.PHONY: help install install-gpu data run run-fresh status test numbers figs \
        paper trace readme verify check countlaws clean distclean

help:
	@$(TASKS)

install:
	@$(TASKS) install

install-gpu:
	@$(TASKS) install-gpu

data:
	@$(TASKS) data

run:
	@$(TASKS) run

run-fresh:
	@$(TASKS) run-fresh

status:
	@$(TASKS) status

test:
	@$(TASKS) test

numbers:
	@$(TASKS) numbers

figs:
	@$(TASKS) figs

paper:
	@$(TASKS) paper

trace:
	@$(TASKS) trace

readme:
	@$(TASKS) readme

verify:
	@$(TASKS) verify

check:
	@$(TASKS) check

countlaws:
	@$(TASKS) countlaws

clean:
	@$(TASKS) clean

## removes the stored run as well: only if you mean to re-run the suite
distclean: clean
	rm -rf results/run_* RUN_METADATA.json ARTIFACT_CHAIN.json ARTIFACT_VERIFY.json
