# Card-Benefits Eval Playground — common commands.
# All `*-local` targets run fully offline with the deterministic stub backend.

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: help venv install install-all test test-local gate eval-local eval-vertex \
        adk-eval adk-web synth deploy clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

venv: ## Create the project virtualenv
	python3 -m venv .venv && $(PIP) install --upgrade pip

install: ## Install base (offline) dependencies + package
	$(PIP) install -e ".[dev]"

install-all: ## Install everything incl. live-GCP deps (adk, vertex, pipelines, ...)
	$(PIP) install -e ".[all]"

test: test-local ## Alias for test-local

test-local: ## Run the full offline pytest suite (stub backend, zero creds)
	EVAL_BACKEND=stub $(PY) -m pytest

eval-local: ## Run all 4 eval layers offline and emit a unified gate report
	EVAL_BACKEND=stub $(PY) pipelines/run_all.py

gate: ## Run the CI gate (offline) -> nonzero exit on failure
	EVAL_BACKEND=stub $(PY) pipelines/ci_gate.py

eval-vertex: ## Run Layer-2 Vertex Gen AI Eval (requires GCP creds)
	EVAL_BACKEND=vertex $(PY) pipelines/run_all.py --layers vertex

adk-eval: ## Run Layer-1 ADK eval via the `adk eval` CLI (requires google-adk)
	bash evals/adk/run_adk_eval.sh

adk-web: ## Launch the ADK web UI (chat + Eval tab) (requires google-adk)
	adk web src

synth: ## Generate a synthetic eval set from the knowledge base
	$(PY) -m evals.gating.synth_data --out evals/datasets/synthetic_qa.jsonl --n 40

deploy: ## Deploy the agent to Vertex Agent Engine (requires GCP creds)
	bash deploy/deploy.sh

clean: ## Remove caches and generated reports
	rm -rf .pytest_cache **/__pycache__ reports/*.json reports/*.md
