# Card-Benefits Eval Playground

A **comprehensive, code-heavy reference** for evaluating pre-production **agentic
deployments** on Google Cloud's **Gemini Enterprise Agent Platform** (formerly
*Generative AI on Vertex AI*), using a **Google ADK** agent as the
System-Under-Test.

The agent is a **Q&A "benefits finder"** for two *fictional* Tangerine-style
cards (Money-Back & World Mastercard). Around it, this repo implements **every
evaluation framework you can exercise on the Agent Platform** — plus the
custom LLM-judge / RAG / safety techniques and the CI/CD gating that a Principal
AI Engineer needs to ship agents safely.

> **Local-first.** Everything runs offline with a deterministic stub backend and
> **zero GCP credentials** (`make test`). Flip `EVAL_BACKEND` to `gemini`/`vertex`
> for live runs. Heavy/cloud deps live in `pip` extras.

---

## The four evaluation layers

| Layer | What | Where | GCP service |
| --- | --- | --- | --- |
| **1. ADK-native** | EvalSets, `tool_trajectory_avg_score`, `final_response_match_v2`, `hallucinations_v1`, `safety_v1`, rubric & custom metrics, `adk eval`/pytest | `evals/adk/` | ADK + Vertex (judge metrics) |
| **2. Vertex Gen AI Eval** | computation, tool-call, trajectory, pointwise/pairwise autoraters, AutoSxS, adaptive rubrics, custom autorater | `evals/vertex/` | Vertex AI Gen AI Evaluation Service |
| **3. Custom judges / RAG / safety** | G-Eval, PoLL jury, bias mitigation, calibration (κ), RAG faithfulness/precision/recall, NLI grounding, PII/injection/refusal | `evals/judges/` | model-agnostic |
| **4. Pre-prod gating / CI** | deterministic scorers, tiered conjunctive gate, synthetic data, baseline no-regression, report | `evals/gating/`, `pipelines/` | Vertex Pipelines / batch / online |

Reference docs (lean, table-heavy) live in [`docs/`](docs/00-overview.md).

---

## Quickstart (offline, no credentials)

```bash
make venv            # create .venv
make install         # base + dev deps (no google-adk / aiplatform needed)
make test            # full pytest suite — agent, judges, gate (stub backend)
make eval-local      # run ALL offline layers + write reports/gate_report.md
make gate            # CI gate -> exit 0/1
```

Example gate output (offline, `full` mode):

```
Gate mode: full  |  result: PASS
Soft composite: 0.80 / 0.70
Report: reports/gate_report.md
```

## Going live on GCP

```bash
pip install -e ".[all]"                 # adk + vertex + pipelines + ...
cp .env.example .env                    # set project/location, EVAL_BACKEND=vertex
gcloud auth application-default login

adk web src                             # chat + capture goldens (Eval tab)
make adk-eval                           # Layer 1: adk eval CLI
EVAL_BACKEND=vertex python pipelines/run_all.py --layers vertex   # Layer 2
make deploy                             # deploy to Agent Engine, then re-validate
```

GCP eval **pipelines** are first-class: `pipelines/vertex_pipeline.py` (Vertex AI
Pipelines / KFP + AutoSxS), `pipelines/batch_evaluate_dataset.py`
(`evaluateDataset`), `pipelines/online_eval_agent_engine.py` (live-traffic eval).

---

## Repository map

```
src/card_benefits_finder/   # the ADK agent (SUT) + offline twin + tools + KB
evals/common/               # backend abstraction, MetricResult, text utils, loaders
evals/datasets/             # golden eval sets (rag_qa, safety_redteam, pairwise, ADK *.test/.evalset)
evals/adk/                  # Layer 1
evals/vertex/               # Layer 2
evals/judges/               # Layer 3
evals/gating/               # Layer 4 gate + thresholds.yaml (posture toggle)
pipelines/                  # run_all, ci_gate, + 3 GCP pipelines
deploy/                     # Agent Engine deploy
docs/                       # 9 lean reference cards
.github/workflows/          # offline eval gate CI
```

## Gating posture (bank-grade)

`evals/gating/thresholds.yaml` (or `GATE_MODE`):

- **`full`** *(default)* — GCP/custom LLM-judge metrics + deterministic + safety.
- **`deterministic`** — only no-LLM metrics (computation/tool/trajectory + lexical
  grounding/safety); LLM-judge metrics are dropped. For strict data-governance.

The gate is **tiered**: conjunctive **hard gates** (PII=0 leak, 0 injection
breach, 0 harmful compliance, grounding/faithfulness floors, trajectory floor)
that block on any failure, plus a **weighted soft composite** with per-metric
floors and an optional **baseline no-regression** check.

## The agent under test

`Agent("card_benefits_finder")` with five deterministic tools over a synthetic
KB: `list_cards`, `lookup_card_benefits`, `get_benefit_details`,
`find_cards_for_category`, `check_eligibility`. The instruction enforces
**grounding to tool output** and **out-of-scope refusal** — the behaviours the
eval layers measure. The offline twin (`local_agent.py`) reproduces this
behaviour without an LLM so the whole pipeline is reproducible offline.

> ⚠️ Cards, benefit terms, and the knowledge base are **fictional** — not real
> Tangerine products. Do not use for real financial decisions.
