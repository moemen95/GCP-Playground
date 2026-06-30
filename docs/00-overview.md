# 00 · Eval Taxonomy & GCP Mapping

Reference card for the **Tangerine Card Benefits Finder** — an ADK Q&A agent over two fictional cards
(`tangerine-money-back`, `tangerine-world`) that retrieves from a benefits/T&C KB and calls 5 lookup tools.
This doc orients the other 8 cards. SUT = System Under Test.

System-under-test surfaces:
- **Offline twin** — `src/card_benefits_finder/local_agent.py` (deterministic, no creds, intentionally imperfect).
- **Live ADK agent** — `src/card_benefits_finder/agent.py` (`root_agent`, Gemini via ADK).
- Both share `tools.py` (`ALL_TOOLS`), `prompts.py` (`SYSTEM_INSTRUCTION`), and the KB (`knowledge/`).

## The 4 eval layers → GCP

| Layer | What it scores | Where in repo | GCP service |
|---|---|---|---|
| **L1 — ADK-native** | Tool trajectory + response match + ADK v2 judges (hallucination/safety/rubric) on `.evalset.json` | `evals/adk/` (`eval_config.json`, `test_config.json`, `run_adk_eval.sh`); datasets in `evals/datasets/` | ADK `adk eval` / `adk web` Eval tab; judges call Vertex Gen AI Eval under the hood |
| **L2 — Vertex Gen AI Eval** | Computation, tool-call, trajectory, model-based pointwise/pairwise, rubric (adaptive) | `evals/vertex/` | **Vertex AI Gen AI evaluation service** (`vertexai.Client` / `google-genai`; legacy `vertexai.evaluation`) |
| **L3 — Custom judges** | G-Eval, RAG faithfulness/context-precision/recall/answer-relevancy, NLI groundedness, safety | `evals/judges/` (backend-pluggable via `evals/common/model_backend.py`) | Any Gemini backend (stub / `gemini` API / `vertex`) |
| **L4 — Pre-prod gating + CI** | Aggregate L1–L3 → hard/soft gate, regression, drift | `evals/gating/` (`thresholds.yaml`), `pipelines/` (`run_all.py`, `ci_gate.py`) | Vertex AI Pipelines (KFP), Agent Engine online eval, Vertex AI Experiments |

All layers emit the uniform `MetricResult` contract (`evals/common/results.py`): `name, layer, family, score(0..1), passed, threshold, uses_llm_judge`. Families: `computation, tool_use, trajectory, pointwise, pairwise, rubric, rag, safety`.

## Local-first vs live-GCP

| Mode | `EVAL_BACKEND` | Creds | SUT | Judges | Command |
|---|---|---|---|---|---|
| **Offline (default)** | `stub` | none | `local_agent.py` | lexical heuristics (`text_utils.py`) | `make eval-local`, `make test-local`, `make gate` |
| Direct Gemini | `gemini` | `GOOGLE_API_KEY` | ADK agent | live Gemini | `EVAL_BACKEND=gemini …` |
| **Live GCP** | `vertex` | project + ADC | ADK agent / Agent Engine | Vertex autoraters | `make eval-vertex`, `make adk-eval`, `make deploy` |

Gating posture (`GATE_MODE`, see `evals/gating/thresholds.yaml`): `full` (judge + deterministic + custom) vs `deterministic` (only computation/tool-call/trajectory — no LLM judge, fully reproducible CI).

## Which metric for which question

| Question about the agent | Family / layer | Headline metric(s) | Card |
|---|---|---|---|
| Did it call the right tool with right args? | tool_use (L1/L2) | `tool_name_match`, `tool_parameter_kv_match`, ADK `tool_trajectory_avg_score` | [03](03-agentic-eval.md) |
| Did it follow the right multi-step path? | trajectory (L1/L2) | `trajectory_in_order_match`, `…precision/recall` | [03](03-agentic-eval.md) |
| Is the answer faithful to the T&C clause? | rag / pointwise | RAGAS `faithfulness`, `groundedness`, ADK `hallucinations_v1` | [04](04-rag-groundedness.md) |
| Did retrieval surface the right clause? | rag | `context_recall`, `context_precision` | [04](04-rag-groundedness.md) |
| Is it correct / relevant / helpful? | pointwise / rubric | `question_answering_correctness`, `RubricMetric.QUESTION_ANSWERING_QUALITY` | [01](01-scorers-catalog.md) |
| Better than the previous prompt? | pairwise | `pairwise_*`, AutoSxS | [02](02-llm-as-judge.md) |
| Is it safe / refuses out-of-scope? | safety | `safety`, ADK `safety_v1`, refusal balance | [05](05-safety-redteam.md) |
| Does it leak PII / fall to injection? | safety / red-team | Presidio PII, indirect-injection probes | [05](05-safety-redteam.md) |
| Will it pass the pre-prod gate? | gating (L4) | tiered conjunctive hard-gates + weighted composite | [06](06-preprod-gating.md) |
| Which GCP SDK/REST call runs it? | — | `evaluateInstances`, `evaluateDataset`, Agent Engine | [07](07-gcp-eval-mapping.md) |
| How do we monitor it in prod? | observability | online eval, tracing, drift | [08](08-market-monitoring.md) |

## Cross-references
[01 Scorers](01-scorers-catalog.md) · [02 LLM-as-judge](02-llm-as-judge.md) · [03 Agentic](03-agentic-eval.md) · [04 RAG](04-rag-groundedness.md) · [05 Safety](05-safety-redteam.md) · [06 Gating](06-preprod-gating.md) · [07 GCP mapping](07-gcp-eval-mapping.md) · [08 Monitoring](08-market-monitoring.md)
