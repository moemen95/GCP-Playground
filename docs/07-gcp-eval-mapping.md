# 07 · GCP Eval Mapping

How the repo's eval layers run on Google Cloud. **Rebrand note:** "Generative AI on Vertex AI" is now marketed as the **Gemini Enterprise Agent Platform**; the underlying **Gen AI evaluation service** is the same API. Env: `EVAL_BACKEND=vertex`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, `STAGING_BUCKET`, `VERTEX_EXPERIMENT`.

## Two SDK surfaces

| Surface | Import | Status | Key objects | Use |
|---|---|---|---|---|
| **GenAI Client (new)** | `vertexai.Client` / `google-genai` | **Recommended** | `client.evals.run_inference()`, `.evaluate()`, `.generate_rubrics()`; `vertexai.types.{RubricMetric, Metric, PrebuiltMetric, LLMMetric}` | All new eval code (repo `evals/vertex/`) |
| **Legacy eval** | `vertexai.evaluation` | **Deprecated — removal ~2026-06-24** | `EvalTask`, `PointwiseMetric`, `PairwiseMetric` | Existing code only; migrate off |

Migration: `EvalTask(...).evaluate()` → `client.evals.evaluate(...)`; `PointwiseMetric`/`PairwiseMetric` → `LLMMetric` / `PrebuiltMetric.*` / `RubricMetric.*`.

## REST methods (`...:method` on the eval service)

| Method | Granularity | Use | Benefits-finder |
|---|---|---|---|
| `evaluateInstances` | single instance, synchronous | One-off / interactive metric calls | Debug a single Q/A's groundedness |
| `evaluateDataset` | **batch**, async | Score a whole golden set | Nightly run over `card_benefits.evalset.json` |
| `generateInstanceRubrics` | per-instance | **Adaptive rubrics** auto-derived per item | Auto "states the 90-day window" checks |
| `generateLossClusters` | dataset | **Error clustering** of failures into themes | Group failing cases (e.g. "missed coverage window") |
| `recommendSpec` | task | Recommend metric spec for the task | Bootstrap metric config |

## Metric identifier mapping

| Repo / generic | New GenAI Client |
|---|---|
| `groundedness`, `question_answering_*`, `safety`, `coherence`, … | `PrebuiltMetric.*` / `LLMMetric` |
| rubric quality | `RubricMetric.{GENERAL_QUALITY, TEXT_QUALITY, QUESTION_ANSWERING_QUALITY, INSTRUCTION_FOLLOWING, GROUNDING}` |
| computation (exact_match, bleu, rouge_*) | computation metrics (no autorater) |
| tool-call / trajectory | agent eval metrics (`trajectory_*`, tool-call) |
| pairwise / AutoSxS | pairwise metrics + AutoSxS pipeline |

## Agent Engine (deploy + evaluate the live agent)

| Step | API / CLI | Notes |
|---|---|---|
| Wrap ADK agent | `AdkApp(agent=root_agent)` | `root_agent` from `src/card_benefits_finder/agent.py` |
| Deploy | `agent_engines.create(...)` **or** `adk deploy agent_engine` | repo `make deploy` → `deploy/deploy.sh`; needs `STAGING_BUCKET` |
| Evaluate deployed | `client.evals.evaluate(runnable=<engine>, ...)` | pass the deployed agent as `runnable=` |
| **Online eval** | Agent Engine online evaluation on **live traffic** | sampled prod scoring → [08](08-market-monitoring.md) |

## Pipelines / orchestration

| Need | GCP |
|---|---|
| Batch eval at scale | `evaluateDataset` (async batch) |
| Repeatable orchestrated runs | **Vertex AI Pipelines (KFP)** — repo `[pipelines]` extra (`kfp`, `google-cloud-pipeline-components`) |
| Model A vs B arbitration | **AutoSxS** pipeline component |
| Experiment tracking | **Vertex AI Experiments** (`VERTEX_EXPERIMENT=card-benefits-eval`) |
| Continuous prod scoring | Agent Engine online eval |

## ADK ↔ Vertex relationship

The ADK L1 judge criteria are **thin wrappers over the Vertex Gen AI eval service**:

| ADK criterion | Backed by Vertex |
|---|---|
| `final_response_match_v2`, `rubric_based_*_v1` | autorater + (adaptive) rubric metrics |
| `hallucinations_v1` | groundedness/faithfulness autorater |
| `safety_v1` | safety autorater |
| `tool_trajectory_avg_score`, `response_match_score` | computation/trajectory metrics (no Vertex call) |

`judge_model_options.{judge_model, num_samples}` in `eval_config.json` maps to Vertex `AutoraterConfig` → [02](02-llm-as-judge.md). So L1 (`adk eval`) and L2 (`evals/vertex`) share scoring infrastructure; L1 is the agent-centric entry point, L2 the dataset/metric-centric one.

## Sources
- Gen AI evaluation overview: https://cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-overview
- GenAI eval SDK (new): https://cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-quickstart
- Agent Engine: https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview · ADK deploy: https://google.github.io/adk-docs/deploy/agent-engine/
- Vertex AI Pipelines: https://cloud.google.com/vertex-ai/docs/pipelines/introduction · Experiments: https://cloud.google.com/vertex-ai/docs/experiments/intro-vertex-ai-experiments
