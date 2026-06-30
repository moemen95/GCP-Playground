# Layer 2 — Vertex AI Gen AI Evaluation Service

Evaluation of the card-benefits-finder agent on Google's **Vertex AI Gen AI
Evaluation Service**. Every module here imports **offline** (lazy `vertexai` /
`google.genai` imports) and raises a clear
`RuntimeError("requires EVAL_BACKEND=vertex and google-cloud-aiplatform")` when the
backend is not configured, so the repo's default test suite runs with no SDK and no
GCP credentials.

> Product naming: "Generative AI on Vertex AI" has been rebranded toward the
> **Gemini Enterprise Agent Platform**. The eval service / SDK names below are
> unchanged.

## Two SDK surfaces

| Surface | Entry point | Status |
| --- | --- | --- |
| **New GenAI client** | `vertexai.Client` -> `client.evals.run_inference()` / `evaluate()` / `generate_rubrics()`; metrics from `vertexai.types.{Metric,PrebuiltMetric,LLMMetric,RubricMetric}` | **PRIMARY** |
| Legacy `vertexai.evaluation` | `EvalTask`, `PointwiseMetric`, `PairwiseMetric`, `MetricPromptTemplateExamples` | **DEPRECATED** (removal ~2026-06-24), see `legacy_evaltask.py` |

## Metric families -> exact identifiers

| Family | Module | Metric ids | LLM judge? |
| --- | --- | --- | --- |
| Computation (text) | `computation_metrics.py` | `exact_match`, `bleu`, `rouge_1`, `rouge_2`, `rouge_l`, `rouge_l_sum` | no |
| Tool use | `computation_metrics.py` | `tool_call_valid`, `tool_name_match`, `tool_parameter_key_match`, `tool_parameter_kv_match` | no |
| Pointwise (model-based) | `pointwise_metrics.py` | `groundedness`, `question_answering_quality`, `question_answering_relevance`, `question_answering_helpfulness`, `question_answering_correctness`, `instruction_following`, `coherence`, `fluency`, `safety`, `verbosity`, `fulfillment` | yes |
| Pairwise / AutoSxS | `pairwise_autosxs.py` | `PairwiseMetric` -> `<metric>/candidate_model_win_rate`; AutoSxS pipeline `autosxs-template/<ver>` (tasks `summarization`, `question_answering`) | yes |
| Trajectory | `trajectory_eval.py` | `trajectory_exact_match`, `trajectory_in_order_match`, `trajectory_any_order_match`, `trajectory_precision`, `trajectory_recall`, `trajectory_single_tool_use` | no |
| Rubric (adaptive/managed) | `rubric_adaptive.py` | `RubricMetric.{GENERAL_QUALITY, TEXT_QUALITY, QUESTION_ANSWERING_QUALITY, INSTRUCTION_FOLLOWING, GROUNDING}` | yes |
| Custom autorater | `custom_autorater.py` | custom `PointwiseMetric` / `PairwiseMetric` (e.g. `benefit_accuracy`) + `AutoraterConfig` | yes |

### Score ranges (pointwise)

Most pointwise metrics are integer **1-5** (higher better). Binary **0/1**:
`groundedness`, `safety`, `question_answering_correctness`. All `MetricResult.score`
values in this layer are normalized to **0..1** for the gate.

## REST methods (under the `aiplatform` API)

| Method | Purpose | SDK call |
| --- | --- | --- |
| `evaluateInstances` | score a single instance | (per-instance eval) |
| `evaluateDataset` | score a whole dataset | `client.evals.evaluate(...)` |
| `generateInstanceRubrics` | adaptive rubric generation | `client.evals.generate_rubrics(...)` |
| `generateLossClusters` | cluster low-scoring examples for error analysis | — |
| `recommendSpec` | recommend an eval spec / metric set | — |

## Backend requirement

All `run_*` functions require **`EVAL_BACKEND=vertex`** plus `GOOGLE_CLOUD_PROJECT`
(and optionally `GOOGLE_CLOUD_LOCATION`, default `us-central1`) and an installed
`google-cloud-aiplatform` / `vertexai`. Without these they raise the shared
`RuntimeError`. The `rubric_adaptive.py` flow additionally **requires the new GenAI
client** (no legacy equivalent). `build_runnable()` (trajectory) and
`autosxs_pipeline_spec()` assemble data and the runnable using only the offline twin,
but still go through the same backend guard.

## Modules

- `_client.py` — `get_genai_client()`, `get_vertex_types()`, `require_vertex()` guard.
- `computation_metrics.py` — text + tool-call computation metrics.
- `pointwise_metrics.py` — model-based pointwise autorater metrics.
- `pairwise_autosxs.py` — in-SDK `PairwiseMetric` + AutoSxS pipeline spec.
- `trajectory_eval.py` — agent trajectory + response, via `runnable=` (offline twin).
- `rubric_adaptive.py` — adaptive / managed rubric metrics (new client required).
- `custom_autorater.py` — custom metrics + `AutoraterConfig` (flipping, multi-sampling).
- `legacy_evaltask.py` — DEPRECATED `vertexai.evaluation` reference.

Each module has lazy imports, the shared `RuntimeError` guard, and a
`python -m evals.vertex.<module>` demo guard.
