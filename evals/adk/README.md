# Layer 1 — ADK-native evaluation

Evaluates the `card_benefits_finder` ADK agent (`src/card_benefits_finder`,
exposing `root_agent`) using Google ADK's own eval machinery: `.test.json` /
`.evalset.json` datasets scored by ADK metrics, plus a custom metric.

## Files

| File | Purpose |
| --- | --- |
| `../datasets/card_benefits_basic.test.json` | Single-session `.test.json` — 4 cases (benefit details, "what benefits", best-card-for-groceries, out-of-scope refusal). |
| `../datasets/card_benefits.evalset.json` | Multi-session evalset incl. a 3-turn conversation and `session_input.state`. |
| `test_config.json` | Simple/local criteria — no Vertex needed. |
| `eval_config.json` | Full `EvalConfig` — LLM-judge + rubric metrics + the custom metric (`custom_metrics` block). |
| `metrics.py` | Custom metric `benefit_citation_score` (+ offline-testable pure-python helper). |
| `test_eval_pytest.py` | Pytest runner (ADK tests `importorskip`-guarded; one always-on offline unit test). |
| `run_adk_eval.sh` | Commented `adk eval` CLI invocations + `adk web` notes. |

## Criteria catalog

| Metric | Threshold (here) | When to use | Needs `GOOGLE_CLOUD_PROJECT`? |
| --- | --- | --- | --- |
| `tool_trajectory_avg_score` | `1.0` (`IN_ORDER` / `ANY_ORDER`) | Exact tool-call trajectory match (name + args). | No |
| `response_match_score` | `0.6`–`0.7` | ROUGE-1 lexical overlap vs. reference final response. | No |
| `final_response_match_v2` | `0.8` | LLM-judge semantic match of the final response. | Yes (Vertex) |
| `hallucinations_v1` | `0.8` | LLM-judge groundedness; `evaluate_intermediate_nl_responses` also checks intermediate NL. | Yes (Vertex) |
| `safety_v1` | `0.8` | Safety/harmlessness of responses. | Yes (Vertex) |
| `rubric_based_final_response_quality_v1` | `0.8` | Rubric-scored response quality (cites limit/eligibility/window, refuses out-of-scope). | Yes (Vertex) |
| `rubric_based_tool_use_quality_v1` | `0.8` | Rubric-scored tool selection + grounded args. | Yes (Vertex) |
| `my_benefit_citation_score` (custom) | `0.7` | Domain proxy: rewards citing coverage limit + eligibility + coverage window. | No |

`test_config.json` uses only the first two (offline-friendly). `eval_config.json`
exercises the LLM-judge/rubric metrics plus the custom metric. The `*_v2`,
rubric-based, `hallucinations_v1`, and `safety_v1` metrics call a Vertex judge
model, so set `GOOGLE_CLOUD_PROJECT` (and `GOOGLE_CLOUD_LOCATION`,
`GOOGLE_GENAI_USE_VERTEXAI=TRUE`) before running them.

## Three ways to run

1. **pytest**

   ```bash
   pytest evals/adk/test_eval_pytest.py -v
   ```

   The ADK integration tests skip automatically when `google-adk` isn't
   installed; the custom-metric unit test always runs.

2. **`adk eval` CLI** — see `run_adk_eval.sh` (uncomment a block):

   ```bash
   adk eval src/card_benefits_finder \
     evals/datasets/card_benefits_basic.test.json \
     --config_file_path evals/adk/test_config.json \
     --num_runs 2 --print_detailed_results
   ```

3. **Web UI**

   ```bash
   adk web src
   ```

   Pick the `card_benefits_finder` app, chat, then in the **Eval** tab click
   **Add current session** to capture the conversation as an eval case and
   **Run Evaluation** to score it.

## Custom metric

`benefit_citation_score` (registered in `eval_config.json` as
`my_benefit_citation_score`) awards equal credit across three dimensions of a
well-grounded benefit answer: a coverage **limit**, an **eligibility**
condition, and a **coverage window**. The scoring lives in the pure-python
`score_benefit_citation_text`, so it is deterministic and unit-tested offline;
the ADK entry point wraps it into an `EvaluationResult`.
