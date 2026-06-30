# 01 · Scorers Catalog (master)

Every scorer usable in this repo, by family. "Judge?" = needs an LLM autorater. Range `0..1` unless noted.
Maps to GCP surfaces in [07](07-gcp-eval-mapping.md). For the benefits-finder, columns "When to use" reference its 5 tools
(`list_cards, lookup_card_benefits, get_benefit_details, find_cards_for_category, check_eligibility`).

## Computation (deterministic, reference-based)

| Identifier | Judge? | Inputs | Range | When to use |
|---|---|---|---|---|
| `exact_match` | no | response, reference | {0,1} | Canonical short answers (card name, yes/no eligibility) |
| `bleu` | no | response, reference | 0..1 | N-gram overlap; weak signal for free-form benefit prose |
| `rouge_1` / `rouge_2` / `rouge_l` / `rouge_l_sum` | no | response, reference | 0..1 | Recall of key benefit terms vs golden answer; `rouge_l` = LCS (used offline in `text_utils.rouge_l_f1`) |

## Tool-call (single-call correctness)

| Identifier | Judge? | Inputs | Range | When to use |
|---|---|---|---|---|
| `tool_call_valid` | no | predicted call, reference | {0,1} | Call is well-formed (valid tool, parsable args) |
| `tool_name_match` | no | predicted, reference names | {0,1} | Right tool chosen (e.g. `get_benefit_details` not `lookup_card_benefits`) |
| `tool_parameter_key_match` | no | predicted, reference args | 0..1 | Right arg keys present (`card_name`, `benefit_category`) |
| `tool_parameter_kv_match` | no | predicted, reference args | 0..1 | Right keys **and** values (`card_name="Tangerine World Mastercard"`) |

## Trajectory (multi-step path)

| Identifier | Judge? | Inputs | Range | When to use |
|---|---|---|---|---|
| `trajectory_exact_match` | no | predicted, reference traj | {0,1} | Strictest: same tools, same order, no extras |
| `trajectory_in_order_match` | no | "" | {0,1} | Reference tools appear in order (extras allowed) — default for drill-down flows |
| `trajectory_any_order_match` | no | "" | {0,1} | Reference tools all present, order-agnostic |
| `trajectory_precision` | no | "" | 0..1 | Of called tools, fraction that were expected (penalizes redundant calls) |
| `trajectory_recall` | no | "" | 0..1 | Of expected tools, fraction called (penalizes missing steps) |
| `trajectory_single_tool_use` | no | traj, tool name | {0,1} | A specific tool was used at least once |

ADK aggregate of the above: `tool_trajectory_avg_score` (`match_type`: `EXACT`/`IN_ORDER`/`ANY_ORDER`).

## Model-based pointwise (single-response autorater)

| Identifier | Judge? | Inputs | Range | When to use |
|---|---|---|---|---|
| `groundedness` | yes | response, context | 0..1 | Claims supported by retrieved T&C — primary RAG gate |
| `question_answering_quality` | yes | prompt, response (+ctx) | 0..1 | Overall QA quality |
| `question_answering_relevance` | yes | prompt, response | 0..1 | On-topic to the question |
| `question_answering_helpfulness` | yes | prompt, response | 0..1 | Actionable / complete |
| `question_answering_correctness` | yes | prompt, response, reference | {0,1}/0..1 | Factually correct vs golden |
| `instruction_following` | yes | instruction, response | 0..1 | Obeys system rules (cite terms, stay in scope) |
| `coherence` | yes | response | 0..1 | Logically consistent |
| `fluency` | yes | response | 0..1 | Grammatical |
| `safety` | yes | response | 0..1 | Free of harmful content |
| `verbosity` | yes | response | 0..1 | Appropriately concise (prompt rule #6) |
| `fulfillment` | yes | instruction, response | 0..1 | Fully addresses the request |
| `summarization_quality` | yes | text, summary | 0..1 | Benefit-summary quality |
| `multi_turn_chat_quality` | yes | conversation | 0..1 | Whole-session quality (drill-down sets) |
| `multi_turn_safety` | yes | conversation | 0..1 | Session-level safety |

## Pairwise (A vs B)

| Identifier | Judge? | Inputs | Range | When to use |
|---|---|---|---|---|
| `pairwise_question_answering_quality` | yes | prompt, A, B | win/tie/lose | Prompt/model regression comparisons |
| `pairwise_summarization_quality`, `pairwise_*` (relevance/helpfulness/instruction_following/coherence/fluency/safety/groundedness/verbosity) | yes | prompt, A, B | win/tie/lose | Per-dimension SxS |
| **AutoSxS** | yes | dataset of (A,B) | win-rate + CI | Batch model arbitration pipeline — see [07](07-gcp-eval-mapping.md) |

Always flip A/B and average to remove position bias → [02](02-llm-as-judge.md).

## Rubric (criteria checklists)

| Identifier | Judge? | Inputs | Range | When to use |
|---|---|---|---|---|
| `RubricMetric.GENERAL_QUALITY` | yes | prompt, response | 0..1 | Broad quality bundle |
| `RubricMetric.TEXT_QUALITY` | yes | response | 0..1 | Writing quality |
| `RubricMetric.QUESTION_ANSWERING_QUALITY` | yes | prompt, response | 0..1 | QA-specific rubric |
| `RubricMetric.INSTRUCTION_FOLLOWING` | yes | instruction, response | 0..1 | Rule adherence |
| `RubricMetric.GROUNDING` | yes | response, context | 0..1 | Grounding rubric |
| **Adaptive rubrics** | yes | prompt | per-rubric pass | `generateInstanceRubrics` auto-derives per-item criteria (e.g. "states the 90-day window") |

## ADK criteria (Layer 1 — `evals/adk/eval_config.json`)

| Identifier | Judge? | Range | When to use (this repo) |
|---|---|---|---|
| `tool_trajectory_avg_score` | no | 0..1 | Threshold 1.0, `ANY_ORDER` — exact tool path |
| `response_match_score` | no | 0..1 | ROUGE-style vs reference (0.6 gate) |
| `final_response_match_v2` | yes | 0..1 | LLM match of final answer (0.8, `num_samples=5`) |
| `hallucinations_v1` | yes | 0..1 | Per-claim grounding incl. intermediate NL responses (0.8) |
| `safety_v1` | yes | 0..1 | Response safety (0.8) |
| `rubric_based_final_response_quality_v1` | yes | 0..1 | Custom rubrics: `cites_coverage_terms`, `cites_coverage_window`, `names_the_benefit`, `refuses_out_of_scope` |
| `rubric_based_tool_use_quality_v1` | yes | 0..1 | Rubrics: `selects_correct_tool`, `passes_grounded_args`, `no_tool_for_refusals` |
| `multi_turn_*` | yes | 0..1 | Session-level quality/safety |
| `my_benefit_citation_score` (custom) | no | 0..1 | Code metric `evals.adk.metrics.benefit_citation_score` — rewards citing limit+eligibility+window |

## Custom judges (Layer 3 — `evals/judges/`)

| Identifier | Judge? | Inputs | Range | When to use |
|---|---|---|---|---|
| **G-Eval** | yes | criteria, response | 1..5 → 0..1 | Auto-CoT eval steps + prob-weighted score; flexible custom criteria → [02](02-llm-as-judge.md) |
| RAG `faithfulness` | yes | answer, contexts | 0..1 | Hallucination gate — claims entailed by retrieved T&C → [04](04-rag-groundedness.md) |
| RAG `context_precision` | yes/det | question, contexts | 0..1 | Ranking quality of retrieved chunks |
| RAG `context_recall` | yes | reference, contexts | 0..1 | Right T&C clause retrieved at all |
| RAG `answer_relevancy` | yes | question, answer | 0..1 | Answer addresses the question |
| NLI `groundedness` | yes (NLI) | answer sentences, context | 0..1 | Sentence-level entailment (TRUE/HHEM-style) → [04](04-rag-groundedness.md) |

## Sources
- Vertex Gen AI eval metrics: https://cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-overview
- ADK eval criteria: https://google.github.io/adk-docs/evaluate/
- RAGAS metrics: https://docs.ragas.io · BLEU: https://aclanthology.org/P02-1040 · ROUGE: https://aclanthology.org/W04-1013
- G-Eval: arXiv 2303.16634
