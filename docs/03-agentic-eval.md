# 03 · Agentic Eval (trajectory & tool use)

Scoring the *process*, not just the answer. Scorer identifiers in [01](01-scorers-catalog.md). Benefits-finder tools:
`list_cards, lookup_card_benefits, get_benefit_details, find_cards_for_category, check_eligibility`.

## Outcome vs trajectory

| Axis | Question | Metrics | Benefits-finder example |
|---|---|---|---|
| **Outcome** | Is the final answer right? | `response_match_score`, `question_answering_correctness`, `final_response_match_v2` | "Mobile device limit = CAD 1,000 / 730 days" stated correctly |
| **Trajectory / process** | Did it take the right steps? | trajectory + tool-call families | Drill-down: `lookup_card_benefits` → `get_benefit_details` |

A correct answer via a wrong/lucky path still fails the trajectory gate — important for a bank: the agent must *retrieve before asserting*.

## Trajectory match modes

| Mode | Passes when | Strictness | Use for |
|---|---|---|---|
| `trajectory_exact_match` | Same tools, same order, no extras | highest | Deterministic fixed flows |
| `trajectory_in_order_match` | Reference tools in order, extras OK | high | Multi-turn drill-down (default here) |
| `trajectory_any_order_match` | All reference tools present, any order | medium | Order doesn't matter |
| `trajectory_precision` | — (0..1) | — | Penalize redundant/extra calls |
| `trajectory_recall` | — (0..1) | — | Penalize missing steps |
| `trajectory_single_tool_use` | One named tool used ≥1 | targeted | "must call `check_eligibility`" |

Repo: ADK `tool_trajectory_avg_score` with `match_type` `IN_ORDER` (`test_config.json`) / `ANY_ORDER` (`eval_config.json`).

## Tool-selection & parameter correctness

| Concern | Metric | Benefits-finder failure mode |
|---|---|---|
| Right tool | `tool_name_match`, rubric `selects_correct_tool` | uses `lookup_card_benefits` when user wants one benefit's terms |
| Right arg keys | `tool_parameter_key_match` | omits `benefit_category` |
| Right arg values | `tool_parameter_kv_match`, rubric `passes_grounded_args` | invents `card_name` / wrong `annual_income` |
| **Irrelevance detection** | rubric `no_tool_for_refusals` | calls a tool on an out-of-scope/advice request |

**BFCL** (Berkeley Function-Calling Leaderboard) methodology: **AST accuracy** (parse the call, match function + params against ground truth) + **executable accuracy** (run it, check output) + **irrelevance/relevance detection** (don't call when no function applies). Mirror this: AST = `tool_parameter_kv_match`; irrelevance = `no_tool_for_refusals`.

## Task success, multi-turn, efficiency

| Dimension | Signal | Repo hook |
|---|---|---|
| **Goal/task success** | End state achieved (right benefit terms returned) | outcome metrics + rubric |
| **Multi-turn** | User-simulator drives a session; eval per turn + whole session | `card_benefits.evalset.json` (`conversation[]`), `multi_turn_chat_quality` |
| **Efficiency** | Step count, redundant/duplicate calls, latency | `trajectory_precision`; gate on extra calls |
| **Reliability** | Stable across reruns (stochastic agent) | pass^k (tau-bench) |

## Benchmarks (methodology to borrow)

| Benchmark | Ref / link | Core idea | Map to benefits-finder |
|---|---|---|---|
| **tau-bench** (τ-bench) | 2406.12045 | Tool-agent-user interaction in retail/airline domains; **pass^k** = passes all k independent trials (reliability) | Rerun each eval case k× → reliability gate before pre-prod |
| **AgentBench** | 2308.03688 | Multi-environment agent reasoning/acting eval | General agentic capability baseline |
| **BFCL** | gorilla.cs.berkeley.edu | AST + executable + irrelevance for function calling | Direct model for tool-call scoring above |
| **WebArena** | webarena.dev | Realistic web tasks, functional success | Not applicable (no browsing) |
| **GAIA** | 2311.12983 | Multi-step real-world assistant tasks | Reasoning-depth reference |
| **SWE-bench** | 2310.06770 | Resolve real GitHub issues, test-verified | Outcome-verification pattern (use deterministic tool output as ground truth) |

## Sources
- tau-bench: 2406.12045 · AgentBench: 2308.03688 · GAIA: 2311.12983 · SWE-bench: 2310.06770
- BFCL: https://gorilla.cs.berkeley.edu/leaderboard.html · WebArena: 2307.13854
- Vertex trajectory metrics: https://cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-overview · ADK: https://google.github.io/adk-docs/evaluate/
