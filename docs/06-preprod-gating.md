# 06 · Pre-Prod Gating & Eval-Ops

Turning scores into a ship/no-ship decision. Repo: `evals/gating/thresholds.yaml` (posture toggle), `pipelines/run_all.py`, `pipelines/ci_gate.py` (`make gate` → nonzero exit on fail).

## Eval hierarchy (Hamel Husain — hamel.dev)

| Level | What | Cost | Benefits-finder |
|---|---|---|---|
| **L1 unit/assertion** | Deterministic code checks | cheap, run every commit | tool name/args match, ROUGE vs gold, format/regex (no LLM) |
| **L2 model/LLM-judge** | Autorater scores vs threshold | medium | groundedness, rubric, G-Eval, faithfulness |
| **L3 human / A-B** | Human review, prod A/B | expensive | spot-check failures, calibrate judges, canary |

Climb only as needed; most regressions die at L1. (Maps to repo `GATE_MODE`: `deterministic` ≈ L1, `full` ≈ L1+L2.)

## Golden datasets

| Property | Guidance | Repo |
|---|---|---|
| **Stratified** | Cover intents × cards × edge cases; **~50/50 pass/fail** so the metric can discriminate | `evals/datasets/*.evalset.json`, `*.test.json` |
| Realistic | Mine real/expected queries, not happy-path only | drill-down, eligibility, recommendation, **refusal** cases |
| Labeled | Gold answer + expected trajectory per item | `final_response` + `tool_uses` in evalset |
| Versioned | Freeze; track score deltas across versions | git-tracked |

Ref: hamel.dev/blog/posts/evals (a balanced set is essential — an all-pass set can't detect regressions).

## Synthetic data

| Method | Ref / link | Use | Pitfall |
|---|---|---|---|
| **RAGAS TestsetGenerator** | docs.ragas.io | Generate Q/A + contexts from `benefits_kb.md` | Drifts from real query distribution |
| **Evol-Instruct** | 2304.12244 | Evolve seeds → harder/deeper/edge variants | Over-complex, off-distribution items |
| Repo synth | `make synth` → `evals.gating.synth_data` → `synthetic_qa.jsonl` | Bootstrap volume | Always human-spot-check; don't gate solely on synthetic |

General pitfalls: self-reinforcing model bias (gen model = SUT family), distribution shift, leakage of answers into questions.

## Regression / CI gates

| Tool | OSS | Use |
|---|---|---|
| **promptfoo** | yes | Declarative assertions + thresholds in CI |
| **DeepEval** | yes | pytest-style LLM eval assertions (`assert_test`) |
| Repo | — | `pipelines/ci_gate.py` consumes `MetricResult[]` → exit code |

## Online vs offline + rollout

| Layer | What | When |
|---|---|---|
| **Offline** | Golden set in CI before merge/deploy | every change (`make gate`) |
| **Online** | Eval live traffic (sampled) | post-deploy, continuous → [08](08-market-monitoring.md) |
| **Shadow** | New agent on real traffic, output not served | pre-canary validation |
| **Canary** | Small % live traffic | gated ramp |
| **A/B** | Split traffic, compare metrics | prompt/model change |
| **Drift monitoring** | Score/dist shift vs baseline over time | continuous; alert on regression |

## Human-in-the-loop & judge calibration

- Calibrate each LLM-judge on a human-labeled slice before trusting its gate — Cohen's κ / Spearman ([02](02-llm-as-judge.md)).
- Route low-confidence / near-threshold items to human review; feed labels back into the golden set.

## Score aggregation

| Strategy | Rule | Use |
|---|---|---|
| **Tiered conjunctive hard-gates** | ALL critical metrics ≥ threshold or **fail** (no averaging) | safety, PII, faithfulness, trajectory — non-negotiable |
| **Weighted soft composite** | Σ(wᵢ·scoreᵢ) ≥ threshold | quality dimensions (fluency, helpfulness, verbosity) |
| **Baseline no-regression** | new ≥ baseline − ε per metric | block silent quality drops |

Recommended gate = hard-gates first (any fail ⇒ stop), then soft composite, then no-regression vs last release.

## `thresholds.yaml` posture toggle

| `mode` | Includes | Determinism | Use |
|---|---|---|---|
| `full` | LLM-judge + deterministic + custom judges | non-deterministic | nightly / release gate |
| `deterministic` | computation + tool-call + trajectory only | fully reproducible | fast PR CI, flake-free |

Drives `uses_llm_judge` filtering in the gate (`evals/common/results.py`); `deterministic` drops all judge metrics.

## Sources
- Hamel Husain evals: https://hamel.dev/blog/posts/evals · golden sets thereof
- Evol-Instruct (WizardLM): 2304.12244 · RAGAS TestsetGenerator: https://docs.ragas.io
- promptfoo: https://promptfoo.dev · DeepEval: https://docs.confident-ai.com
