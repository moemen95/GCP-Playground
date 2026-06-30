# 02 · LLM-as-Judge

How autoraters score the benefits-finder, their biases, and how GCP exposes them. Catalog of judge metrics in [01](01-scorers-catalog.md).

## Judge modes

| Mode | Question answered | Needs reference? | GCP / repo surface | When to use |
|---|---|---|---|---|
| **Pointwise** | "Score this response 0..1" | optional | Vertex `PointwiseMetric`, ADK `*_v1`, L3 G-Eval | Absolute gating thresholds |
| **Pairwise (SxS)** | "Is A or B better?" | optional | Vertex `PairwiseMetric`, AutoSxS | Prompt/model regression, leaderboards |
| **Reference-based** | "Match the golden answer?" | yes | `question_answering_correctness`, `final_response_match_v2` | Have a gold answer per item |
| **Reference-free** | "Quality on its own?" | no | `groundedness`, `coherence`, rubric | No gold answer; judge uses context/criteria |

Pairwise is more reliable than pointwise when absolute calibration is hard; reference-free scales to large synthetic sets.

## G-Eval (arXiv 2303.16634)

| Step | Mechanism |
|---|---|
| 1. Define task + criteria | NL criteria (e.g. "cites coverage limit, window, eligibility") |
| 2. Auto-CoT | Judge LLM generates explicit **evaluation steps** from the criteria |
| 3. Score | LLM emits a score (e.g. 1–5) following the steps |
| 4. Prob-weighting | Weight scores by output-token probabilities → finer-grained, less ties |

Repo: `LLMResult.score_distribution {score: prob}` (`evals/common/model_backend.py`) carries the token distribution for prob-weighted scoring. Higher human correlation than vanilla direct-scoring prompts.

## Benchmarks for judge reliability

| Benchmark | Ref | Finding |
|---|---|---|
| **MT-Bench / Chatbot Arena** | 2306.05685 | Strong LLM judges reach **~80–85% agreement** with human preferences (≈ human–human); introduced position/verbosity/self-enhancement bias taxonomy |
| **PoLL — Panel of LLM evaluators** | Verga et al. 2404.18796 | A **diverse-family jury** (e.g. small models from 3 vendors) beats a single GPT-4 judge: κ **0.763 vs 0.627** on NQ, reduces intra-model bias, **~7–8× cheaper** |

Takeaway for Tangerine gating: prefer a **cross-family jury** (e.g. Gemini + a second family) for high-stakes gates over a single autorater.

## Bias → mitigation

| Bias | Symptom | Mitigation | Ref |
|---|---|---|---|
| **Position** | Prefers first (or last) candidate in SxS | Flip A/B, average both orders | 2305.17926 |
| **Verbosity / length** | Longer answers score higher | Length-controlled scoring; penalize verbosity dimension | 2404.04475 |
| **Self-enhancement** | Judge favors its own family's outputs | Cross-family jury (PoLL) | 2306.05685 |
| **Sycophancy** | Agrees with user/asserted framing | Neutral prompts; don't reveal preferred answer | 2310.13548 |
| **Formatting** | Markdown/structure sways score | Strip/normalize formatting; rubric anchors | 2306.05685 |

## Calibration (validate the judge vs humans)

| Metric | Use | Bands |
|---|---|---|
| **Cohen's κ** | Binary/categorical judge vs human | <0.40 poor · 0.40–0.60 moderate · 0.60–0.80 substantial · >0.80 near-perfect |
| **Spearman ρ / Kendall τ** | Ordinal/continuous score correlation | report ρ on a held-out human-labeled slice |
| **TPR / TNR** | Binary judges (e.g. `refuses_out_of_scope`) | gate needs high TNR (don't pass unsafe) **and** TPR |

Calibrate on a labeled subset of the golden set before trusting a judge in the gate → [06](06-preprod-gating.md).

## How GCP exposes this — `AutoraterConfig`

| Knob | Effect |
|---|---|
| `autorater_model` / judge model | Pick the autorater (e.g. `gemini-flash-latest`); repo `JUDGE_MODEL` env |
| **response flipping** | Auto-runs both A/B orders for pairwise → removes position bias |
| **multi-sampling** | `num_samples` (repo uses 5 in `eval_config.json`) → averages judge variance |
| sampling count / temperature | Stability vs cost trade-off |

ADK `judge_model_options` (`judge_model`, `num_samples`) is the L1 wrapper over the same Vertex autorater.

## Sources
- G-Eval: arXiv 2303.16634 · MT-Bench/Arena: 2306.05685 · PoLL: 2404.18796
- Position bias: 2305.17926 · Length-controlled: 2404.04475 · Sycophancy: 2310.13548
- Vertex AutoraterConfig / pairwise: https://cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-overview
