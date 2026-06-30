# 04 · RAG & Groundedness

The benefits-finder retrieves T&C/benefit chunks (`knowledge/benefits_kb.md` via `retriever.py`, `context_for()`) before answering.
The faithfulness gate is the single most important quality control for a bank Q&A agent.

## RAGAS metrics (arXiv 2309.15217)

### Generation-side (answer quality)

| Metric | Inputs | Range | What it catches | Benefits-finder use |
|---|---|---|---|---|
| **faithfulness** | answer, retrieved contexts | 0..1 | Claims **not** entailed by context (hallucination) | **Primary hallucination gate** — every coverage limit/window must trace to a T&C chunk |
| **answer_relevancy** | question, answer | 0..1 | Off-topic / padded answers | Answer addresses the benefit asked about |
| **answer_correctness** | answer, ground-truth | 0..1 | Factual + semantic match to gold | vs golden answer in evalset |

### Retrieval-side (context quality)

| Metric | Inputs | Range | What it catches | Benefits-finder use |
|---|---|---|---|---|
| **context_precision** | question, contexts (ranked) | 0..1 | Irrelevant chunks ranked high | Retriever returns the right clause near top-k |
| **context_recall** | ground-truth, contexts | 0..1 | Needed clause **missing** from retrieval | **Right T&C clause retrieved at all** (e.g. the 730-day mobile window) |
| **context_entities_recall** | gt entities, contexts | 0..1 | Key entities (limits, windows) absent | Numbers/limits present in context |
| **noise_sensitivity** | answer, contexts, gt | 0..1 (lower better) | Errors induced by irrelevant retrieved chunks | Robustness when KB returns near-miss chunks |

**Decompose failures:** low faithfulness + high context_recall → generation bug (model ignores good context); low context_recall → retrieval bug. Fix the right layer.

## Hallucination detection (sentence-level)

| Method | Ref | Mechanism | When to use |
|---|---|---|---|
| **NLI / entailment** | TRUE 2204.04991 | Per-sentence: does context **entail** the claim? Score = fraction entailed | Reference-grounded faithfulness; repo L3 NLI groundedness |
| **SelfCheckGPT** | 2303.08896 | Sample multiple responses; inconsistency ⇒ likely hallucinated | **Reference-free** (no retrieved context available) |
| **Vectara HHEM** | huggingface.co/vectara/hallucination_evaluation_model | Lightweight trained hallucination classifier | Cheap high-throughput batch faithfulness scoring |

Repo pattern: split answer into sentences (`text_utils`), check each against `context_for(...)` output; aggregate to a 0..1 groundedness. ADK equivalent: `hallucinations_v1` (with `evaluate_intermediate_nl_responses: true`).

## Frameworks

| Framework | Ref / link | What it adds |
|---|---|---|
| **RAGAS** | 2309.15217 · docs.ragas.io | The metric suite above + `TestsetGenerator` for synthetic eval data ([06](06-preprod-gating.md)) |
| **ARES** | 2311.09476 | Trains lightweight LLM judges + uses PPI (prediction-powered inference) for statistically-bounded RAG scores |
| **TruLens — RAG triad** | trulens.org | 3 feedback fns: **context relevance** (Q↔context) · **groundedness** (answer↔context) · **answer relevance** (Q↔answer) |

The **RAG triad** is the minimal mental model: each edge of (Question, Context, Answer) gets a score; a weak edge localizes the defect.

## Map to the benefits-finder gate

| Gate | Metric | Threshold posture |
|---|---|---|
| No hallucinated coverage terms | RAGAS `faithfulness` / NLI groundedness / `hallucinations_v1` | **hard gate** (high, e.g. ≥0.8) — conjunctive |
| Right clause retrieved | `context_recall` | hard gate |
| Retriever ranking | `context_precision` | soft / monitored |
| Answer on-topic | `answer_relevancy` | soft |

## Sources
- RAGAS: 2309.15217 · ARES: 2311.09476 · TRUE: 2204.04991 · SelfCheckGPT: 2303.08896
- Vectara HHEM: https://huggingface.co/vectara/hallucination_evaluation_model · TruLens: https://www.trulens.org
- Vertex groundedness: https://cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-overview
