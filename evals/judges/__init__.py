"""Layer 3 — custom LLM-as-judge + RAG + safety evaluators.

These build what GCP's managed service doesn't give you out of the box: G-Eval
(auto-CoT + probability-weighted scoring), a diverse-family jury (PoLL), judge
bias mitigations, judge↔human calibration, RAG metrics (faithfulness, context
precision/recall, answer relevancy), sentence-level NLI groundedness, and a
safety suite (PII, prompt-injection, refusal balance).

Every evaluator runs offline against the deterministic ``stub`` backend (lexical
heuristics) and live against ``gemini``/``vertex`` — same code path.
"""
