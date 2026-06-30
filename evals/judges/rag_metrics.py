"""RAG evaluation metrics (RAGAS family, Es et al. 2023, arXiv:2309.15217).

Generation-side: **faithfulness** (claims supported by context — the hallucination
gate), **answer relevancy** (does the answer address the question). Retrieval-side:
**context precision** (signal-to-noise of retrieved chunks) and **context recall**
(did we retrieve everything needed, vs a reference).

Default implementations are *reference-free lexical heuristics* so they produce
meaningful numbers offline; pass ``backend=`` to use an LLM judge for the claim/
support decisions instead (closer to RAGAS). For full RAGAS, install ``[ragas]``.
"""
from __future__ import annotations

from evals.common import MetricResult, RAG, mean
from evals.common.model_backend import ModelBackend
from evals.common.text_utils import containment, split_sentences, token_set

# A claim counts as supported if this fraction of its content tokens appear in the
# context. Tunable; mirrors an NLI entailment threshold.
SUPPORT_TAU = 0.6


def _claims(answer: str) -> list[str]:
    return split_sentences(answer) or ([answer] if answer.strip() else [])


def faithfulness(answer: str, context: str, *, tau: float = SUPPORT_TAU,
                 backend: ModelBackend | None = None) -> float:
    """Fraction of answer claims supported by the context. 1.0 = fully grounded."""
    claims = _claims(answer)
    if not claims:
        return 1.0
    if backend is not None and backend.name != "stub":
        supported = 0
        for c in claims:
            instr = "Is the claim fully supported by the context? Score 1 if yes, 0 if not."
            r = backend.rate(instr, response=c, context=context, scale=(0, 1))
            supported += 1 if r.normalized >= 0.5 else 0
        return supported / len(claims)
    supported = sum(1 for c in claims if containment(c, context) >= tau)
    return supported / len(claims)


def answer_relevancy(answer: str, question: str) -> float:
    """Proxy for RAGAS answer relevancy: how well the answer covers the question's
    content terms (reference-free, lexical)."""
    if not question.strip():
        return 1.0
    return round(containment(question, answer), 4)


def context_precision(question: str, contexts: list[str], answer: str,
                      *, tau: float = 0.3) -> float:
    """Rank-aware precision: are the relevant chunks ranked first? A chunk is
    'relevant' if it overlaps the answer/question above ``tau``."""
    if not contexts:
        return 0.0
    target = f"{question} {answer}"
    rel = [1 if containment(c, target) >= tau or containment(target, c) >= tau else 0
           for c in contexts]
    if not any(rel):
        return 0.0
    # average precision @ each relevant rank
    hits = 0
    precisions = []
    for i, r in enumerate(rel, 1):
        if r:
            hits += 1
            precisions.append(hits / i)
    return round(mean(precisions), 4)


def context_recall(reference: str, contexts: list[str], *, tau: float = SUPPORT_TAU) -> float:
    """Fraction of reference claims that are recoverable from the retrieved context."""
    claims = _claims(reference)
    if not claims:
        return 1.0
    joined = " ".join(contexts)
    found = sum(1 for c in claims if containment(c, joined) >= tau)
    return round(found / len(claims), 4)


def rag_triad(rows: list[dict], *, threshold: float = 0.7,
              backend: ModelBackend | None = None) -> list[MetricResult]:
    """Score a dataset of ``{question, answer/response, context|contexts, reference?}``
    on the RAG metric family and return one MetricResult per metric."""
    def ctx_list(r):
        if isinstance(r.get("contexts"), list):
            return r["contexts"]
        c = r.get("context", "")
        return c.split("\n") if c else []

    faith, relev, cprec, crec = [], [], [], []
    for r in rows:
        ans = r.get("response") or r.get("answer", "")
        ctx = r.get("context") or "\n".join(ctx_list(r))
        faith.append(faithfulness(ans, ctx, backend=backend))
        relev.append(answer_relevancy(ans, r.get("question", "")))
        cprec.append(context_precision(r.get("question", ""), ctx_list(r), ans))
        if r.get("reference"):
            crec.append(context_recall(r["reference"], ctx_list(r)))

    def mr(name, vals, family=RAG, judge=False):
        avg = mean(vals)
        return MetricResult(name=name, layer="judges", family=family, score=round(avg, 4),
                            passed=avg >= threshold, threshold=threshold, uses_llm_judge=judge,
                            n=len(vals), details={"per_row": [round(v, 3) for v in vals]})

    out = [
        mr("faithfulness", faith, judge=backend is not None and backend.name != "stub"),
        mr("answer_relevancy", relev),
        mr("context_precision", cprec),
    ]
    if crec:
        out.append(mr("context_recall", crec))
    return out


if __name__ == "__main__":  # pragma: no cover
    from evals.common import load_jsonl

    for m in rag_triad(load_jsonl("rag_qa.jsonl")):
        print(m.name, m.score, "pass" if m.passed else "FAIL")
