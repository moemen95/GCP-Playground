"""G-Eval pointwise judge (Liu et al., 2023, arXiv:2303.16634).

Two ideas, both implemented here:

1. **Auto-CoT eval steps** — from a coarse criterion we derive an explicit
   step-by-step scoring checklist before scoring (de-vagues the criterion).
2. **Probability-weighted scoring** — instead of the single argmax integer, the
   score is ``E[s] = Σ p(s)·s`` over the score-token distribution, which escapes
   the integer-clustering / tie problem and yields continuous scores.

Runs on any :class:`ModelBackend`. The ``stub`` backend supplies a deterministic
peaked distribution so the prob-weighting is exercised offline.
"""
from __future__ import annotations

from dataclasses import dataclass

from evals.common import MetricResult, POINTWISE, mean
from evals.common.model_backend import ModelBackend, RatingResult, get_backend

# A few card-domain criteria with their definitions (the coarse criterion the
# auto-CoT step expands).
CRITERIA = {
    "groundedness": "The response only states benefit facts that are supported by the provided context.",
    "completeness": "The response includes the key terms of the benefit: coverage limit, eligibility, and notable exclusions.",
    "correctness": "The response correctly answers the user's question about the card benefit.",
    "tone": "The response is concise, professional, and easy for a cardholder to act on.",
}


def auto_cot_steps(criterion_name: str, criterion_def: str, backend: ModelBackend) -> list[str]:
    """Generate (or, offline, template) the chain-of-thought evaluation steps."""
    if backend.name == "stub":
        return [
            f"Read the user question and the response for '{criterion_name}'.",
            f"Identify each claim relevant to: {criterion_def}",
            "Check each claim against the context / reference.",
            "Penalize unsupported, missing, or incorrect claims.",
            "Assign an integer score reflecting how well the criterion is met.",
        ]
    prompt = (
        f"You are designing an evaluation rubric. Criterion '{criterion_name}': "
        f"{criterion_def}\nList 4-6 concise, ordered steps a judge should follow to "
        f"score a response on this criterion. One step per line."
    )
    out = backend.complete(prompt, temperature=0.0)
    steps = [ln.strip(" -*0123456789.") for ln in out.text.splitlines() if ln.strip()]
    return steps or auto_cot_steps(criterion_name, criterion_def, get_backend("stub"))


@dataclass
class GEvalScore:
    criterion: str
    expected_score: float          # probability-weighted, continuous
    normalized: float              # 0..1
    argmax_score: float            # the single most-likely integer (for contrast)
    scale: tuple[int, int]
    steps: list[str]
    rationale: str


def g_eval(
    *,
    criterion_name: str,
    response: str,
    question: str | None = None,
    context: str | None = None,
    reference: str | None = None,
    scale: tuple[int, int] = (1, 5),
    backend: ModelBackend | None = None,
    criterion_def: str | None = None,
) -> GEvalScore:
    backend = backend or get_backend()
    criterion_def = criterion_def or CRITERIA.get(criterion_name, criterion_name)
    steps = auto_cot_steps(criterion_name, criterion_def, backend)
    instruction = (
        f"Evaluate the response for the criterion '{criterion_name}': {criterion_def}\n"
        f"Question: {question or '(n/a)'}\n"
        "Follow these evaluation steps:\n- " + "\n- ".join(steps)
    )
    rating: RatingResult = backend.rate(
        instruction, response=response, reference=reference, context=context, scale=scale
    )
    # Probability-weighted expected score.
    dist = rating.distribution or {int(round(rating.score)): 1.0}
    total = sum(dist.values()) or 1.0
    expected = sum(s * p for s, p in dist.items()) / total
    lo, hi = scale
    argmax = max(dist.items(), key=lambda kv: kv[1])[0]
    return GEvalScore(
        criterion=criterion_name,
        expected_score=round(expected, 4),
        normalized=round((expected - lo) / (hi - lo) if hi > lo else 0.0, 4),
        argmax_score=float(argmax),
        scale=scale,
        steps=steps,
        rationale=rating.rationale,
    )


def g_eval_dataset(
    rows: list[dict],
    *,
    criterion_name: str,
    threshold: float = 0.7,
    backend: ModelBackend | None = None,
) -> MetricResult:
    """Aggregate G-Eval over rows of ``{question, response, context?, reference?}``."""
    backend = backend or get_backend()
    scores = [
        g_eval(
            criterion_name=criterion_name,
            response=r.get("response", ""),
            question=r.get("question"),
            context=r.get("context"),
            reference=r.get("reference"),
            backend=backend,
        ).normalized
        for r in rows
    ]
    avg = mean(scores)
    return MetricResult(
        name=f"g_eval_{criterion_name}",
        layer="judges",
        family=POINTWISE,
        score=round(avg, 4),
        passed=avg >= threshold,
        threshold=threshold,
        uses_llm_judge=True,
        n=len(rows),
        details={"per_row": [round(s, 3) for s in scores], "judge": backend.name},
    )


if __name__ == "__main__":  # pragma: no cover
    from evals.common import load_jsonl

    rows = load_jsonl("rag_qa.jsonl")
    print(g_eval_dataset(rows, criterion_name="groundedness").to_dict())
