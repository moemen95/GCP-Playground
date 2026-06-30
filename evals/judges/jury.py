"""Panel of LLM evaluators — a "jury" (PoLL, Verga et al. 2024, arXiv:2404.18796).

A diverse panel of judges from *different model families*, aggregated by vote
(binary) or mean (graded), correlates better with humans and cancels single-model
self-preference bias — at a fraction of the cost of one frontier judge.

Live: pass real backends from different vendors. Offline: ``stub_panel(n)`` builds
``n`` deterministically *salted* stub judges so the aggregation logic is exercised
without credentials.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from evals.common import MetricResult, POINTWISE, mean
from evals.common.model_backend import ModelBackend, RatingResult, StubBackend, get_backend


class _SaltedStub(StubBackend):
    """A stub judge with a per-juror salt so panel members disagree slightly,
    emulating cross-family variance for offline demos/tests."""

    def __init__(self, salt: str):
        self.name = f"stub:{salt}"
        self._salt = salt

    def rate(self, instruction, *, response, reference=None, context=None, scale=(1, 5)):
        r: RatingResult = super().rate(
            instruction, response=response, reference=reference, context=context, scale=scale
        )
        lo, hi = scale
        jitter = (int(hashlib.sha256((self._salt + response).encode()).hexdigest(), 16) % 7 - 3) / 10.0
        score = max(lo, min(hi, r.score + jitter))
        return RatingResult(score=round(score, 3), scale=scale, rationale=f"[{self.name}]",
                            distribution=r.distribution)


def stub_panel(n: int = 3) -> list[ModelBackend]:
    return [_SaltedStub(f"juror{i}") for i in range(n)]


@dataclass
class JuryVerdict:
    mean_score: float            # normalized 0..1, averaged across jurors
    juror_scores: list[float]    # normalized per juror
    spread: float                # max-min (lower = more agreement)


class Jury:
    """Aggregate several judge backends. Use diverse *families* in production."""

    def __init__(self, panel: list[ModelBackend] | None = None):
        self.panel = panel or stub_panel(3)

    def rate(self, instruction: str, *, response: str, reference: str | None = None,
             context: str | None = None, scale=(1, 5)) -> JuryVerdict:
        norms = []
        for judge in self.panel:
            r = judge.rate(instruction, response=response, reference=reference,
                           context=context, scale=scale)
            norms.append(r.normalized)
        return JuryVerdict(
            mean_score=round(mean(norms), 4),
            juror_scores=[round(x, 3) for x in norms],
            spread=round(max(norms) - min(norms), 3) if norms else 0.0,
        )

    def majority_vote(self, instruction: str, *, response: str, threshold: float = 0.5,
                      reference=None, context=None) -> bool:
        """Binary pass via majority of jurors (each juror passes if normalized>=threshold)."""
        votes = [
            judge.rate(instruction, response=response, reference=reference, context=context).normalized
            >= threshold
            for judge in self.panel
        ]
        return sum(votes) > len(votes) / 2


def jury_dataset(rows: list[dict], *, instruction: str, threshold: float = 0.7,
                 panel: list[ModelBackend] | None = None) -> MetricResult:
    jury = Jury(panel)
    verdicts = [
        jury.rate(instruction, response=r.get("response", ""),
                  reference=r.get("reference"), context=r.get("context"))
        for r in rows
    ]
    avg = mean(v.mean_score for v in verdicts)
    return MetricResult(
        name="jury_quality",
        layer="judges",
        family=POINTWISE,
        score=round(avg, 4),
        passed=avg >= threshold,
        threshold=threshold,
        uses_llm_judge=True,
        n=len(rows),
        details={
            "panel": [j.name for j in jury.panel],
            "avg_spread": round(mean(v.spread for v in verdicts), 3) if verdicts else 0.0,
        },
    )


if __name__ == "__main__":  # pragma: no cover
    from evals.common import load_jsonl

    rows = load_jsonl("rag_qa.jsonl")
    print(jury_dataset(rows, instruction="Rate the answer's correctness and grounding.").to_dict())
