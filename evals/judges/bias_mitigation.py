"""Mitigations for known LLM-judge biases.

* **Position/order bias** (Wang et al. 2023, arXiv:2305.17926): pairwise judges
  favour a slot. ``balanced_pairwise`` runs both orders and only declares a winner
  when the two orders agree, else "tie".
* **Verbosity/length bias** (Length-Controlled AlpacaEval, arXiv:2404.04475):
  judges over-prefer longer answers. ``length_controlled_winrate`` regresses the
  raw win signal on length difference and reports the length-neutralized rate.
"""
from __future__ import annotations

from dataclasses import dataclass

from evals.common.model_backend import ModelBackend, get_backend
from evals.common.text_utils import tokenize


@dataclass
class PairwiseOutcome:
    verdict: str          # "A" | "B" | "tie"
    consistent: bool      # did both orderings agree?
    raw: tuple[str, str]  # (order1 verdict, order2-mapped verdict)


def balanced_pairwise(instruction: str, *, response_a: str, response_b: str,
                      criteria: str | None = None,
                      backend: ModelBackend | None = None) -> PairwiseOutcome:
    """Flip-and-average: judge (A,B) and (B,A); require agreement for a winner."""
    backend = backend or get_backend()
    v1, _ = backend.choose(instruction, response_a=response_a, response_b=response_b, criteria=criteria)
    # Second pass with positions swapped; remap the verdict back to A/B space.
    v2_raw, _ = backend.choose(instruction, response_a=response_b, response_b=response_a, criteria=criteria)
    v2 = {"A": "B", "B": "A", "tie": "tie"}[v2_raw]
    consistent = v1 == v2
    verdict = v1 if consistent else "tie"
    return PairwiseOutcome(verdict=verdict, consistent=consistent, raw=(v1, v2))


def position_bias_rate(pairs: list[dict], *, backend: ModelBackend | None = None) -> float:
    """Fraction of pairs whose verdict flips when order is swapped (0 = unbiased)."""
    backend = backend or get_backend()
    flips = 0
    for p in pairs:
        out = balanced_pairwise(p.get("instruction", "Which response is better?"),
                                response_a=p["response_a"], response_b=p["response_b"],
                                criteria=p.get("criteria"), backend=backend)
        flips += 0 if out.consistent else 1
    return flips / len(pairs) if pairs else 0.0


def length_controlled_winrate(pairs: list[dict]) -> dict:
    """Length-neutralized win rate for candidate (A) vs baseline (B).

    ``pairs`` rows: ``{win: 1|0|0.5, len_a, len_b}`` (or provide ``response_a/b``).
    Returns raw win rate, a simple length-bias slope, and the length-controlled
    win rate (raw minus the portion explained by length advantage).
    """
    xs, ys = [], []
    for p in pairs:
        la = p.get("len_a", len(tokenize(p.get("response_a", ""))))
        lb = p.get("len_b", len(tokenize(p.get("response_b", ""))))
        denom = (la + lb) or 1
        xs.append((la - lb) / denom)            # length advantage of A, in [-1,1]
        ys.append(float(p["win"]))               # 1 if A won, 0 if B, 0.5 tie
    n = len(xs)
    if n == 0:
        return {"raw_winrate": 0.0, "length_slope": 0.0, "length_controlled_winrate": 0.0, "n": 0}
    mx, my = sum(xs) / n, sum(ys) / n
    var = sum((x - mx) ** 2 for x in xs) or 1e-9
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = cov / var                            # how much length advantage drives wins
    # Neutralize: predict win rate at zero length advantage (intercept).
    controlled = my - slope * mx
    return {
        "raw_winrate": round(my, 4),
        "length_slope": round(slope, 4),
        "length_controlled_winrate": round(max(0.0, min(1.0, controlled)), 4),
        "n": n,
    }
