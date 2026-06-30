"""Synthetic eval-set generation from the knowledge base.

Bootstraps Q/A test cases from the card catalog (every benefit, fee, and
eligibility fact), evolves a few phrasing variants (a light Evol-Instruct), and
captures the agent's transcript so the rows are gate-ready. Mix the output with
real production traces before trusting it — synthetic questions skew clean.

Live mode (``--backend gemini|vertex``) can paraphrase questions with an LLM;
offline it uses deterministic templates.

Usage:
    python -m evals.gating.synth_data --out evals/datasets/synthetic_qa.jsonl --n 40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from card_benefits_finder import local_agent, tools  # noqa: E402
from evals.common import write_jsonl  # noqa: E402
from evals.common.model_backend import get_backend  # noqa: E402

# Phrasing templates (the "breadth" axis of Evol-Instruct).
_BENEFIT_TEMPLATES = [
    "What does {card}'s {title} cover?",
    "Tell me about the {title} on the {card}.",
    "Does the {card} include {title}?",
    "What's the limit on {title} for the {card}?",
]
_CATEGORY_TO_QUESTION = {
    "rental_car_insurance": "Does the {card} cover rental cars?",
    "mobile_device_insurance": "Is my phone insured with the {card}?",
    "travel_medical_insurance": "Does the {card} have travel medical coverage?",
    "purchase_protection": "Are new purchases protected on the {card}?",
    "extended_warranty": "Does the {card} extend manufacturer warranties?",
    "travel_perk": "Do I get airport wi-fi with the {card}?",
    "fraud_protection": "Am I protected from fraud on the {card}?",
}


def _evolve(question: str, backend) -> str:
    """Optionally deepen/rephrase a question with an LLM (live backends only)."""
    if backend.name == "stub":
        return question
    out = backend.complete(
        f"Rephrase this cardholder question to be more specific and natural, "
        f"keeping the same intent. Question: {question}\nRephrased:", temperature=0.3)
    return (out.text.strip().splitlines() or [question])[0][:200] or question


def generate(n: int = 40, *, backend_name: str = "stub") -> list[dict]:
    backend = get_backend(backend_name)
    catalog = tools._catalog()
    rows: list[dict] = []
    ti = 0
    for card in catalog["cards"]:
        card_name = card["name"]
        for b in card["benefits"]:
            cat = b["category"]
            template = _CATEGORY_TO_QUESTION.get(cat, _BENEFIT_TEMPLATES[ti % len(_BENEFIT_TEMPLATES)])
            q = _evolve(template.format(card=card_name, title=b["title"]), backend)
            ti += 1
            ref = f"{b['title']}: {b['summary']} Limit: {b['limit']} Eligibility: {b['eligibility']}"
            turn = local_agent.run_turn(q)
            rows.append({
                "id": f"synth-{card['card_id']}-{cat}",
                "question": q,
                "reference": ref,
                "expected_trajectory": [{"name": "get_benefit_details",
                                         "args": {"card_name": card_name, "benefit_category": cat}}],
                "expected_refusal": False,
                "response": turn.final_response,
                "context": turn.context,
                "predicted_trajectory": turn.trajectory,
                "refused": turn.refused,
                "synthetic": True,
            })
            if len(rows) >= n:
                return rows
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a synthetic eval set from the KB.")
    ap.add_argument("--out", default="evals/datasets/synthetic_qa.jsonl")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--backend", default="stub", choices=["stub", "gemini", "vertex"])
    args = ap.parse_args()
    rows = generate(args.n, backend_name=args.backend)
    path = write_jsonl(args.out, rows)
    print(f"Wrote {len(rows)} synthetic cases -> {path}")


if __name__ == "__main__":
    main()
