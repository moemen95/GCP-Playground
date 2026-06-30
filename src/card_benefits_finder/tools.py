"""Deterministic tools for the benefits-finder agent.

Each tool returns a plain ``dict`` with a ``status`` key (``"ok"`` | ``"error"``)
so both the agent and the trajectory evaluators can reason about outcomes. The
tools are pure functions over ``cards.json`` — no network, fully reproducible —
which is what lets the groundedness metrics have a ground truth.
"""
from __future__ import annotations

import json
from functools import lru_cache

from .config import CARDS_JSON
from .knowledge.retriever import context_for


@lru_cache(maxsize=1)
def _catalog() -> dict:
    return json.loads(CARDS_JSON.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _cards_by_id() -> dict[str, dict]:
    return {c["card_id"]: c for c in _catalog()["cards"]}


def _resolve_card(card_name: str) -> dict | None:
    """Fuzzy-resolve a user-supplied card name to a catalog card."""
    if not card_name:
        return None
    needle = card_name.strip().lower()
    cards = _catalog()["cards"]
    for c in cards:
        if needle == c["card_id"] or needle == c["name"].lower():
            return c
    # token-overlap fallback ("world card" -> Tangerine World Mastercard)
    needle_tokens = set(needle.replace("-", " ").split())
    best, best_score = None, 0
    for c in cards:
        hay = f"{c['name']} {c['card_id']}".lower().replace("-", " ")
        score = len(needle_tokens & set(hay.split()))
        if score > best_score:
            best, best_score = c, score
    return best if best_score > 0 else None


def list_cards() -> dict:
    """List the credit cards this assistant knows about.

    Returns:
        dict with 'status' and 'cards' (list of {card_id, name, tier, annual_fee}).
    """
    cards = [
        {
            "card_id": c["card_id"],
            "name": c["name"],
            "tier": c["tier"],
            "annual_fee": c["fees"]["annual_fee"],
        }
        for c in _catalog()["cards"]
    ]
    return {"status": "ok", "cards": cards}


def lookup_card_benefits(card_name: str) -> dict:
    """Return the full benefit catalog for a named credit card.

    Args:
        card_name: The card to look up, e.g. "Tangerine World Mastercard".
    Returns:
        dict with 'status' and either 'card'/'benefits' or 'error_message'.
    """
    card = _resolve_card(card_name)
    if card is None:
        return {
            "status": "error",
            "error_message": f"Unknown card {card_name!r}. Use list_cards to see options.",
        }
    benefits = [
        {
            "benefit_id": b["benefit_id"],
            "category": b["category"],
            "title": b["title"],
            "summary": b["summary"],
        }
        for b in card["benefits"]
    ]
    return {"status": "ok", "card": card["name"], "card_id": card["card_id"], "benefits": benefits}


def get_benefit_details(card_name: str, benefit_category: str) -> dict:
    """Return detailed terms for one benefit category on a card.

    Args:
        card_name: e.g. "Tangerine World Mastercard".
        benefit_category: e.g. "rental_car_insurance", "mobile_device_insurance",
            "travel_medical_insurance", "purchase_protection", "extended_warranty".
    Returns:
        dict with 'status' and either the benefit terms + a grounding 'context'
        citation, or 'error_message'.
    """
    card = _resolve_card(card_name)
    if card is None:
        return {"status": "error", "error_message": f"Unknown card {card_name!r}."}
    cat = (benefit_category or "").strip().lower().replace(" ", "_").replace("-", "_")
    for b in card["benefits"]:
        if b["category"] == cat or cat in b["category"] or cat in b["title"].lower():
            return {
                "status": "ok",
                "card": card["name"],
                "benefit": b,
                "context": context_for(f"{card['name']} {b['title']}"),
            }
    available = sorted({b["category"] for b in card["benefits"]})
    return {
        "status": "error",
        "error_message": f"{card['name']} has no '{benefit_category}' benefit.",
        "available_categories": available,
    }


def find_cards_for_category(spend_category: str) -> dict:
    """Recommend cards offering the best rewards for a spending category.

    Args:
        spend_category: e.g. "groceries", "gas", "restaurants", "travel".
    Returns:
        dict with 'status' and 'recommendations' (card + rate for that category).
    """
    cat = (spend_category or "").strip().lower().replace(" ", "-")
    recs = []
    for c in _catalog()["cards"]:
        r = c["rewards"]
        if cat in r["selectable_categories"]:
            rate = r["selected_category_rate"]
            note = "earns the boosted rate if you select this as one of your 2% categories"
        else:
            rate = r["base_rate"]
            note = "earns the base rate (not a selectable bonus category)"
        recs.append({"card_id": c["card_id"], "card": c["name"], "rate": rate, "note": note})
    recs.sort(key=lambda x: x["rate"], reverse=True)
    return {"status": "ok", "spend_category": spend_category, "recommendations": recs}


def check_eligibility(card_name: str, annual_income: float, age: int) -> dict:
    """Check whether an applicant meets a card's stated eligibility requirements.

    Args:
        card_name: e.g. "Tangerine World Mastercard".
        annual_income: Applicant personal annual income in CAD.
        age: Applicant age in years.
    Returns:
        dict with 'status', 'eligible' (bool), and 'reasons'.
    """
    card = _resolve_card(card_name)
    if card is None:
        return {"status": "error", "error_message": f"Unknown card {card_name!r}."}
    elig = card["eligibility"]
    reasons = []
    ok = True
    if age < elig.get("min_age", 0):
        ok = False
        reasons.append(f"Minimum age is {elig['min_age']}.")
    min_income = elig.get("min_annual_income", 0.0)
    if annual_income < min_income:
        ok = False
        reasons.append(f"Minimum personal income is CAD {min_income:,.0f}.")
    if ok:
        reasons.append("Meets the stated minimum age and income requirements.")
    return {"status": "ok", "card": card["name"], "eligible": ok, "reasons": reasons}


# The tool set exported to the agent (and referenced by trajectory eval cases).
ALL_TOOLS = [
    list_cards,
    lookup_card_benefits,
    get_benefit_details,
    find_cards_for_category,
    check_eligibility,
]
