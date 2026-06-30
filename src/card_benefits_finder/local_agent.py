"""A deterministic, dependency-free twin of the ADK agent.

This reproduces the agent's *behaviour* (intent -> tool call -> grounded answer)
with transparent rules instead of an LLM, so the eval layers can produce real
trajectories, responses, and grounding context with zero credentials. It is the
System-Under-Test for the offline path; the live ADK agent (``agent.py``) is the
SUT for the GCP path. Both share the same tools and knowledge base.

It is intentionally simple — and intentionally *imperfect* (e.g. it can pick a
slightly wrong benefit category) — so the evaluators have non-trivial signal to
score rather than a guaranteed 100%.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import tools
from .knowledge.retriever import context_for

# ---- intent cues -----------------------------------------------------------
_OUT_OF_SCOPE = (
    "mortgage", "invest", "stock", "crypto", "rrsp", "tfsa", "tax", "legal advice",
    "my balance", "statement", "my account", "amex", "american express",
    "visa infinite", "scotiabank", "rbc", "td ", "cibc", "should i buy",
    "should i move", "refinance", "credit score will", "lawsuit",
)
_BENEFIT_CATEGORY_CUES = {
    "rental_car_insurance": ("rental car", "car rental", "rent a car", "cdw", "ldw", "collision"),
    "mobile_device_insurance": ("mobile", "phone", "cell", "device insurance", "smartphone"),
    "travel_medical_insurance": ("travel medical", "medical emergency", "health insurance", "emergency medical"),
    "purchase_protection": ("purchase protection", "purchase assurance", "stolen", "damaged", "theft of"),
    "extended_warranty": ("extended warranty", "warranty", "manufacturer warranty"),
    "travel_perk": ("wi-fi", "wifi", "boingo", "lounge", "airport"),
    "fraud_protection": ("fraud", "unauthorized", "zero liability"),
}
_CARD_CUES = {
    "tangerine-world": ("world", "premium", "travel card"),
    "tangerine-money-back": ("money-back", "money back", "moneyback", "cash back card", "standard"),
}
_SPEND_CUES = (
    "groceries", "grocery", "gas", "restaurants", "dining", "furniture",
    "hotel", "entertainment", "drug store", "transit", "parking",
)


@dataclass
class ToolCall:
    name: str
    args: dict
    result: dict | None = None


@dataclass
class AgentTurn:
    query: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_response: str = ""
    context: str = ""  # grounding text used to support the answer
    refused: bool = False

    @property
    def trajectory(self) -> list[dict]:
        return [{"name": tc.name, "args": tc.args} for tc in self.tool_calls]


def _mentioned_card(q: str) -> str | None:
    for card_id, cues in _CARD_CUES.items():
        if any(c in q for c in cues):
            return card_id
    return None


def _mentioned_category(q: str) -> str | None:
    for cat, cues in _BENEFIT_CATEGORY_CUES.items():
        if any(c in q for c in cues):
            return cat
    return None


def _extract_income(q: str) -> float | None:
    m = re.search(r"\$?\s*([0-9]{2,3}(?:,?[0-9]{3})+|[0-9]+k)", q)
    if not m:
        return None
    raw = m.group(1).lower().replace(",", "")
    if raw.endswith("k"):
        return float(raw[:-1]) * 1000
    return float(raw)


def _extract_age(q: str) -> int | None:
    m = re.search(r"\b(1[89]|[2-9][0-9])\s*(?:years|yrs|yo|year)?\b", q)
    return int(m.group(1)) if m else None


def run_turn(query: str) -> AgentTurn:
    """Route a single user query to a tool call and a grounded answer."""
    q = query.lower()
    turn = AgentTurn(query=query)

    # 1. Scope guard.
    if any(cue in q for cue in _OUT_OF_SCOPE):
        turn.refused = True
        turn.final_response = (
            "I can only help with the published benefits, rewards, fees, and "
            "eligibility of the Tangerine Money-Back and Tangerine World "
            "Mastercards. I can't give personalized financial advice or access "
            "account-specific information. Is there a card benefit I can explain?"
        )
        return turn

    card_id = _mentioned_card(q)
    card_name = tools._cards_by_id().get(card_id, {}).get("name") if card_id else None

    # 2. Eligibility intent.
    if any(w in q for w in ("eligible", "qualify", "qualifie", "income require", "approved")):
        income = _extract_income(q) or 0.0
        age = _extract_age(q) or 18
        name = card_name or "Tangerine World Mastercard"
        res = tools.check_eligibility(name, income, age)
        turn.tool_calls.append(ToolCall("check_eligibility",
                                        {"card_name": name, "annual_income": income, "age": age}, res))
        # Ground in retrieved KB + the tool's own reasons (agents ground in tool output too).
        turn.context = context_for(f"{name} eligibility income") + " " + " ".join(res.get("reasons", []))
        if res.get("status") == "ok":
            verdict = "appear to meet" if res["eligible"] else "may not meet"
            turn.final_response = (
                f"Based on the stated requirements, you {verdict} the eligibility "
                f"for the {res['card']}. " + " ".join(res["reasons"])
            )
        else:
            turn.final_response = res.get("error_message", "I couldn't check that card.")
        return turn

    # 3. Best-card-for-spend intent.
    if any(w in q for w in ("which card", "best card", "best for", "more rewards", "most cash back")) \
            and any(s in q for s in _SPEND_CUES):
        cat = next((s for s in _SPEND_CUES if s in q), "groceries")
        cat = {"grocery": "groceries", "dining": "restaurants", "transit": "public-transportation-parking"}.get(cat, cat)
        res = tools.find_cards_for_category(cat)
        turn.tool_calls.append(ToolCall("find_cards_for_category", {"spend_category": cat}, res))
        turn.context = context_for(f"reward categories {cat}")
        top = res["recommendations"][0]
        turn.final_response = (
            f"For {cat}, both cards earn up to {top['rate']*100:.1f}% cash back if "
            f"you select it as one of your 2% bonus categories; otherwise spend "
            f"earns the 0.5% base rate."
        )
        return turn

    # 4. Specific benefit detail.
    category = _mentioned_category(q)
    if category:
        name = card_name or "Tangerine World Mastercard"
        res = tools.get_benefit_details(name, category)
        turn.tool_calls.append(ToolCall("get_benefit_details",
                                        {"card_name": name, "benefit_category": category}, res))
        if res.get("status") == "ok":
            b = res["benefit"]
            turn.context = res.get("context", "")
            turn.final_response = (
                f"{res['card']} — {b['title']}: {b['summary']} Limit: {b['limit']} "
                f"Eligibility: {b['eligibility']}"
            )
        else:
            turn.context = context_for(f"{name} benefits")
            turn.final_response = res.get("error_message", "I couldn't find that benefit.")
        return turn

    # 5. List a card's benefits.
    if card_name and any(w in q for w in ("benefit", "perk", "coverage", "insurance", "offer")):
        res = tools.lookup_card_benefits(card_name)
        turn.tool_calls.append(ToolCall("lookup_card_benefits", {"card_name": card_name}, res))
        titles = [b["title"] for b in res.get("benefits", [])]
        # Ground in the actual benefit sections so the listed titles are citable.
        turn.context = context_for(f"{card_name} " + " ".join(titles), k=6)
        turn.final_response = f"The {res.get('card', card_name)} includes: {', '.join(titles)}."
        return turn

    # 6. What cards exist.
    if any(w in q for w in ("what cards", "which cards", "list cards", "cards do you")):
        res = tools.list_cards()
        turn.tool_calls.append(ToolCall("list_cards", {}, res))
        names = ", ".join(c["name"] for c in res["cards"])
        turn.context = f"Supported cards: {names}."
        turn.final_response = f"I can help with these cards: {names}."
        return turn

    # 7. Fallback: list benefits of the most-likely card.
    name = card_name or "Tangerine Money-Back Mastercard"
    res = tools.lookup_card_benefits(name)
    turn.tool_calls.append(ToolCall("lookup_card_benefits", {"card_name": name}, res))
    titles = [b["title"] for b in res.get("benefits", [])]
    turn.context = context_for(f"{name} " + " ".join(titles), k=6)
    turn.final_response = (
        f"The {res.get('card', name)} includes: {', '.join(titles)}. "
        "Ask me about any one for details."
    )
    return turn
