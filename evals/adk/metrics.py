"""Custom ADK eval metric: ``benefit_citation_score``.

A domain-specific groundedness proxy for the Tangerine Card Benefits Finder. It
rewards a model's *final response* for citing the concrete terms a good benefits
answer should contain: a coverage **limit**, an **eligibility** condition, and a
**coverage window** (a time period such as "90 days").

ADK invokes a custom metric through the callable registered in the eval config's
``custom_metrics`` block (see ``eval_config.json``) with this exact signature::

    benefit_citation_score(
        eval_metric,            # google.adk.evaluation.eval_metrics.EvalMetric
        actual_invocations,     # list[Invocation] produced by the agent run
        expected_invocations,   # list[Invocation] from the eval dataset
        conversation_scenario,  # optional scenario context (may be None)
    ) -> google.adk.evaluation.evaluator.EvaluationResult

The whole module is import-guarded so it loads fine *without* ``google-adk``
installed. The scoring logic lives in a pure-python helper
(:func:`score_benefit_citation_text`) that needs no ADK types, so it is unit
testable offline. The ADK-typed entry point is a thin wrapper around it.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

# --- Offline-safe ADK imports -------------------------------------------------
# These types only exist when google-adk is installed. We import them lazily /
# defensively so this module can still be imported (and the pure-python helper
# tested) in an environment without ADK. The decorated function is exposed
# regardless; it references the ADK types only at call time.
try:  # pragma: no cover - exercised only when google-adk is installed
    from google.adk.evaluation.evaluator import (  # type: ignore
        EvaluationResult,
        PerInvocationResult,
    )
    from google.adk.evaluation.eval_metrics import (  # type: ignore
        EvalMetric,
        EvalStatus,
    )

    _ADK_AVAILABLE = True
except Exception:  # pragma: no cover - the offline path
    EvaluationResult = Any  # type: ignore
    PerInvocationResult = Any  # type: ignore
    EvalMetric = Any  # type: ignore
    EvalStatus = Any  # type: ignore
    _ADK_AVAILABLE = False


# --- Pure-python scoring (no ADK required) ------------------------------------

# Cue phrases that signal each of the three citation dimensions we care about.
_LIMIT_CUES = (
    "limit",
    "up to",
    "maximum",
    "max ",
    "per claim",
    "lifetime",
    "cad",
    "$",
)
_ELIGIBILITY_CUES = (
    "eligib",
    "eligible",
    "to qualify",
    "qualify",
    "must be purchased",
    "must be charged",
    "decline",
    "charged to the card",
    "purchased on the card",
    "purchased entirely on the card",
)
_WINDOW_CUES = (
    "day",
    "days",
    "year",
    "months",
    "month",
    "consecutive",
    "first 15",
    "90 day",
    "31 day",
    "coverage period",
    "coverage window",
    "warranty",
)

# A duration like "90 days", "31 consecutive days", "one additional year".
_DURATION_RE = re.compile(
    r"\b(\d{1,4}|one|two|three|several)\s*"
    r"(?:additional\s+|consecutive\s+)?(day|days|year|years|month|months|week|weeks)\b",
    re.IGNORECASE,
)


def _final_text(invocation: Any) -> str:
    """Best-effort extraction of the model final-response text from an Invocation.

    Works on ADK ``Invocation`` objects, plain dicts shaped like the .test.json
    schema, and anything exposing ``final_response.parts[*].text``.
    """
    if invocation is None:
        return ""

    # dict shape (matches the .test.json / evalset schema)
    if isinstance(invocation, dict):
        final = invocation.get("final_response") or {}
    else:
        final = getattr(invocation, "final_response", None)

    if final is None:
        return ""

    parts = final.get("parts") if isinstance(final, dict) else getattr(final, "parts", None)
    if not parts:
        # final_response might itself be a string
        if isinstance(final, str):
            return final
        return ""

    texts: list[str] = []
    for part in parts:
        if isinstance(part, dict):
            t = part.get("text")
        else:
            t = getattr(part, "text", None)
        if t:
            texts.append(str(t))
    return " ".join(texts)


def _contains_any(text: str, cues: Iterable[str]) -> bool:
    return any(cue in text for cue in cues)


def score_benefit_citation_text(text: str) -> float:
    """Score one final-response string in ``[0.0, 1.0]``.

    Credit is awarded across three equally weighted dimensions a well-grounded
    benefit answer should include:

    * a coverage **limit** (e.g. "up to CAD 1,000"),
    * an **eligibility** condition (e.g. "must be charged to the card"),
    * a **coverage window** / time period (e.g. "90 days", "one additional year").

    Pure python — no ADK, no network — so it is deterministic and unit testable.
    """
    if not text:
        return 0.0
    low = text.lower()

    has_limit = _contains_any(low, _LIMIT_CUES)
    has_eligibility = _contains_any(low, _ELIGIBILITY_CUES)
    has_window = bool(_DURATION_RE.search(low)) or _contains_any(low, _WINDOW_CUES)

    hits = sum((has_limit, has_eligibility, has_window))
    return round(hits / 3.0, 4)


def score_invocations(invocations: Iterable[Any]) -> list[float]:
    """Per-invocation scores for an iterable of Invocations (ADK or dict)."""
    return [score_benefit_citation_text(_final_text(inv)) for inv in invocations]


# --- ADK metric entry point ---------------------------------------------------

def benefit_citation_score(
    eval_metric: Any,
    actual_invocations: Any,
    expected_invocations: Any = None,
    conversation_scenario: Any = None,
) -> Any:
    """ADK custom-metric entry point. Returns an ``EvaluationResult``.

    The pass/fail threshold is read from ``eval_metric.threshold`` when present
    (configured in ``eval_config.json`` as ``0.7``); we default to ``0.7`` when
    ADK isn't supplying a metric object.
    """
    actual = list(actual_invocations or [])
    threshold = getattr(eval_metric, "threshold", None)
    if threshold is None:
        threshold = 0.7

    scores = score_invocations(actual)
    overall = round(sum(scores) / len(scores), 4) if scores else 0.0

    if not _ADK_AVAILABLE:  # pragma: no cover - offline fallback
        # Return a lightweight, dict-shaped result so the logic remains usable
        # (and inspectable) without google-adk installed.
        return {
            "overall_score": overall,
            "overall_eval_status": "PASSED" if overall >= threshold else "FAILED",
            "per_invocation_results": [
                {
                    "score": s,
                    "eval_status": "PASSED" if s >= threshold else "FAILED",
                }
                for s in scores
            ],
        }

    def _status(value: float):
        return EvalStatus.PASSED if value >= threshold else EvalStatus.FAILED

    per_invocation_results = []
    for actual_inv, score in zip(actual, scores):
        # Pair with the matching expected invocation when available.
        per_invocation_results.append(
            PerInvocationResult(
                actual_invocation=actual_inv,
                score=score,
                eval_status=_status(score),
            )
        )

    return EvaluationResult(
        overall_score=overall,
        overall_eval_status=_status(overall),
        per_invocation_results=per_invocation_results,
    )
