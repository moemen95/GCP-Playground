"""Computation (lexical) + tool-call metrics on the Vertex Gen AI Evaluation Service.

These metrics are deterministic string/structure comparisons — no autorater, no
LLM judge, no cost. They are the cheapest signal in the gate and run first.

Text-comparison metric ids (``vertexai.types.Metric(name=...)``)
================================================================
* ``exact_match``   — 1.0 iff ``response`` equals ``reference`` exactly, else 0.0.
* ``bleu``          — BLEU n-gram precision of ``response`` vs ``reference``.
* ``rouge_1``       — ROUGE unigram overlap.
* ``rouge_2``       — ROUGE bigram overlap.
* ``rouge_l``       — ROUGE longest-common-subsequence (sentence level).
* ``rouge_l_sum``   — ROUGE-L computed summary-level (split on newlines).

Tool-call metric ids (operate on the JSON tool-call columns, not free text)
===========================================================================
* ``tool_call_valid``         — 1.0 iff the response is a *syntactically valid*
                                tool call (parseable JSON of the expected shape).
* ``tool_name_match``         — 1.0 iff predicted tool *name* == reference name.
* ``tool_parameter_key_match``— fraction of reference argument *keys* present.
* ``tool_parameter_kv_match`` — fraction of reference argument *key=value* pairs
                                matched exactly.

Dataset shape
=============
Text metrics need ``prompt`` / ``response`` / ``reference`` columns. Tool-call
metrics need ``reference`` and ``response`` columns where **each cell is a JSON
string** of the form::

    {"content": "", "tool_calls": [{"name": "...", "arguments": {...}}]}

The ``content`` field carries any natural-language part of the turn; ``tool_calls``
is the list of structured calls the agent emitted.
"""
from __future__ import annotations

import json
import os

from evals.common import MetricResult, COMPUTATION, TOOL_USE

from ._client import get_genai_client, get_vertex_types, require_vertex

# Metric ids grouped by family — used both to build the request and to tag results.
TEXT_METRIC_IDS = ("exact_match", "bleu", "rouge_1", "rouge_2", "rouge_l", "rouge_l_sum")
TOOL_METRIC_IDS = (
    "tool_call_valid",
    "tool_name_match",
    "tool_parameter_key_match",
    "tool_parameter_kv_match",
)


def _tool_call_cell(name: str, arguments: dict, content: str = "") -> str:
    """Serialize one tool call into the JSON-string cell the service expects."""
    return json.dumps({"content": content, "tool_calls": [{"name": name, "arguments": arguments}]})


def build_text_dataset() -> list[dict]:
    """Card-domain rows for the text-comparison metrics."""
    return [
        {
            "prompt": "What is the rental car insurance limit on the Tangerine World Mastercard?",
            "response": "The Tangerine World Mastercard covers rental cars up to 48 consecutive days.",
            "reference": "The Tangerine World Mastercard's rental car CDW covers up to 48 consecutive days.",
        },
        {
            "prompt": "Does the Money-Back card include mobile device insurance?",
            "response": "No, mobile device insurance is a Tangerine World Mastercard benefit.",
            "reference": "Mobile device insurance is offered on the Tangerine World Mastercard, not the Money-Back card.",
        },
    ]


def build_tool_call_dataset() -> list[dict]:
    """Card-domain rows for the tool-call metrics.

    Each ``reference`` / ``response`` cell is the JSON tool-call envelope. The
    reference encodes the *correct* call; the response is what the agent emitted —
    here deliberately mixing exact matches and a wrong-argument case so the metrics
    produce non-trivial scores.
    """
    return [
        {
            "prompt": "Tell me about rental car coverage on the World Mastercard.",
            # Reference: get_benefit_details(card_name="Tangerine World Mastercard",
            #                                benefit_category="rental_car_insurance")
            "reference": _tool_call_cell(
                "get_benefit_details",
                {"card_name": "Tangerine World Mastercard", "benefit_category": "rental_car_insurance"},
            ),
            "response": _tool_call_cell(
                "get_benefit_details",
                {"card_name": "Tangerine World Mastercard", "benefit_category": "rental_car_insurance"},
            ),
        },
        {
            "prompt": "Which card is best for groceries?",
            "reference": _tool_call_cell("find_cards_for_category", {"spend_category": "groceries"}),
            # Wrong argument value -> key matches, kv does not.
            "response": _tool_call_cell("find_cards_for_category", {"spend_category": "gas"}),
        },
    ]


def run_computation_metrics() -> list[MetricResult]:
    """Run text + tool-call computation metrics via the new GenAI client.

    New-client call shape::

        from vertexai import types
        client.evals.evaluate(
            dataset=text_df,
            metrics=[types.Metric(name="exact_match"),
                     types.Metric(name="bleu"),
                     types.Metric(name="rouge_1"), ...],
        )

        client.evals.evaluate(
            dataset=tool_df,
            metrics=[types.Metric(name="tool_call_valid"),
                     types.Metric(name="tool_name_match"), ...],
        )

    Returns one :class:`MetricResult` per metric (``layer="vertex"``). Raises the
    shared :class:`RuntimeError` offline.
    """
    require_vertex()
    client = get_genai_client()
    types = get_vertex_types()

    results: list[MetricResult] = []

    text_df = build_text_dataset()
    text_eval = client.evals.evaluate(
        dataset=text_df,
        metrics=[types.Metric(name=mid) for mid in TEXT_METRIC_IDS],
    )
    results.extend(_collect(text_eval, TEXT_METRIC_IDS, COMPUTATION, n=len(text_df)))

    tool_df = build_tool_call_dataset()
    tool_eval = client.evals.evaluate(
        dataset=tool_df,
        metrics=[types.Metric(name=mid) for mid in TOOL_METRIC_IDS],
    )
    results.extend(_collect(tool_eval, TOOL_METRIC_IDS, TOOL_USE, n=len(tool_df)))

    return results


def _collect(eval_result, metric_ids, family: str, *, n: int) -> list[MetricResult]:
    """Pull aggregate scores out of an EvaluationResult into MetricResults.

    The SDK exposes per-metric aggregates on ``eval_result.summary_metrics`` (a list
    of ``{"metric_name": ..., "mean_score": ...}``-like records). We defensively look
    the score up by id so this survives minor SDK shape changes.
    """
    summary = getattr(eval_result, "summary_metrics", None) or {}
    out: list[MetricResult] = []
    for mid in metric_ids:
        score = _lookup_score(summary, mid)
        out.append(
            MetricResult(
                name=mid,
                layer="vertex",
                family=family,
                score=float(score),
                uses_llm_judge=False,
                n=n,
                details={"source": "client.evals.evaluate"},
            )
        )
    return out


def _lookup_score(summary, metric_id: str) -> float:
    if isinstance(summary, dict):
        val = summary.get(metric_id) or summary.get(f"{metric_id}/mean")
        return float(val) if val is not None else 0.0
    for rec in summary:  # list of records
        name = getattr(rec, "metric_name", None) or (rec.get("metric_name") if isinstance(rec, dict) else None)
        if name == metric_id:
            score = getattr(rec, "mean_score", None)
            if score is None and isinstance(rec, dict):
                score = rec.get("mean_score")
            return float(score) if score is not None else 0.0
    return 0.0


if __name__ == "__main__":  # pragma: no cover - manual demo
    print("evals.vertex.computation_metrics — offline import OK")
    print("Text metrics:", TEXT_METRIC_IDS)
    print("Tool metrics:", TOOL_METRIC_IDS)
    print("\nSample tool-call cell:")
    print(build_tool_call_dataset()[0]["reference"])
    try:
        for r in run_computation_metrics():
            print(r.to_dict())
    except RuntimeError as exc:
        print("Guard raised (expected offline):", exc)
