"""Agent / trajectory evaluation on the Vertex Gen AI Evaluation Service.

These metrics score the *sequence of tool calls* an agent made (its trajectory),
not just the final text. They compare a ``predicted_trajectory`` against a
``reference_trajectory``.

Trajectory metric ids
====================
* ``trajectory_exact_match``      — predicted == reference (same calls, same order).
* ``trajectory_in_order_match``   — all reference calls appear, in order (extras OK).
* ``trajectory_any_order_match``  — all reference calls appear, any order.
* ``trajectory_precision``        — fraction of predicted calls that are in reference.
* ``trajectory_recall``           — fraction of reference calls that were predicted.
* ``trajectory_single_tool_use``  — did the agent use a specific named tool at all?

Dataset shape
============
* ``prompt``                — the user query.
* ``reference_trajectory``  — gold list of ``{"tool_name": ..., "tool_input": {...}}``.
* ``predicted_trajectory``  — the agent's actual calls (same shape). When you pass a
                              live agent via ``runnable=``, the service **auto-captures
                              this column** for you.
* ``response``              — the agent's final text (also scored, see below).
* ``reference`` (optional)  — gold final answer for response-quality metrics.

The ``runnable=`` path
=====================
Instead of precomputing ``predicted_trajectory`` / ``response``, you can hand the
service a callable agent::

    client.evals.evaluate(
        dataset=df,                       # only needs prompt (+ references)
        metrics=[types.Metric(name="trajectory_in_order_match"),
                 types.PrebuiltMetric.QUESTION_ANSWERING_QUALITY],
        runnable=build_runnable(),        # ADK / LangGraph / custom callable
    )

The service invokes the runnable per prompt, captures its trajectory and final
response, and scores **both trajectory and response in ONE ``evaluate()`` call**.

This module wraps the offline twin (``local_agent.run_turn``) into such a callable so
the trajectory shape is realistic without needing the live ADK agent or creds.
"""
from __future__ import annotations

from evals.common import MetricResult, TRAJECTORY

from ._client import get_genai_client, get_vertex_types, require_vertex

TRAJECTORY_METRIC_IDS = (
    "trajectory_exact_match",
    "trajectory_in_order_match",
    "trajectory_any_order_match",
    "trajectory_precision",
    "trajectory_recall",
    "trajectory_single_tool_use",
)


def build_runnable():
    """Wrap ``local_agent.run_turn`` into a callable the service can drive.

    The returned function takes a prompt string and returns the dict shape the Vertex
    agent-eval ``runnable=`` path expects: a ``predicted_trajectory`` (list of
    ``{"tool_name", "tool_input"}``) plus the final ``response`` text. This mirrors
    what an ADK / LangGraph ``runnable`` would emit, so trajectory metrics see a
    realistic capture.

    Imports the SUT lazily so this module stays importable without the agent package.
    """
    from src.card_benefits_finder.local_agent import run_turn

    def runnable(prompt: str) -> dict:
        turn = run_turn(prompt)
        predicted = [{"tool_name": tc["name"], "tool_input": tc["args"]} for tc in turn.trajectory]
        return {
            "response": turn.final_response,
            "predicted_trajectory": predicted,
            "context": turn.context,
        }

    return runnable


def build_dataset() -> list[dict]:
    """Card-domain rows with reference trajectories (predicted captured by runnable)."""
    return [
        {
            "prompt": "Tell me about rental car coverage on the World Mastercard.",
            "reference_trajectory": [
                {
                    "tool_name": "get_benefit_details",
                    "tool_input": {
                        "card_name": "Tangerine World Mastercard",
                        "benefit_category": "rental_car_insurance",
                    },
                }
            ],
            "reference": (
                "The Tangerine World Mastercard covers rental car CDW for up to 48 "
                "consecutive days."
            ),
        },
        {
            "prompt": "Which card is best for groceries?",
            "reference_trajectory": [
                {"tool_name": "find_cards_for_category", "tool_input": {"spend_category": "groceries"}}
            ],
            "reference": "Both cards earn up to 2% cash back on groceries if selected as a bonus category.",
        },
    ]


def run_trajectory_eval() -> list[MetricResult]:
    """Run trajectory + response metrics in a single ``evaluate()`` call via ``runnable=``.

    The runnable (offline twin) produces ``predicted_trajectory`` and ``response``;
    the service scores the trajectory metrics against ``reference_trajectory`` and
    (here) ``question_answering_quality`` against ``reference`` — all in one call::

        client.evals.evaluate(
            dataset=df,
            metrics=[types.Metric(name="trajectory_in_order_match"), ...,
                     types.PrebuiltMetric.QUESTION_ANSWERING_QUALITY],
            runnable=build_runnable(),
        )

    Returns one :class:`MetricResult` per trajectory metric. Raises the shared
    :class:`RuntimeError` offline.
    """
    require_vertex()
    client = get_genai_client()
    types = get_vertex_types()

    metrics = [types.Metric(name=mid) for mid in TRAJECTORY_METRIC_IDS]
    # Response quality scored in the SAME call as trajectory.
    qa = getattr(types.PrebuiltMetric, "QUESTION_ANSWERING_QUALITY", None)
    if qa is not None:
        metrics.append(qa)

    df = build_dataset()
    eval_result = client.evals.evaluate(dataset=df, metrics=metrics, runnable=build_runnable())
    summary = getattr(eval_result, "summary_metrics", None) or {}

    results: list[MetricResult] = []
    for mid in TRAJECTORY_METRIC_IDS:
        results.append(
            MetricResult(
                name=mid,
                layer="vertex",
                family=TRAJECTORY,
                score=float(_lookup_score(summary, mid)),
                uses_llm_judge=False,  # trajectory matching is deterministic
                n=len(df),
                details={"captured_via": "runnable"},
            )
        )
    return results


def _lookup_score(summary, metric_id: str) -> float:
    if isinstance(summary, dict):
        val = summary.get(metric_id) or summary.get(f"{metric_id}/mean")
        return float(val) if val is not None else 0.0
    for rec in summary:
        name = getattr(rec, "metric_name", None) or (rec.get("metric_name") if isinstance(rec, dict) else None)
        if name == metric_id:
            score = getattr(rec, "mean_score", None)
            if score is None and isinstance(rec, dict):
                score = rec.get("mean_score")
            return float(score) if score is not None else 0.0
    return 0.0


if __name__ == "__main__":  # pragma: no cover - manual demo
    print("evals.vertex.trajectory_eval — offline import OK")
    print("Metrics:", TRAJECTORY_METRIC_IDS)
    # The runnable itself works fully offline (it only needs the local twin).
    try:
        rn = build_runnable()
        print("Sample runnable output:", rn("Tell me about rental car coverage on the World Mastercard."))
    except Exception as exc:  # noqa: BLE001 - demo only
        print("Runnable demo skipped:", exc)
    try:
        for r in run_trajectory_eval():
            print(r.to_dict())
    except RuntimeError as exc:
        print("Guard raised (expected offline):", exc)
