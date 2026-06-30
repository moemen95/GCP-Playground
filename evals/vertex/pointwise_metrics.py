"""Model-based (autorater) pointwise metrics on the Vertex Gen AI Evaluation Service.

Unlike the computation metrics, these use a Gemini autorater to *judge* a single
response against criteria. Each returns a numeric score **and a rationale**.

Pointwise metric ids
====================
(scored by an LLM autorater; ``uses_llm_judge=True``)

* ``groundedness``                    — is the response supported by ``context``?  (0/1)
* ``question_answering_quality``      — overall QA answer quality.                (1-5)
* ``question_answering_relevance``    — does it address the question?             (1-5)
* ``question_answering_helpfulness``  — is it useful/actionable?                  (1-5)
* ``question_answering_correctness``  — is it factually correct vs reference?     (0/1)
* ``instruction_following``           — did it obey the prompt instructions?      (1-5)
* ``coherence``                       — logically organized / consistent?         (1-5)
* ``fluency``                         — grammatical/natural language?             (1-5)
* ``safety``                          — free of harmful content?                  (0/1)
* ``verbosity``                       — appropriately concise (not too long)?     (1-5)
* ``fulfillment``                     — does it fulfill the user's intent?        (1-5)

Score ranges
============
Most metrics return an integer 1..5 (higher = better). The exceptions return a
binary 0/1: ``groundedness``, ``safety``, and ``question_answering_correctness``.
We normalize to 0..1 in :class:`MetricResult.score` so the gate sees a uniform scale.

Dataset shape
============
``prompt`` (the question/instruction), ``response`` (what to judge), ``context``
(grounding text; needed by ``groundedness``) and ``reference`` (gold answer; needed
by ``question_answering_correctness``).

SDK surfaces
===========
* **New client (PRIMARY)** — pass prebuilt metrics::

      from vertexai import types
      client.evals.evaluate(
          dataset=df,
          metrics=[types.PrebuiltMetric.GROUNDEDNESS,
                   types.PrebuiltMetric.QUESTION_ANSWERING_QUALITY, ...],
      )

  Equivalently the metric-name strings (``types.Metric(name="groundedness")``) are
  accepted for the model-based ids too.

* **Legacy (DEPRECATED)** — ``MetricPromptTemplateExamples.Pointwise.GROUNDEDNESS``
  passed to a ``PointwiseMetric`` / ``EvalTask`` (see ``legacy_evaltask.py``).
"""
from __future__ import annotations

from evals.common import MetricResult, POINTWISE, SAFETY

from ._client import get_genai_client, get_vertex_types, require_vertex

# All pointwise autorater metric ids, in canonical order.
POINTWISE_METRIC_IDS = (
    "groundedness",
    "question_answering_quality",
    "question_answering_relevance",
    "question_answering_helpfulness",
    "question_answering_correctness",
    "instruction_following",
    "coherence",
    "fluency",
    "safety",
    "verbosity",
    "fulfillment",
)

# Binary 0/1 metrics; everything else is on a 1..5 scale.
_BINARY_METRICS = frozenset({"groundedness", "safety", "question_answering_correctness"})
# Map the prebuilt metric ids to their ``types.PrebuiltMetric`` enum attribute names.
_PREBUILT_ATTR = {mid: mid.upper() for mid in POINTWISE_METRIC_IDS}


def _scale(metric_id: str) -> tuple[int, int]:
    return (0, 1) if metric_id in _BINARY_METRICS else (1, 5)


def build_dataset() -> list[dict]:
    """Card-domain rows with all four columns the pointwise metrics may consume."""
    return [
        {
            "prompt": "Does the Tangerine World Mastercard cover rental cars, and for how long?",
            "response": (
                "Yes. The Tangerine World Mastercard includes rental car collision/loss "
                "damage coverage for rentals up to 48 consecutive days when you charge the "
                "full rental cost to the card."
            ),
            "context": (
                "Tangerine World Mastercard — Auto Rental Collision/Loss Damage Insurance: "
                "covers theft of or damage to a rental vehicle for rentals of up to 48 "
                "consecutive days, provided the full cost is charged to the card."
            ),
            "reference": (
                "The Tangerine World Mastercard covers rental car CDW for up to 48 "
                "consecutive days when the rental is paid in full with the card."
            ),
        },
        {
            "prompt": "What annual fee does the Tangerine Money-Back Mastercard charge?",
            "response": "The Tangerine Money-Back Mastercard has no annual fee.",
            "context": "Tangerine Money-Back Mastercard — Annual fee: $0.",
            "reference": "The Tangerine Money-Back Mastercard has a $0 annual fee.",
        },
    ]


def run_pointwise_metrics() -> list[MetricResult]:
    """Run all pointwise autorater metrics via the new GenAI client.

    Builds the metric list from ``types.PrebuiltMetric.<NAME>`` (falling back to
    ``types.Metric(name=...)`` if an attribute is absent in the installed SDK), then
    issues a single ``client.evals.evaluate`` call. Returns one
    :class:`MetricResult` per metric, normalized to 0..1. Raises the shared
    :class:`RuntimeError` offline.
    """
    require_vertex()
    client = get_genai_client()
    types = get_vertex_types()

    metrics = []
    for mid in POINTWISE_METRIC_IDS:
        prebuilt = getattr(types.PrebuiltMetric, _PREBUILT_ATTR[mid], None)
        metrics.append(prebuilt if prebuilt is not None else types.Metric(name=mid))

    df = build_dataset()
    eval_result = client.evals.evaluate(dataset=df, metrics=metrics)
    summary = getattr(eval_result, "summary_metrics", None) or {}

    results: list[MetricResult] = []
    for mid in POINTWISE_METRIC_IDS:
        raw = _lookup_score(summary, mid)
        lo, hi = _scale(mid)
        norm = (raw - lo) / (hi - lo) if hi != lo else 0.0
        results.append(
            MetricResult(
                name=mid,
                layer="vertex",
                family=SAFETY if mid == "safety" else POINTWISE,
                score=float(max(0.0, min(1.0, norm))),
                uses_llm_judge=True,
                n=len(df),
                details={"raw_score": raw, "scale": [lo, hi], "has_rationale": True},
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
    print("evals.vertex.pointwise_metrics — offline import OK")
    for mid in POINTWISE_METRIC_IDS:
        print(f"  {mid:34s} scale={_scale(mid)}")
    try:
        for r in run_pointwise_metrics():
            print(r.to_dict())
    except RuntimeError as exc:
        print("Guard raised (expected offline):", exc)
