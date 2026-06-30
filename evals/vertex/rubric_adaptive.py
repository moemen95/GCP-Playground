"""Adaptive & managed rubric metrics (new GenAI client REQUIRED).

Rubric metrics decompose "quality" into a checklist of yes/no criteria. Instead of a
single 1-5 score, the autorater verifies each rubric item and reports how many pass.
Rubrics can be **generated adaptively per example** by the service.

This capability is exposed ONLY through the new GenAI client — there is no legacy
``vertexai.evaluation`` equivalent. It is backed by the ``generateInstanceRubrics``
REST method.

Predefined rubric metrics (``vertexai.types.RubricMetric``)
==========================================================
* ``RubricMetric.GENERAL_QUALITY``              — broad multi-dimension quality.
* ``RubricMetric.TEXT_QUALITY``                 — writing quality of free text.
* ``RubricMetric.QUESTION_ANSWERING_QUALITY``   — QA-specific answer quality.
* ``RubricMetric.INSTRUCTION_FOLLOWING``        — adherence to prompt instructions.
* ``RubricMetric.GROUNDING``                    — support of claims by context.

Three usage modes (when to use which)
====================================
1. **Adaptive (inline)** — pass a ``RubricMetric`` straight to ``evaluate()``; the
   service generates rubrics on the fly per example. Easiest; rubrics are not reused.
   Use for exploratory / one-off eval runs.

2. **Pre-generate -> review -> reuse (rubric group)** — call ``generate_rubrics``
   first to materialize a named **rubric group**, eyeball/edit the rubrics, then pass
   ``RubricMetric.X(rubric_group_name=...)`` to ``evaluate``. Use when you want the
   *same* rubrics applied across runs/candidates for a fair, stable comparison, or
   when a human should approve the criteria first.

3. **Static rubrics** — author a fixed rubric set yourself (no generation) and reuse
   it. Use when the criteria are well-known and you want zero autorater variance in
   what is being checked.

Flow for mode (2)
=================
    client.evals.generate_rubrics(
        src=df,
        rubric_group_name="card_benefits_quality",
        metric=types.RubricMetric.GENERAL_QUALITY,
    )                                  # -> generateInstanceRubrics; persists the group
    # ... (optional human review/edit of the generated rubrics) ...
    client.evals.evaluate(
        dataset=df,
        metrics=[types.RubricMetric.GENERAL_QUALITY(rubric_group_name="card_benefits_quality")],
    )
"""
from __future__ import annotations

from evals.common import MetricResult, RUBRIC

from ._client import get_genai_client, get_vertex_types, require_vertex

RUBRIC_METRIC_IDS = (
    "GENERAL_QUALITY",
    "TEXT_QUALITY",
    "QUESTION_ANSWERING_QUALITY",
    "INSTRUCTION_FOLLOWING",
    "GROUNDING",
)
RUBRIC_GROUP_NAME = "card_benefits_quality"


def build_dataset() -> list[dict]:
    """Card-domain rows for rubric evaluation."""
    return [
        {
            "prompt": "Explain the rental car coverage on the Tangerine World Mastercard.",
            "response": (
                "The Tangerine World Mastercard includes auto rental collision/loss damage "
                "insurance covering rentals of up to 48 consecutive days, as long as you "
                "charge the entire rental cost to the card."
            ),
            "context": (
                "Tangerine World Mastercard — Auto Rental Collision/Loss Damage Insurance: "
                "up to 48 consecutive days; full rental cost must be charged to the card."
            ),
        },
        {
            "prompt": "What rewards does the Money-Back Mastercard earn on groceries?",
            "response": (
                "The Tangerine Money-Back Mastercard earns 2% cash back on groceries if you "
                "choose it as one of your bonus categories, and 0.5% otherwise."
            ),
            "context": (
                "Tangerine Money-Back Mastercard — 2% Money-Back Rewards in up to three "
                "chosen categories (e.g. groceries); 0.5% on all other purchases."
            ),
        },
    ]


def generate_rubric_group(rubric_group_name: str = RUBRIC_GROUP_NAME):
    """Mode (2) step 1: pre-generate a reusable rubric group.

    Calls ``client.evals.generate_rubrics`` (REST: ``generateInstanceRubrics``) to
    materialize a named rubric group from the dataset, which you can review/edit
    before reuse. Returns whatever the SDK returns (the generated rubrics handle).
    Raises the shared :class:`RuntimeError` offline.
    """
    require_vertex()
    client = get_genai_client()
    types = get_vertex_types()
    return client.evals.generate_rubrics(
        src=build_dataset(),
        rubric_group_name=rubric_group_name,
        metric=types.RubricMetric.GENERAL_QUALITY,
    )


def run_rubric_eval(rubric_group_name: str | None = RUBRIC_GROUP_NAME) -> list[MetricResult]:
    """Run rubric-based evaluation via the new GenAI client.

    If ``rubric_group_name`` is given, runs mode (2): pre-generate the group, then
    evaluate with ``RubricMetric.GENERAL_QUALITY(rubric_group_name=...)`` so the same
    rubrics are reused. If ``None``, runs mode (1): pass the bare ``RubricMetric`` and
    let the service generate rubrics adaptively/inline.

    Returns one :class:`MetricResult` for the rubric metric. Raises the shared
    :class:`RuntimeError` offline.
    """
    require_vertex()
    client = get_genai_client()
    types = get_vertex_types()

    df = build_dataset()
    if rubric_group_name:
        generate_rubric_group(rubric_group_name)  # mode (2): pre-generate + reuse
        metric = types.RubricMetric.GENERAL_QUALITY(rubric_group_name=rubric_group_name)
        mode = "saved_group"
    else:
        metric = types.RubricMetric.GENERAL_QUALITY  # mode (1): adaptive/inline
        mode = "adaptive"

    eval_result = client.evals.evaluate(dataset=df, metrics=[metric])
    summary = getattr(eval_result, "summary_metrics", None) or {}

    return [
        MetricResult(
            name="general_quality",
            layer="vertex",
            family=RUBRIC,
            score=float(_lookup_score(summary, "general_quality")),
            uses_llm_judge=True,
            n=len(df),
            details={"mode": mode, "rubric_group_name": rubric_group_name},
        )
    ]


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
    print("evals.vertex.rubric_adaptive — offline import OK")
    print("Predefined rubric metrics:", RUBRIC_METRIC_IDS)
    try:
        for r in run_rubric_eval():
            print(r.to_dict())
    except RuntimeError as exc:
        print("Guard raised (expected offline):", exc)
