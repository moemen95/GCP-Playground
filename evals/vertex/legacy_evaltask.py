"""DEPRECATED reference: the legacy ``vertexai.evaluation`` (``EvalTask``) path.

    !!  DEPRECATED — scheduled for removal ~2026-06-24.  !!
    !!  Use the new GenAI client (``vertexai.Client`` / ``client.evals.*``) instead. !!
    !!  See ``_client.py`` and the other modules in this package for the PRIMARY path. !!

This module is kept ONLY so the older API shape is documented in one place. Prefer:

    legacy ``EvalTask(...).evaluate(...)``   ->   new ``client.evals.evaluate(...)``
    legacy ``PointwiseMetric`` / ``PairwiseMetric`` from ``vertexai.evaluation``
                                              ->   ``vertexai.types.*`` metrics
    legacy ``MetricPromptTemplateExamples``  ->   ``vertexai.types.PrebuiltMetric.*``

Legacy API shape
===============
    from vertexai.evaluation import (
        EvalTask, PointwiseMetric, PairwiseMetric, MetricPromptTemplateExamples,
    )

    eval_task = EvalTask(
        dataset=df,                       # pandas DataFrame: prompt/response/reference/context
        metrics=[
            MetricPromptTemplateExamples.Pointwise.GROUNDEDNESS,
            MetricPromptTemplateExamples.Pointwise.QUESTION_ANSWERING_QUALITY,
        ],
        experiment="card-benefits-eval",
    )
    eval_result = eval_task.evaluate(
        model=candidate_model,            # optional: generate responses inline
        prompt_template="{prompt}",
    )
    eval_result.summary_metrics          # dict of aggregate scores
    eval_result.metrics_table            # per-row pandas DataFrame
"""
from __future__ import annotations

import warnings

from evals.common import MetricResult, POINTWISE

from ._client import require_vertex

_DEPRECATION = (
    "vertexai.evaluation.EvalTask is DEPRECATED (removal ~2026-06-24); "
    "use the new GenAI client (vertexai.Client / client.evals.evaluate)."
)


def build_dataset() -> list[dict]:
    return [
        {
            "prompt": "Does the World Mastercard cover rental cars?",
            "response": "Yes, for up to 48 consecutive days when paid in full with the card.",
            "context": "Tangerine World Mastercard — Auto Rental CDW: up to 48 consecutive days.",
            "reference": "Yes, up to 48 consecutive days.",
        }
    ]


def run_legacy_evaltask() -> list[MetricResult]:
    """Run the DEPRECATED ``EvalTask`` path (groundedness + QA quality).

    Emits a ``DeprecationWarning``, then lazily imports ``vertexai.evaluation`` and
    runs ``EvalTask(...).evaluate(...)``. Provided only for reference; new code should
    use the modules built on the new GenAI client. Raises the shared
    :class:`RuntimeError` offline.
    """
    warnings.warn(_DEPRECATION, DeprecationWarning, stacklevel=2)
    require_vertex()
    try:
        import vertexai  # type: ignore
        from vertexai.evaluation import (  # type: ignore
            EvalTask,
            MetricPromptTemplateExamples,
        )
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "requires EVAL_BACKEND=vertex and google-cloud-aiplatform"
        ) from exc

    import os

    vertexai.init(
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
    )

    eval_task = EvalTask(
        dataset=build_dataset(),
        metrics=[
            MetricPromptTemplateExamples.Pointwise.GROUNDEDNESS,
            MetricPromptTemplateExamples.Pointwise.QUESTION_ANSWERING_QUALITY,
        ],
        experiment="card-benefits-eval-legacy",
    )
    eval_result = eval_task.evaluate(prompt_template="{prompt}")
    summary = getattr(eval_result, "summary_metrics", {}) or {}

    results: list[MetricResult] = []
    for mid in ("groundedness", "question_answering_quality"):
        results.append(
            MetricResult(
                name=mid,
                layer="vertex",
                family=POINTWISE,
                score=float(summary.get(f"{mid}/mean", summary.get(mid, 0.0)) or 0.0),
                uses_llm_judge=True,
                n=len(build_dataset()),
                details={"deprecated": True, "api": "vertexai.evaluation.EvalTask"},
            )
        )
    return results


if __name__ == "__main__":  # pragma: no cover - manual demo
    print("evals.vertex.legacy_evaltask — offline import OK (DEPRECATED path)")
    print(_DEPRECATION)
    try:
        for r in run_legacy_evaltask():
            print(r.to_dict())
    except RuntimeError as exc:
        print("Guard raised (expected offline):", exc)
