"""Pairwise comparison: in-SDK ``PairwiseMetric`` + the AutoSxS pipeline.

Two ways to ask "is candidate A better than baseline B?" on Vertex:

(a) In-SDK pairwise metric — synchronous, good for small dev datasets.
(b) AutoSxS — a managed Vertex Pipeline for large-scale, human-aligned SxS.

(a) In-SDK pairwise metric
=========================
A ``PairwiseMetric`` built from a ``PairwiseMetricPromptTemplate``:

* ``criteria``       — dict of named judging dimensions (what "better" means).
* ``rating_rubric``  — dict keyed ``"A"`` / ``"B"`` / ``"tie"`` describing each verdict.
* baseline responses — either a ``baseline_model`` (the autorater generates B on the
  fly) or a precomputed ``baseline_model_response`` column in the dataset.

The result reports a win rate for the candidate as
``<metric_name>/candidate_model_win_rate`` (fraction of examples where the autorater
preferred the candidate over the baseline).

(b) AutoSxS pipeline (Vertex Pipelines)
======================================
Template: ``autosxs-template/<version>`` (a Google-published pipeline template).
Supported ``task`` values: ``"summarization"`` and ``"question_answering"``.

Key pipeline parameters:

* ``evaluation_dataset``           — JSONL/BigQuery of prompts + both responses.
* ``id_columns``                   — columns that uniquely identify each example.
* ``task``                         — ``summarization`` | ``question_answering``.
* ``autorater_prompt_parameters``  — maps your columns to the autorater inputs
                                     (e.g. ``inference_context``, ``inference_instruction``).
* ``response_column_a`` / ``response_column_b`` — the two candidate response columns.
* ``human_preference_column``      — optional gold human label, enabling alignment metrics.

Outputs:

* a **judgments table** — per-example winner + autorater **confidence** + **explanation**.
* **win-rate metrics** — aggregate share of A-wins vs B-wins.
* **human-alignment metrics** — when ``human_preference_column`` is set, agreement
  with humans including **Cohen's Kappa**.
"""
from __future__ import annotations

from evals.common import MetricResult, PAIRWISE

from ._client import get_genai_client, get_vertex_types, require_vertex

# AutoSxS pipeline template + the only two supported tasks.
AUTOSXS_TEMPLATE = "autosxs-template/2.18.0"  # autosxs-template/<version>
AUTOSXS_TASKS = ("summarization", "question_answering")

# Card-domain pairwise example: two candidate answers to the same benefits question.
PAIRWISE_CRITERIA = {
    "benefit_accuracy": "States the correct limit/eligibility for the named card benefit.",
    "groundedness": "Claims are supported by the card's published terms.",
    "helpfulness": "Directly and concisely answers the user's question.",
}
PAIRWISE_RATING_RUBRIC = {
    "A": "Response A is more accurate, grounded, and helpful than B.",
    "B": "Response B is more accurate, grounded, and helpful than A.",
    "tie": "Both responses are comparable in accuracy, grounding, and helpfulness.",
}


def build_pairwise_dataset() -> list[dict]:
    """Rows with a candidate ``response`` and a ``baseline_model_response`` to beat."""
    return [
        {
            "prompt": "How long does Tangerine World Mastercard rental car coverage last?",
            # Candidate (correct): 48 days.
            "response": "Rental car CDW on the Tangerine World Mastercard covers up to 48 consecutive days.",
            # Baseline (vaguer / partly wrong): generic, no day limit.
            "baseline_model_response": "The card gives you some rental car insurance when you book a car.",
        },
        {
            "prompt": "Does the Money-Back Mastercard have an annual fee?",
            "response": "No — the Tangerine Money-Back Mastercard has a $0 annual fee.",
            "baseline_model_response": "There may be a small annual fee depending on your plan.",
        },
    ]


def run_pairwise_metric() -> list[MetricResult]:
    """(a) In-SDK pairwise comparison via the new GenAI client.

    Constructs a ``PairwiseMetric`` from a ``PairwiseMetricPromptTemplate``
    (``criteria`` + ``rating_rubric`` keyed A/B/tie) and evaluates the candidate
    ``response`` against the ``baseline_model_response`` column::

        from vertexai import types
        metric = types.PairwiseMetric(
            metric="benefit_pairwise",
            metric_prompt_template=types.PairwiseMetricPromptTemplate(
                criteria=PAIRWISE_CRITERIA,
                rating_rubric=PAIRWISE_RATING_RUBRIC,
            ),
            baseline_response_column_name="baseline_model_response",
        )
        result = client.evals.evaluate(dataset=df, metrics=[metric])
        win_rate = result.summary_metrics["benefit_pairwise/candidate_model_win_rate"]

    Returns a single :class:`MetricResult` carrying the candidate win rate. Raises
    the shared :class:`RuntimeError` offline.
    """
    require_vertex()
    client = get_genai_client()
    types = get_vertex_types()

    metric = types.PairwiseMetric(
        metric="benefit_pairwise",
        metric_prompt_template=types.PairwiseMetricPromptTemplate(
            criteria=PAIRWISE_CRITERIA,
            rating_rubric=PAIRWISE_RATING_RUBRIC,
        ),
        baseline_response_column_name="baseline_model_response",
    )

    df = build_pairwise_dataset()
    eval_result = client.evals.evaluate(dataset=df, metrics=[metric])
    summary = getattr(eval_result, "summary_metrics", {}) or {}
    win_rate = _win_rate(summary, "benefit_pairwise")

    return [
        MetricResult(
            name="benefit_pairwise/candidate_model_win_rate",
            layer="vertex",
            family=PAIRWISE,
            score=float(win_rate),
            uses_llm_judge=True,
            n=len(df),
            details={"criteria": list(PAIRWISE_CRITERIA), "baseline": "baseline_model_response"},
        )
    ]


def autosxs_pipeline_spec(task: str = "question_answering") -> dict:
    """(b) Build the AutoSxS Vertex-Pipeline parameter spec (no execution).

    Returns the ``parameter_values`` dict you would pass to
    ``aiplatform.PipelineJob(template_path=<AUTOSXS_TEMPLATE>, ...)`` to launch the
    managed AutoSxS pipeline. Pure data assembly — safe to call offline (it still
    guards so the demo path is consistent across modules).
    """
    require_vertex()
    if task not in AUTOSXS_TASKS:
        raise ValueError(f"task must be one of {AUTOSXS_TASKS}, got {task!r}")
    return {
        "evaluation_dataset": "gs://<bucket>/card_benefits_sxs.jsonl",
        "id_columns": ["prompt"],
        "task": task,
        "autorater_prompt_parameters": {
            "inference_instruction": {"column": "prompt"},
            "inference_context": {"column": "context"},
        },
        "response_column_a": "response",
        "response_column_b": "baseline_model_response",
        # Optional gold labels enable human-alignment + Cohen's Kappa in the output.
        "human_preference_column": "human_preference",
        "_template_path": AUTOSXS_TEMPLATE,
    }


def _win_rate(summary, metric_name: str) -> float:
    key = f"{metric_name}/candidate_model_win_rate"
    if isinstance(summary, dict):
        return float(summary.get(key, 0.0) or 0.0)
    for rec in summary:
        name = getattr(rec, "metric_name", None) or (rec.get("metric_name") if isinstance(rec, dict) else None)
        if name == key:
            score = getattr(rec, "mean_score", None)
            if score is None and isinstance(rec, dict):
                score = rec.get("mean_score")
            return float(score or 0.0)
    return 0.0


if __name__ == "__main__":  # pragma: no cover - manual demo
    print("evals.vertex.pairwise_autosxs — offline import OK")
    print("AutoSxS template:", AUTOSXS_TEMPLATE, "tasks:", AUTOSXS_TASKS)
    try:
        print("AutoSxS spec:", autosxs_pipeline_spec("question_answering"))
        for r in run_pairwise_metric():
            print(r.to_dict())
    except RuntimeError as exc:
        print("Guard raised (expected offline):", exc)
