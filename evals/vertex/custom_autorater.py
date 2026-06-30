"""Custom autorater metrics + autorater configuration.

When the prebuilt metrics don't capture what you care about, define your own
autorater metric: give the judge model your ``criteria`` and ``rating_rubric`` and it
scores each response accordingly. You can also tune *how* the autorater judges via
``AutoraterConfig``.

Custom pointwise metric
======================
``types.PointwiseMetric`` built from ``types.PointwiseMetricPromptTemplate``:

* ``criteria``        — named judging dimensions (dict).
* ``rating_rubric``   — what each score on the scale means (dict, e.g. 1..5).
* ``input_variables`` — which dataset columns the template may reference
                        (e.g. ``["prompt", "response", "context", "reference"]``).

Custom pairwise metric
=====================
``types.PairwiseMetric`` with the same ``PointwiseMetricPromptTemplate`` building
blocks plus a baseline response, for "which of A/B is better" custom judging.

AutoraterConfig (how the judge behaves)
======================================
``types.AutoraterConfig`` controls the judge model itself:

* judge model selection — which model rates (default: **Gemini 2.5 Flash**).
* **response flipping**  — also judge with A/B positions swapped and average, to
                           cancel positional bias of the autorater.
* **multi-sampling**     — sample the judge multiple times and aggregate, improving
                           consistency / reducing variance of the verdict.

Judge-model evaluation
=====================
Before trusting an autorater, align it to human labels: run it over a
human-annotated set and measure agreement (e.g. accuracy / Cohen's Kappa). Pick the
judge model + config (flipping, sampling count) that best matches human judgement.

Card-domain custom metric
========================
``benefit_accuracy`` — scores whether the stated **limits / exclusions** in a
response match the card's published Terms & Conditions (the ``context``).
"""
from __future__ import annotations

from evals.common import MetricResult, POINTWISE

from ._client import get_genai_client, get_vertex_types, require_vertex

# Default judge model for the autorater (overridable via AutoraterConfig).
DEFAULT_JUDGE_MODEL = "gemini-2.5-flash"

# The card-domain custom metric definition.
BENEFIT_ACCURACY_CRITERIA = {
    "limit_accuracy": (
        "Every coverage limit, dollar amount, day count, or cap stated in the response "
        "matches the card's Terms & Conditions in the context."
    ),
    "exclusion_accuracy": (
        "Every exclusion or eligibility condition stated in the response is consistent "
        "with the Terms & Conditions; the response invents no benefits not in context."
    ),
}
BENEFIT_ACCURACY_RUBRIC = {
    "5": "All limits and exclusions exactly match the T&Cs; nothing invented.",
    "4": "Limits/exclusions essentially correct with a minor, non-misleading imprecision.",
    "3": "Mostly correct but one limit or exclusion is vague or slightly off.",
    "2": "A material limit or exclusion is wrong or missing.",
    "1": "Multiple limits/exclusions are wrong or benefits are fabricated.",
}
BENEFIT_ACCURACY_INPUT_VARS = ["prompt", "response", "context", "reference"]


def build_dataset() -> list[dict]:
    return [
        {
            "prompt": "What's the rental car coverage limit on the World Mastercard?",
            "response": "It covers rentals up to 48 consecutive days when paid in full with the card.",
            "context": (
                "Tangerine World Mastercard — Auto Rental CDW: up to 48 consecutive days; "
                "full rental cost must be charged to the card."
            ),
            "reference": "Up to 48 consecutive days, full rental charged to the card.",
        }
    ]


def build_benefit_accuracy_metric():
    """Construct the custom ``benefit_accuracy`` ``PointwiseMetric``.

    Equivalent to::

        from vertexai import types
        metric = types.PointwiseMetric(
            metric="benefit_accuracy",
            metric_prompt_template=types.PointwiseMetricPromptTemplate(
                criteria=BENEFIT_ACCURACY_CRITERIA,
                rating_rubric=BENEFIT_ACCURACY_RUBRIC,
                input_variables=BENEFIT_ACCURACY_INPUT_VARS,
            ),
        )

    Raises the shared :class:`RuntimeError` offline.
    """
    require_vertex()
    types = get_vertex_types()
    return types.PointwiseMetric(
        metric="benefit_accuracy",
        metric_prompt_template=types.PointwiseMetricPromptTemplate(
            criteria=BENEFIT_ACCURACY_CRITERIA,
            rating_rubric=BENEFIT_ACCURACY_RUBRIC,
            input_variables=BENEFIT_ACCURACY_INPUT_VARS,
        ),
    )


def build_autorater_config(
    *, judge_model: str = DEFAULT_JUDGE_MODEL, flip_enabled: bool = True, sampling_count: int = 4
):
    """Construct an ``AutoraterConfig`` enabling response flipping + multi-sampling.

    Equivalent to::

        types.AutoraterConfig(
            autorater_model=judge_model,
            flip_enabled=flip_enabled,       # position-bias mitigation
            sampling_count=sampling_count,   # consistency via repeated sampling
        )

    Raises the shared :class:`RuntimeError` offline.
    """
    require_vertex()
    types = get_vertex_types()
    return types.AutoraterConfig(
        autorater_model=judge_model,
        flip_enabled=flip_enabled,
        sampling_count=sampling_count,
    )


def run_custom_autorater() -> list[MetricResult]:
    """Evaluate the custom ``benefit_accuracy`` metric with a tuned autorater.

    Wires the custom metric + an ``AutoraterConfig`` (flipping + multi-sampling) into
    a single ``client.evals.evaluate`` call::

        client.evals.evaluate(
            dataset=df,
            metrics=[build_benefit_accuracy_metric()],
            autorater_config=build_autorater_config(),
        )

    Returns one :class:`MetricResult` (normalized from the 1..5 scale). Raises the
    shared :class:`RuntimeError` offline.
    """
    require_vertex()
    client = get_genai_client()

    df = build_dataset()
    eval_result = client.evals.evaluate(
        dataset=df,
        metrics=[build_benefit_accuracy_metric()],
        autorater_config=build_autorater_config(),
    )
    summary = getattr(eval_result, "summary_metrics", None) or {}
    raw = _lookup_score(summary, "benefit_accuracy")
    norm = (raw - 1) / 4 if raw else 0.0  # 1..5 -> 0..1

    return [
        MetricResult(
            name="benefit_accuracy",
            layer="vertex",
            family=POINTWISE,
            score=float(max(0.0, min(1.0, norm))),
            uses_llm_judge=True,
            n=len(df),
            details={
                "raw_score": raw,
                "scale": [1, 5],
                "judge_model": DEFAULT_JUDGE_MODEL,
                "flip_enabled": True,
                "sampling_count": 4,
            },
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
    print("evals.vertex.custom_autorater — offline import OK")
    print("Custom metric: benefit_accuracy; judge:", DEFAULT_JUDGE_MODEL)
    try:
        for r in run_custom_autorater():
            print(r.to_dict())
    except RuntimeError as exc:
        print("Guard raised (expected offline):", exc)
