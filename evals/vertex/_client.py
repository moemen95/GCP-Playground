"""Lazy client construction + offline guard for the Vertex Gen AI Evaluation layer.

This module is the single place every other ``evals/vertex/`` module goes to obtain
a live SDK handle. It is deliberately importable **offline**: none of the
``vertexai`` / ``google.genai`` packages are imported at module load time. They are
imported lazily inside the functions below, so ``import evals.vertex._client`` works
with no SDK installed and no GCP credentials. Calling a function that actually needs
the SDK raises a clear :class:`RuntimeError`.

Two SDK surfaces
================
There are TWO ways to drive the Vertex AI Gen AI Evaluation Service. This repo
treats the new GenAI client as PRIMARY and the legacy module as DEPRECATED.

1. **New GenAI client (PRIMARY)** — ``vertexai.Client`` (a thin wrapper over the
   ``google-genai`` SDK). Entry points live under ``client.evals``:

   * ``client.evals.run_inference(model=..., src=...)`` — batch-generate candidate
     responses for a dataset (so you can evaluate a live model end-to-end).
   * ``client.evals.evaluate(dataset=..., metrics=[...])`` — score a dataset against
     a list of metrics. Backs the ``evaluateDataset`` REST method.
   * ``client.evals.generate_rubrics(src=..., rubric_group_name=..., metric=...)`` —
     adaptively generate per-example rubrics. Backs ``generateInstanceRubrics``.

   Metric types come from ``vertexai.types``:

   * ``vertexai.types.Metric(name="rouge_1")`` — computation / tool-call metrics.
   * ``vertexai.types.PrebuiltMetric.GROUNDEDNESS`` — model-based pointwise metrics.
   * ``vertexai.types.LLMMetric`` — custom autorater metric (criteria + rubric).
   * ``vertexai.types.RubricMetric.GENERAL_QUALITY`` — adaptive / managed rubrics.

2. **Legacy ``vertexai.evaluation`` (DEPRECATED, removal ~2026-06-24)** —
   ``EvalTask``, ``PointwiseMetric``, ``PairwiseMetric``,
   ``MetricPromptTemplateExamples``. Kept for reference in ``legacy_evaltask.py``.
   Prefer surface (1) for all new work.

REST surface (for reference / non-Python callers)
==================================================
The service exposes these methods under the ``aiplatform`` API:

* ``evaluateInstances``     — score a single instance (one prompt/response).
* ``evaluateDataset``       — score a whole dataset (what ``client.evals.evaluate`` calls).
* ``generateInstanceRubrics`` — adaptive rubric generation (``generate_rubrics``).
* ``generateLossClusters``  — cluster low-scoring examples for error analysis.
* ``recommendSpec``         — recommend an eval spec / metric set for a dataset.

Product naming note
===================
"Generative AI on Vertex AI" has been rebranded toward the **Gemini Enterprise
Agent Platform**; the eval service and SDK names referenced here are unchanged.
"""
from __future__ import annotations

import os

# The single, shared error message the whole layer raises when the backend is not
# wired for live Vertex calls. Keep the exact string stable — tests assert on it.
_REQUIRES_MSG = "requires EVAL_BACKEND=vertex and google-cloud-aiplatform"


def require_vertex() -> None:
    """Guard: raise unless the process is configured for live Vertex eval calls.

    Checks (a) ``EVAL_BACKEND == "vertex"`` and (b) a project id is present in
    ``GOOGLE_CLOUD_PROJECT``. Does NOT import any SDK — call this first in any
    function that will then construct a client, so the failure mode is the same
    clear message whether the SDK is missing or merely unconfigured.
    """
    backend = os.environ.get("EVAL_BACKEND", "stub").strip().lower()
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if backend != "vertex" or not project:
        raise RuntimeError(_REQUIRES_MSG)


def get_genai_client():
    """Lazily construct the new GenAI client (``vertexai.Client``).

    Equivalent to::

        from vertexai import Client
        client = Client(
            project=os.environ["GOOGLE_CLOUD_PROJECT"],
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )

    Imports ``vertexai`` lazily so this module stays importable offline. Raises the
    shared :class:`RuntimeError` if the backend is not configured or the SDK is not
    installed.
    """
    require_vertex()
    try:
        from vertexai import Client  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without the SDK
        raise RuntimeError(_REQUIRES_MSG) from exc
    return Client(
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
    )


def get_vertex_types():
    """Lazily return the ``vertexai.types`` module (Metric / PrebuiltMetric / ...)."""
    require_vertex()
    try:
        from vertexai import types  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(_REQUIRES_MSG) from exc
    return types


if __name__ == "__main__":  # pragma: no cover - manual demo
    print("evals.vertex._client — offline import OK")
    print("Backend:", os.environ.get("EVAL_BACKEND", "stub"))
    try:
        get_genai_client()
        print("Live Vertex client constructed.")
    except RuntimeError as exc:
        print("Guard raised (expected offline):", exc)
