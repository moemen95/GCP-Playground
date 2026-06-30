"""Layer 2 — Vertex AI Gen AI Evaluation Service.

Every module here is importable **offline**: ``vertexai`` / ``google.genai`` are
imported lazily inside functions, never at module top level. Functions that need live
Vertex calls raise ``RuntimeError("requires EVAL_BACKEND=vertex and
google-cloud-aiplatform")`` when the backend is not configured.

Primary SDK surface: the new GenAI client (``vertexai.Client`` -> ``client.evals.*``)
with metric types from ``vertexai.types``. The legacy ``vertexai.evaluation``
(``EvalTask``) path is DEPRECATED and kept only in ``legacy_evaltask.py``.

See ``README.md`` for the full metric-family -> identifier mapping.
"""
from __future__ import annotations

from ._client import get_genai_client, get_vertex_types, require_vertex
from .computation_metrics import (
    TEXT_METRIC_IDS,
    TOOL_METRIC_IDS,
    run_computation_metrics,
)
from .pointwise_metrics import POINTWISE_METRIC_IDS, run_pointwise_metrics
from .pairwise_autosxs import (
    AUTOSXS_TASKS,
    AUTOSXS_TEMPLATE,
    autosxs_pipeline_spec,
    run_pairwise_metric,
)
from .trajectory_eval import (
    TRAJECTORY_METRIC_IDS,
    build_runnable,
    run_trajectory_eval,
)
from .rubric_adaptive import (
    RUBRIC_METRIC_IDS,
    generate_rubric_group,
    run_rubric_eval,
)
from .custom_autorater import (
    build_autorater_config,
    build_benefit_accuracy_metric,
    run_custom_autorater,
)
from .legacy_evaltask import run_legacy_evaltask

__all__ = [
    # client / guard
    "get_genai_client",
    "get_vertex_types",
    "require_vertex",
    # computation + tool-call
    "TEXT_METRIC_IDS",
    "TOOL_METRIC_IDS",
    "run_computation_metrics",
    # pointwise
    "POINTWISE_METRIC_IDS",
    "run_pointwise_metrics",
    # pairwise / autosxs
    "AUTOSXS_TASKS",
    "AUTOSXS_TEMPLATE",
    "autosxs_pipeline_spec",
    "run_pairwise_metric",
    # trajectory
    "TRAJECTORY_METRIC_IDS",
    "build_runnable",
    "run_trajectory_eval",
    # rubric
    "RUBRIC_METRIC_IDS",
    "generate_rubric_group",
    "run_rubric_eval",
    # custom autorater
    "build_autorater_config",
    "build_benefit_accuracy_metric",
    "run_custom_autorater",
    # legacy (deprecated)
    "run_legacy_evaltask",
]
