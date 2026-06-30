"""GCP pipeline orchestration #3 — Agent Engine ONLINE evaluation on live traffic.

After the agent is deployed to Agent Engine (see ``deploy/``), the Agent Platform
can score live production traces with reference-free metrics (groundedness,
safety, instruction-following) and surface drift on the agent dashboard / Unified
Trace Viewer. This module shows wiring an online evaluation against the deployed
resource and sampling drifted traces back into the offline golden set.

Importable offline; live calls need ``EVAL_BACKEND=vertex`` + a deployed agent.
"""
from __future__ import annotations

import os


def require_vertex():
    if os.environ.get("EVAL_BACKEND") != "vertex":
        raise RuntimeError("set EVAL_BACKEND=vertex")
    try:
        import vertexai  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('requires `pip install -e ".[vertex,agent_engines]"`') from exc


def configure_online_eval(agent_engine_resource: str, *, project: str | None = None,
                          location: str = "us-central1",
                          metrics: list[str] | None = None,
                          sampling_rate: float = 0.1):
    """Register an online (reference-free) evaluation on a deployed Agent Engine.

    ``agent_engine_resource``: projects/.../reasoningEngines/<id>.
    ``metrics``: reference-free ids suitable for production (no ground truth):
        groundedness, safety, instruction_following, coherence.
    ``sampling_rate``: fraction of live traffic to score.
    """
    require_vertex()
    from vertexai import Client

    project = project or os.environ["GOOGLE_CLOUD_PROJECT"]
    metrics = metrics or ["groundedness", "safety", "instruction_following"]
    client = Client(project=project, location=location)
    # The Agent Platform exposes online evaluation config on the deployed agent;
    # this mirrors the documented online-eval + dashboard/Unified-Trace-Viewer flow.
    return {
        "agent_engine": agent_engine_resource,
        "online_metrics": metrics,
        "sampling_rate": sampling_rate,
        "client": client.__class__.__name__,
        "note": "wire to client.evals online evaluation / Agent Engine monitoring config",
    }


def harvest_drifted_traces(agent_engine_resource: str, *, since: str,
                           min_severity: str = "high") -> list[dict]:
    """Pull flagged/low-scoring or drifted traces to fold into the golden set.

    Closes the online->offline loop: drift alert -> sample traces -> human review
    (annotation queue) -> add to evals/datasets -> recalibrate the gate.
    """
    require_vertex()
    # Placeholder for the trace-export query; returns rows shaped like the golden set.
    return []


if __name__ == "__main__":  # pragma: no cover
    try:
        print(configure_online_eval("projects/P/locations/us-central1/reasoningEngines/123"))
    except (RuntimeError, KeyError) as exc:
        print(f"[online_eval_agent_engine] requires vertex backend + deployed agent: {exc}")
