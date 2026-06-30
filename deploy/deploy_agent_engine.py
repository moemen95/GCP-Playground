"""Deploy the ADK agent to Vertex AI Agent Engine, then (optionally) re-validate
it with the Layer-2 managed evaluation.

    EVAL_BACKEND=vertex GOOGLE_CLOUD_PROJECT=... STAGING_BUCKET=gs://... \
        python deploy/deploy_agent_engine.py

Needs ``pip install -e ".[adk,vertex]"`` and authenticated gcloud.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def deploy():
    try:
        import vertexai
        from vertexai import agent_engines
        from vertexai.preview.reasoning_engines import AdkApp
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('requires `pip install -e ".[adk,vertex]"`') from exc

    from card_benefits_finder.agent import root_agent  # needs google-adk

    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    bucket = os.environ["STAGING_BUCKET"]

    vertexai.init(project=project, location=location, staging_bucket=bucket)
    app = AdkApp(agent=root_agent, enable_tracing=True)
    remote = agent_engines.create(
        agent_engine=app,
        requirements=["google-cloud-aiplatform[adk,agent_engines]"],
        display_name="card-benefits-finder",
    )
    print(f"Deployed: {remote.resource_name}")
    return remote


if __name__ == "__main__":
    try:
        deploy()
    except (RuntimeError, KeyError) as exc:
        print(f"[deploy] requires vertex backend + project/bucket + google-adk: {exc}")
