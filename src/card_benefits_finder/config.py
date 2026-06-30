"""Central configuration + backend selection for the agent and eval layers.

The whole repo is *local-first*: with ``EVAL_BACKEND=stub`` (the default) nothing
here imports ``google-adk`` or ``google-cloud-aiplatform`` and no credentials are
needed. Setting ``EVAL_BACKEND=gemini`` or ``vertex`` switches the agent's model
and the LLM-judges to a live model.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = PACKAGE_DIR / "knowledge"
CARDS_JSON = KNOWLEDGE_DIR / "cards.json"
BENEFITS_KB = KNOWLEDGE_DIR / "benefits_kb.md"

VALID_BACKENDS = {"stub", "gemini", "vertex"}


@dataclass(frozen=True)
class Settings:
    backend: str
    agent_model: str
    judge_model: str
    project: str | None
    location: str
    staging_bucket: str | None
    experiment: str
    gate_mode: str

    @property
    def is_offline(self) -> bool:
        return self.backend == "stub"

    @property
    def use_vertexai(self) -> bool:
        return self.backend == "vertex"


def load_settings() -> Settings:
    backend = os.environ.get("EVAL_BACKEND", "stub").strip().lower()
    if backend not in VALID_BACKENDS:
        raise ValueError(
            f"EVAL_BACKEND={backend!r} invalid; expected one of {sorted(VALID_BACKENDS)}"
        )
    return Settings(
        backend=backend,
        agent_model=os.environ.get("AGENT_MODEL", "gemini-2.5-flash"),
        judge_model=os.environ.get("JUDGE_MODEL", "gemini-2.5-flash"),
        project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        staging_bucket=os.environ.get("STAGING_BUCKET"),
        experiment=os.environ.get("VERTEX_EXPERIMENT", "card-benefits-eval"),
        gate_mode=os.environ.get("GATE_MODE", "full").strip().lower(),
    )


SETTINGS = load_settings()
