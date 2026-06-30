"""The Google ADK agent definition (the System-Under-Test).

Importing this module requires ``google-adk`` (``pip install -e ".[adk]"``). The
offline eval path does NOT import this module — it uses ``local_agent.py`` — so
the repo's default test suite runs without ADK installed.

ADK discovers ``root_agent`` from this package via ``__init__.py``.
"""
from __future__ import annotations

from google.adk.agents import Agent  # type: ignore

from .config import SETTINGS
from .prompts import SYSTEM_INSTRUCTION
from .tools import ALL_TOOLS

root_agent = Agent(
    name="card_benefits_finder",
    model=SETTINGS.agent_model,
    description=(
        "Answers questions about Tangerine credit card benefits, rewards, fees, "
        "and eligibility, grounded in a benefits knowledge base."
    ),
    instruction=SYSTEM_INSTRUCTION,
    tools=ALL_TOOLS,
)
