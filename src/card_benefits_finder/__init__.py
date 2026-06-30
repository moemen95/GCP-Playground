"""Tangerine Card Benefits Finder — a Google ADK agent + its offline twin.

ADK tooling imports ``agent`` lazily (it needs ``google-adk``); the offline eval
path imports ``local_agent`` instead, which has no cloud dependencies.
"""

# Re-export the ADK agent so `adk web` / `adk eval` discover `root_agent`, but stay
# importable offline (submodules like `.tools` work even without google-adk).
try:  # pragma: no cover - exercised only when google-adk is installed
    from . import agent  # noqa: F401
except ImportError:
    pass

