"""The shared metric-result contract that every eval layer produces and the
Layer-4 gate consumes. Keeping this in ``common`` lets Layers 1-3 stay decoupled
from the gate while still feeding it a uniform shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

# Metric families (used by the gate to apply hard/soft policy by class).
COMPUTATION = "computation"
TOOL_USE = "tool_use"
TRAJECTORY = "trajectory"
POINTWISE = "pointwise"
PAIRWISE = "pairwise"
RUBRIC = "rubric"
RAG = "rag"
SAFETY = "safety"


@dataclass
class MetricResult:
    name: str                       # e.g. "groundedness", "trajectory_in_order_match"
    layer: str                      # "adk" | "vertex" | "judges"
    family: str                     # one of the constants above
    score: float                    # normalized 0..1, higher = better
    passed: bool | None = None      # vs its own threshold, if evaluated
    threshold: float | None = None
    uses_llm_judge: bool = False    # gate can drop these in deterministic mode
    n: int = 1                      # number of examples aggregated
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
