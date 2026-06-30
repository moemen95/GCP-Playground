"""The release gate: tiered conjunctive hard-gates + weighted soft composite.

Policy:
* **Hard gates** (safety, grounding, trajectory floors) are conjunctive — ANY
  failure blocks the release, regardless of the soft score. Fail-closed if a
  hard-gate metric is missing.
* **Soft metrics** form a weighted composite; the composite must clear its
  threshold AND every present soft metric must clear ``min_floor``.
* **Deterministic mode** drops LLM-judge metrics (``uses_llm_judge=True``) before
  gating, for strict data-governance.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from evals.common import MetricResult
from .aggregate import (
    floor_failures, load_baseline, regression_check, weighted_composite,
)

THRESHOLDS_PATH = Path(__file__).resolve().parent / "thresholds.yaml"


def load_config(path: str | Path = THRESHOLDS_PATH) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    # GATE_MODE env overrides the file.
    cfg["mode"] = os.environ.get("GATE_MODE", cfg.get("mode", "full")).strip().lower()
    return cfg


@dataclass
class GateOutcome:
    passed: bool
    mode: str
    hard: list[dict] = field(default_factory=list)
    soft: dict = field(default_factory=dict)
    baseline: dict = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed, "mode": self.mode, "hard": self.hard,
            "soft": self.soft, "baseline": self.baseline, "failures": self.failures,
        }


def _index(metrics: list[MetricResult]) -> dict[str, MetricResult]:
    by_name: dict[str, MetricResult] = {}
    for m in metrics:
        # If two layers report the same metric name, keep the larger-sample one.
        if m.name not in by_name or m.n > by_name[m.name].n:
            by_name[m.name] = m
    return by_name


def evaluate_gate(metrics: list[MetricResult], cfg: dict | None = None,
                  *, baseline_path: str | Path = "reports/baseline.json") -> GateOutcome:
    cfg = cfg or load_config()
    mode = cfg.get("mode", "full")

    # Deterministic mode: drop LLM-judge metrics entirely.
    if mode == "deterministic":
        metrics = [m for m in metrics if not m.uses_llm_judge]

    by_name = _index(metrics)
    failures: list[str] = []

    # ---- hard gates (conjunctive, fail-closed) ----
    hard_results = []
    for name, threshold in (cfg.get("hard_gates") or {}).items():
        m = by_name.get(name)
        if m is None:
            hard_results.append({"name": name, "present": False, "passed": False,
                                 "threshold": threshold})
            failures.append(f"hard:{name} (missing)")
            continue
        ok = m.score >= threshold
        hard_results.append({"name": name, "present": True, "score": m.score,
                             "threshold": threshold, "passed": ok,
                             "uses_llm_judge": m.uses_llm_judge})
        if not ok:
            failures.append(f"hard:{name} ({m.score:.3f}<{threshold})")

    # ---- soft composite ----
    soft_cfg = cfg.get("soft_metrics") or {}
    weights = soft_cfg.get("weights", {})
    comp = weighted_composite(by_name, weights)
    floors = floor_failures(by_name, weights, soft_cfg.get("min_floor", 0.0))
    composite_threshold = soft_cfg.get("composite_threshold", 0.0)
    composite_ok = comp["composite"] >= composite_threshold
    if not composite_ok:
        failures.append(f"soft:composite ({comp['composite']:.3f}<{composite_threshold})")
    for n in floors:
        failures.append(f"soft:floor:{n} ({by_name[n].score:.3f}<{soft_cfg.get('min_floor')})")

    soft = {
        "composite": comp["composite"],
        "threshold": composite_threshold,
        "passed": composite_ok and not floors,
        "per_metric": comp["per_metric"],
        "missing_weighted": comp["missing"],
        "floor_failures": floors,
        "min_floor": soft_cfg.get("min_floor"),
    }

    # ---- baseline no-regression ----
    base_cfg = cfg.get("baseline") or {}
    baseline = {"checked": False}
    if base_cfg.get("enabled"):
        baseline = regression_check(comp["composite"], load_baseline(baseline_path),
                                    base_cfg.get("allowed_regression", 0.0))
        if baseline.get("checked") and not baseline.get("passed"):
            failures.append(f"baseline:regression ({baseline['delta']})")

    passed = not failures
    return GateOutcome(passed=passed, mode=mode, hard=hard_results, soft=soft,
                       baseline=baseline, failures=failures)
