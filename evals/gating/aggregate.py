"""Score aggregation + baseline no-regression check for the release gate."""
from __future__ import annotations

import json
from pathlib import Path

from evals.common import MetricResult


def weighted_composite(metrics_by_name: dict[str, MetricResult], weights: dict[str, float]) -> dict:
    """Weighted soft composite over the metrics that are present.

    Weights are renormalized across present metrics so a missing optional metric
    doesn't silently deflate the score. Returns composite + per-metric detail.
    """
    present = {n: w for n, w in weights.items() if n in metrics_by_name}
    total_w = sum(present.values())
    per_metric = {}
    composite = 0.0
    for name, w in present.items():
        score = metrics_by_name[name].score
        norm_w = w / total_w if total_w else 0.0
        composite += score * norm_w
        per_metric[name] = {"score": round(score, 4), "weight": round(norm_w, 4)}
    missing = [n for n in weights if n not in metrics_by_name]
    return {
        "composite": round(composite, 4),
        "per_metric": per_metric,
        "missing": missing,
    }


def floor_failures(metrics_by_name: dict[str, MetricResult], weights: dict[str, float],
                   min_floor: float) -> list[str]:
    return [n for n in weights
            if n in metrics_by_name and metrics_by_name[n].score < min_floor]


def load_baseline(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def save_baseline(path: str | Path, composite: float, hard: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"composite": composite, "hard": hard}, indent=2), encoding="utf-8")


def regression_check(current_composite: float, baseline: dict | None,
                     allowed_regression: float) -> dict:
    if not baseline:
        return {"checked": False}
    prev = baseline.get("composite", 0.0)
    delta = round(current_composite - prev, 4)
    return {
        "checked": True,
        "baseline_composite": prev,
        "delta": delta,
        "passed": delta >= -allowed_regression,
        "allowed_regression": allowed_regression,
    }
