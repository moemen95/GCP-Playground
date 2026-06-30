"""Offline tests for the Layer-4 gate."""
from __future__ import annotations

from evals.common import MetricResult, SAFETY, RAG, POINTWISE, TRAJECTORY, load_jsonl
from evals.gating import aggregate
from evals.gating.deterministic_metrics import deterministic_metrics
from evals.gating.gate import evaluate_gate


def _cfg():
    return {
        "mode": "full",
        "hard_gates": {"pii_no_leak": 1.0, "faithfulness": 0.85},
        "soft_metrics": {
            "composite_threshold": 0.7,
            "min_floor": 0.45,
            "weights": {"g_eval_correctness": 0.5, "response_match": 0.5},
        },
        "baseline": {"enabled": False},
    }


def _metrics(faith=0.9, pii=1.0, corr=0.8, resp=0.7, judge=True):
    return [
        MetricResult("faithfulness", "judges", RAG, faith, uses_llm_judge=False),
        MetricResult("pii_no_leak", "judges", SAFETY, pii, uses_llm_judge=False),
        MetricResult("g_eval_correctness", "judges", POINTWISE, corr, uses_llm_judge=judge),
        MetricResult("response_match", "deterministic", "computation", resp, uses_llm_judge=False),
    ]


def test_deterministic_metrics_shapes():
    ms = deterministic_metrics(load_jsonl("rag_qa.jsonl"))
    names = {m.name for m in ms}
    assert {"trajectory_in_order_match", "tool_name_match", "response_match"} <= names
    assert all(0.0 <= m.score <= 1.0 for m in ms)


def test_gate_passes_when_all_good():
    out = evaluate_gate(_metrics(), _cfg())
    assert out.passed and not out.failures


def test_hard_gate_failure_blocks():
    out = evaluate_gate(_metrics(faith=0.5), _cfg())
    assert not out.passed
    assert any("faithfulness" in f for f in out.failures)


def test_missing_hard_gate_is_fail_closed():
    ms = [m for m in _metrics() if m.name != "pii_no_leak"]
    out = evaluate_gate(ms, _cfg())
    assert not out.passed
    assert any("pii_no_leak" in f and "missing" in f for f in out.failures)


def test_soft_floor_failure_blocks():
    out = evaluate_gate(_metrics(corr=0.2), _cfg())  # below min_floor 0.45
    assert not out.passed
    assert any("floor:g_eval_correctness" in f for f in out.failures)


def test_deterministic_mode_drops_llm_metrics():
    cfg = _cfg()
    cfg["mode"] = "deterministic"
    # g_eval_correctness is an LLM judge -> dropped -> composite from response_match only
    out = evaluate_gate(_metrics(corr=0.99, judge=True), cfg)
    assert "g_eval_correctness" not in out.soft["per_metric"]


def test_weighted_composite_renormalizes_on_missing():
    by_name = {"a": MetricResult("a", "x", "y", 1.0), "b": MetricResult("b", "x", "y", 0.0)}
    comp = aggregate.weighted_composite(by_name, {"a": 0.5, "b": 0.5, "c": 0.5})
    assert comp["composite"] == 0.5 and comp["missing"] == ["c"]


def test_regression_check():
    assert aggregate.regression_check(0.8, {"composite": 0.79}, 0.02)["passed"]
    assert not aggregate.regression_check(0.7, {"composite": 0.8}, 0.02)["passed"]
    assert aggregate.regression_check(0.8, None, 0.02) == {"checked": False}
