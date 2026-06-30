"""Offline, no-LLM scorers over agent transcripts.

These are the deterministic tier of the gate — trajectory match, tool-call
accuracy, and response/ROUGE match — computed locally from
``{expected_trajectory, predicted_trajectory, reference, response}`` rows. They
mirror the identifiers GCP computes live (ADK ``tool_trajectory_avg_score`` /
Vertex ``trajectory_*`` + ``tool_*``) so the same gate runs offline and on GCP.
"""
from __future__ import annotations

from evals.common import (
    MetricResult, COMPUTATION, TOOL_USE, TRAJECTORY, mean, rouge_l_f1,
)


def _names(traj: list[dict]) -> list[str]:
    return [c.get("name") for c in traj]


def _is_subsequence(sub: list, seq: list) -> bool:
    it = iter(seq)
    return all(x in it for x in sub)


def trajectory_exact_match(rows: list[dict]) -> float:
    return mean(1.0 if r.get("predicted_trajectory") == r.get("expected_trajectory") else 0.0
                for r in rows)


def trajectory_in_order_match(rows: list[dict]) -> float:
    vals = []
    for r in rows:
        exp, pred = _names(r.get("expected_trajectory", [])), _names(r.get("predicted_trajectory", []))
        vals.append(1.0 if _is_subsequence(exp, pred) else 0.0)
    return mean(vals)


def tool_name_match(rows: list[dict]) -> float:
    """Over rows that expect a tool: did the agent call the right tool name first?"""
    vals = []
    for r in rows:
        exp, pred = r.get("expected_trajectory", []), r.get("predicted_trajectory", [])
        if not exp:
            continue
        vals.append(1.0 if pred and pred[0].get("name") == exp[0].get("name") else 0.0)
    return mean(vals) if vals else 1.0


def tool_parameter_kv_match(rows: list[dict]) -> float:
    """Over rows that expect a tool: did the first call's args match exactly?"""
    vals = []
    for r in rows:
        exp, pred = r.get("expected_trajectory", []), r.get("predicted_trajectory", [])
        if not exp:
            continue
        ok = bool(pred) and pred[0].get("name") == exp[0].get("name") and \
            pred[0].get("args") == exp[0].get("args")
        vals.append(1.0 if ok else 0.0)
    return mean(vals) if vals else 1.0


def response_match(rows: list[dict]) -> float:
    """ROUGE-L F1 of response vs reference, over non-refusal rows."""
    vals = [rouge_l_f1(r.get("response", ""), r.get("reference", ""))
            for r in rows if not r.get("expected_refusal")]
    return mean(vals)


def deterministic_metrics(rows: list[dict], *, thresholds: dict | None = None) -> list[MetricResult]:
    th = thresholds or {}
    specs = [
        ("trajectory_exact_match", TRAJECTORY, trajectory_exact_match(rows), 0.7),
        ("trajectory_in_order_match", TRAJECTORY, trajectory_in_order_match(rows), 0.8),
        ("tool_name_match", TOOL_USE, tool_name_match(rows), 0.9),
        ("tool_parameter_kv_match", TOOL_USE, tool_parameter_kv_match(rows), 0.8),
        ("response_match", COMPUTATION, response_match(rows), 0.3),
    ]
    out = []
    for name, family, score, default_th in specs:
        thr = th.get(name, default_th)
        out.append(MetricResult(
            name=name, layer="deterministic", family=family, score=round(score, 4),
            passed=score >= thr, threshold=thr, uses_llm_judge=False, n=len(rows),
        ))
    return out


if __name__ == "__main__":  # pragma: no cover
    from evals.common import load_jsonl

    for m in deterministic_metrics(load_jsonl("rag_qa.jsonl")):
        print(f"{m.name:28s} {m.score:.3f} {'pass' if m.passed else 'FAIL'}")
