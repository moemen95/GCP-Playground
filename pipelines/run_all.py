"""Run every (offline-capable) eval layer against the agent and gate the result.

Collects metrics from:
* Layer 4 deterministic scorers (trajectory / tool / response),
* Layer 3 judges (RAG, grounding, G-Eval, safety),
* optionally Layer 2 Vertex (``--layers vertex``, needs ``EVAL_BACKEND=vertex``),

then applies the Layer-4 gate and writes a unified report.

    python pipelines/run_all.py                 # offline, stub backend, full mode
    GATE_MODE=deterministic python pipelines/run_all.py
    EVAL_BACKEND=vertex python pipelines/run_all.py --layers vertex
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from evals.common import MetricResult, load_jsonl  # noqa: E402
from evals.common.model_backend import get_backend  # noqa: E402
from evals.gating.deterministic_metrics import deterministic_metrics  # noqa: E402
from evals.gating.gate import evaluate_gate, load_config  # noqa: E402
from evals.gating.report import write_report  # noqa: E402
from evals.judges import geval, jury, nli_groundedness, rag_metrics, safety_suite  # noqa: E402


def collect_offline_metrics(backend_name: str) -> list[MetricResult]:
    backend = get_backend(backend_name)
    qa = load_jsonl("rag_qa.jsonl")
    safety = load_jsonl("safety_redteam.jsonl")
    answered = [r for r in qa if not r.get("expected_refusal")]

    metrics: list[MetricResult] = []

    # --- Layer 4 deterministic (no LLM) ---
    metrics += deterministic_metrics(qa)

    # --- Layer 3 RAG + grounding ---
    metrics += rag_metrics.rag_triad(answered, backend=backend)
    metrics.append(nli_groundedness.hallucination_metric(answered, backend=backend))

    # --- Layer 3 G-Eval (custom LLM judge) ---
    metrics.append(geval.g_eval_dataset(answered, criterion_name="correctness", backend=backend))
    metrics.append(geval.g_eval_dataset(answered, criterion_name="completeness", backend=backend))
    # Jury is informational here (not a gate metric) but demonstrates PoLL.
    metrics.append(jury.jury_dataset(answered, instruction="Rate correctness and grounding."))

    # --- Layer 3 safety (routed by suite) ---
    injection = [r for r in safety if r.get("suite") == "injection"]
    scope = [r for r in safety if r.get("suite") == "scope"]
    metrics.append(safety_suite.injection_metric(injection))
    metrics += safety_suite.refusal_metrics(scope)
    metrics.append(safety_suite.pii_leak_metric(qa + safety))

    return metrics


def collect_vertex_metrics() -> list[MetricResult]:
    """Layer 2 — only runs with EVAL_BACKEND=vertex + creds + SDK."""
    from evals.vertex import trajectory_eval  # lazy; imports offline but calls need creds
    try:
        return trajectory_eval.run_trajectory_eval(load_jsonl("rag_qa.jsonl"))
    except Exception as exc:  # noqa: BLE001 - surface, don't crash the offline run
        print(f"[vertex] skipped: {exc}")
        return []


def run(backend_name: str | None = None, layers: list[str] | None = None):
    backend_name = backend_name or os.environ.get("EVAL_BACKEND", "stub")
    layers = layers or ["offline"]
    metrics = collect_offline_metrics(backend_name)
    if "vertex" in layers:
        metrics += collect_vertex_metrics()

    cfg = load_config()
    outcome = evaluate_gate(metrics, cfg)
    paths = write_report(outcome, metrics, backend=backend_name)
    return outcome, metrics, paths


def main() -> int:
    ap = argparse.ArgumentParser(description="Run all eval layers and gate.")
    ap.add_argument("--layers", nargs="*", default=["offline"],
                    help="offline (default) and/or vertex")
    ap.add_argument("--backend", default=None, help="override EVAL_BACKEND")
    args = ap.parse_args()

    outcome, metrics, paths = run(args.backend, args.layers)
    print(f"\nGate mode: {outcome.mode}  |  result: "
          f"{'PASS' if outcome.passed else 'FAIL'}")
    print(f"Soft composite: {outcome.soft['composite']} / {outcome.soft['threshold']}")
    if outcome.failures:
        print("Failures:")
        for f in outcome.failures:
            print(f"  - {f}")
    print(f"Report: {paths['markdown']}")
    return 0 if outcome.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
