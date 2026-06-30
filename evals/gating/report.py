"""Render a unified gate report as Markdown + JSON."""
from __future__ import annotations

import json
from pathlib import Path

from evals.common import MetricResult
from .gate import GateOutcome

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


def _badge(ok: bool) -> str:
    return "✅ PASS" if ok else "❌ FAIL"


def render_markdown(outcome: GateOutcome, metrics: list[MetricResult], *,
                    backend: str = "stub") -> str:
    lines = [
        "# Pre-Prod Eval Gate Report",
        "",
        f"**Result:** {_badge(outcome.passed)}  |  **Mode:** `{outcome.mode}`  "
        f"|  **Backend:** `{backend}`",
        "",
        "## Hard gates (conjunctive — all must pass)",
        "",
        "| Metric | Score | Threshold | Result |",
        "| --- | --- | --- | --- |",
    ]
    for h in outcome.hard:
        score = f"{h['score']:.3f}" if h.get("present") else "—(missing)"
        lines.append(f"| `{h['name']}` | {score} | {h['threshold']} | {_badge(h['passed'])} |")

    soft = outcome.soft
    lines += [
        "",
        f"## Soft composite — {soft['composite']:.3f} / {soft['threshold']} "
        f"{_badge(soft['passed'])}",
        "",
        "| Metric | Score | Weight |",
        "| --- | --- | --- |",
    ]
    for name, d in soft.get("per_metric", {}).items():
        lines.append(f"| `{name}` | {d['score']:.3f} | {d['weight']:.2f} |")
    if soft.get("floor_failures"):
        lines.append("")
        lines.append(f"> ⚠️ Below `min_floor` ({soft['min_floor']}): "
                     + ", ".join(f"`{n}`" for n in soft["floor_failures"]))

    if outcome.baseline.get("checked"):
        b = outcome.baseline
        lines += ["", "## Baseline no-regression",
                  f"- delta `{b['delta']}` vs baseline `{b['baseline_composite']}` "
                  f"(allowed −{b['allowed_regression']}) {_badge(b['passed'])}"]

    lines += ["", "## All metrics", "", "| Layer | Metric | Family | Judge? | Score | n |",
              "| --- | --- | --- | --- | --- | --- |"]
    for m in sorted(metrics, key=lambda x: (x.layer, x.family, x.name)):
        lines.append(f"| {m.layer} | `{m.name}` | {m.family} | "
                     f"{'yes' if m.uses_llm_judge else 'no'} | {m.score:.3f} | {m.n} |")

    if outcome.failures:
        lines += ["", "## Failures", ""]
        lines += [f"- `{f}`" for f in outcome.failures]
    lines.append("")
    return "\n".join(lines)


def write_report(outcome: GateOutcome, metrics: list[MetricResult], *,
                 backend: str = "stub", out_dir: Path = REPORTS_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = render_markdown(outcome, metrics, backend=backend)
    (out_dir / "gate_report.md").write_text(md, encoding="utf-8")
    payload = {
        "backend": backend,
        "outcome": outcome.to_dict(),
        "metrics": [m.to_dict() for m in metrics],
    }
    (out_dir / "gate_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"markdown": str(out_dir / "gate_report.md"), "json": str(out_dir / "gate_report.json")}
