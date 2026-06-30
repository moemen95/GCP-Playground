"""CI entrypoint: run the offline gate and exit nonzero on failure.

Used by .github/workflows/eval-gate.yml. Runs with the stub backend by default so
CI needs no GCP credentials.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipelines.run_all import run  # noqa: E402


def main() -> int:
    outcome, metrics, paths = run()
    status = "PASS" if outcome.passed else "FAIL"
    print(f"::group::Eval gate [{outcome.mode}] -> {status}")
    for h in outcome.hard:
        mark = "ok" if h["passed"] else "FAIL"
        score = h.get("score", "missing")
        print(f"  hard {h['name']}: {score} (>= {h['threshold']}) [{mark}]")
    print(f"  soft composite: {outcome.soft['composite']} (>= {outcome.soft['threshold']})")
    print("::endgroup::")
    if not outcome.passed:
        print(f"::error::Eval gate FAILED: {', '.join(outcome.failures)}")
    print(f"Report written to {paths['markdown']}")
    return 0 if outcome.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
