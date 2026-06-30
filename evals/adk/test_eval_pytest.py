"""Pytest entry point for ADK-native evaluation (Layer 1).

Two of these tests drive the real ADK ``AgentEvaluator`` against the agent
module ``card_benefits_finder`` and the datasets under ``evals/datasets/``. They
require ``google-adk`` to be installed (and, for the LLM-judge / rubric metrics,
``GOOGLE_CLOUD_PROJECT`` + Vertex access), so the whole ADK section is guarded
by ``pytest.importorskip`` — the suite still passes offline by skipping.

The final test exercises the custom metric's pure-python fallback helper and is
NOT skipped, so there is always offline coverage of the scoring logic.

Run only this layer::

    pytest evals/adk/test_eval_pytest.py -v
"""
from __future__ import annotations

import os

import pytest

# --- Paths --------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATASETS = os.path.normpath(os.path.join(_THIS_DIR, "..", "datasets"))

TEST_FILE = os.path.join(_DATASETS, "card_benefits_basic.test.json")
EVALSET_FILE = os.path.join(_DATASETS, "card_benefits.evalset.json")
TEST_CONFIG = os.path.join(_THIS_DIR, "test_config.json")
EVAL_CONFIG = os.path.join(_THIS_DIR, "eval_config.json")


# --- ADK integration tests (skipped when google-adk is absent) ----------------

@pytest.mark.asyncio
async def test_card_benefits_basic_test_file():
    """Run the single-session .test.json through ADK's AgentEvaluator."""
    pytest.importorskip("google.adk")
    from google.adk.evaluation.agent_evaluator import AgentEvaluator

    await AgentEvaluator.evaluate(
        agent_module="card_benefits_finder",
        eval_dataset_file_path_or_dir=TEST_FILE,
        num_runs=2,
    )


@pytest.mark.asyncio
async def test_card_benefits_evalset():
    """Run the multi-session evalset through ADK's AgentEvaluator."""
    pytest.importorskip("google.adk")
    from google.adk.evaluation.agent_evaluator import AgentEvaluator

    await AgentEvaluator.evaluate(
        agent_module="card_benefits_finder",
        eval_dataset_file_path_or_dir=EVALSET_FILE,
        num_runs=2,
    )


# --- Offline unit test of the custom metric's pure-python helper ---------------

def test_benefit_citation_helper_offline():
    """The pure-python scoring helper works without google-adk installed."""
    from evals.adk.metrics import score_benefit_citation_text

    # Full citation: names a limit, an eligibility condition, and a time window.
    strong = (
        "The Mobile Device Insurance covers up to CAD 1,000 per claim. "
        "The full retail price must be charged to the card, and coverage "
        "lasts for up to 730 days."
    )
    assert score_benefit_citation_text(strong) == pytest.approx(1.0)

    # A bare refusal cites none of the three dimensions.
    weak = "I'm sorry, I can't help with that request."
    assert score_benefit_citation_text(weak) == pytest.approx(0.0)

    # Partial: mentions a window only.
    partial = "It is covered for 90 days."
    assert 0.0 < score_benefit_citation_text(partial) < 1.0

    # Empty / missing text scores zero, never raises.
    assert score_benefit_citation_text("") == 0.0


def test_benefit_citation_score_offline_fallback():
    """The ADK entry point returns a usable result even without ADK installed."""
    from evals.adk.metrics import benefit_citation_score

    invocations = [
        {
            "final_response": {
                "parts": [
                    {
                        "text": (
                            "Purchase Assurance covers up to CAD 60,000; the item "
                            "must be purchased entirely on the card and is covered "
                            "for 90 days."
                        )
                    }
                ],
                "role": "model",
            }
        }
    ]

    result = benefit_citation_score(
        eval_metric=None,
        actual_invocations=invocations,
        expected_invocations=invocations,
        conversation_scenario=None,
    )

    # Offline fallback yields a dict; with ADK installed it's an EvaluationResult.
    if isinstance(result, dict):
        assert result["overall_score"] == pytest.approx(1.0)
        assert result["overall_eval_status"] == "PASSED"
        assert len(result["per_invocation_results"]) == 1
    else:  # pragma: no cover - only when google-adk is installed
        assert result.overall_score == pytest.approx(1.0)
