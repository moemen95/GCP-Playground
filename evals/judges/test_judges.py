"""Offline tests for the Layer-3 judges (run with the deterministic stub backend)."""
from __future__ import annotations

from evals.common import load_jsonl
from evals.common.model_backend import StubBackend, get_backend

from evals.judges import bias_mitigation, calibration, geval, jury, nli_groundedness, rag_metrics, safety_suite


def _qa():
    return load_jsonl("rag_qa.jsonl")


def _safety():
    return load_jsonl("safety_redteam.jsonl")


# ---- G-Eval ----------------------------------------------------------------
def test_geval_probability_weighted_is_continuous():
    rows = _qa()
    s = geval.g_eval(criterion_name="groundedness", response=rows[0]["response"],
                     question=rows[0]["question"], context=rows[0]["context"])
    assert 0.0 <= s.normalized <= 1.0
    # probability-weighted expected score need not equal the argmax integer
    assert s.expected_score != s.argmax_score or len(s.steps) >= 3


def test_geval_dataset_grounded_scores_high():
    res = geval.g_eval_dataset(_qa(), criterion_name="groundedness")
    assert res.uses_llm_judge and res.n == len(_qa())
    assert res.score > 0.5  # grounded transcripts should score well


# ---- Jury ------------------------------------------------------------------
def test_jury_aggregates_panel():
    res = jury.jury_dataset(_qa(), instruction="Rate correctness and grounding.")
    assert len(res.details["panel"]) == 3
    assert 0.0 <= res.score <= 1.0


# ---- bias mitigation -------------------------------------------------------
def test_balanced_pairwise_consistency_field():
    out = bias_mitigation.balanced_pairwise(
        "Which answer is more grounded?",
        response_a="The limit is CAD 1,000 per claim.",
        response_b="The limit is generous and depends on many wonderful factors.",
        criteria="grounded, specific, cites the limit")
    assert out.verdict in {"A", "B", "tie"}
    assert isinstance(out.consistent, bool)


def test_length_controlled_winrate_neutralizes_length():
    pairs = load_jsonl("pairwise_baseline.jsonl")
    out = bias_mitigation.length_controlled_winrate(pairs)
    assert 0.0 <= out["length_controlled_winrate"] <= 1.0
    assert out["n"] == len(pairs)


# ---- calibration -----------------------------------------------------------
def test_cohen_kappa_perfect_and_none():
    assert abs(calibration.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) - 1.0) < 1e-9
    assert calibration.cohen_kappa([1, 1, 1, 1], [1, 1, 1, 1]) == 1.0  # degenerate -> 1
    assert calibration.interpret_kappa(0.85) == "almost perfect"
    assert calibration.interpret_kappa(0.5) == "moderate"


def test_spearman_monotonic():
    assert calibration.spearman([1, 2, 3, 4], [10, 20, 30, 40]) > 0.99
    assert calibration.spearman([1, 2, 3, 4], [40, 30, 20, 10]) < -0.99


def test_binary_calibration_tpr_tnr():
    c = calibration.binary_calibration([1, 1, 0, 0], [1, 0, 0, 0])
    assert c.tpr == 0.5 and c.tnr == 1.0 and c.n == 4


# ---- RAG -------------------------------------------------------------------
def test_rag_triad_faithfulness_high_on_grounded():
    # RAG metrics apply to grounded answers, not refusals.
    answered = [r for r in _qa() if not r["expected_refusal"]]
    metrics = {m.name: m for m in rag_metrics.rag_triad(answered)}
    assert metrics["faithfulness"].score >= 0.7
    assert "context_precision" in metrics and "answer_relevancy" in metrics


def test_faithfulness_penalizes_ungrounded_claim():
    grounded = rag_metrics.faithfulness("The limit is CAD 1,000 per claim.",
                                        "Coverage is up to CAD 1,000 per claim.")
    invented = rag_metrics.faithfulness("The limit is one million dollars and includes a free car.",
                                        "Coverage is up to CAD 1,000 per claim.")
    assert grounded > invented


# ---- NLI groundedness ------------------------------------------------------
def test_hallucination_metric_flags_unsupported():
    res = nli_groundedness.hallucination_metric(_qa())
    assert 0.0 <= res.score <= 1.0
    rep = nli_groundedness.grounding_report(
        "The card gives you a free vacation to Mars every year.",
        "The card includes purchase assurance and extended warranty.")
    assert rep.accuracy < 1.0


# ---- safety ----------------------------------------------------------------
def test_pii_scan_detects_card_number():
    assert "card_number" in safety_suite.scan_pii("My card is 4111 1111 1111 1111")
    assert safety_suite.scan_pii("The coverage limit is CAD 1,000.") == {}


def test_injection_blocked_when_no_canary_leak():
    rows = [r for r in _safety() if r.get("suite") == "injection"]
    res = safety_suite.injection_metric(rows)
    assert res.score == 1.0  # stub agent never leaks the canary


def test_refusal_balance_on_scope_suite():
    rows = [r for r in _safety() if r.get("suite") == "scope"]
    s = safety_suite.refusal_balance(rows)
    assert s.out_of_scope_refusal_rate == 1.0
    assert s.harmful_compliance_rate == 0.0
    assert s.over_refusal_rate == 0.0


def test_get_backend_offline_is_stub():
    assert isinstance(get_backend("stub"), StubBackend)
