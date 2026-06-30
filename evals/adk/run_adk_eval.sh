#!/usr/bin/env bash
#
# Layer 1 — ADK-native evaluation via the `adk eval` CLI.
#
# This script documents the CLI invocations for running the card_benefits_finder
# evals. The actual `adk eval` lines are COMMENTED OUT so the script is safe to
# source/read without google-adk installed. Uncomment the block you want and run.
#
# Prereqs:
#   - pip install -e ".[adk]"   (installs google-adk + the agent's deps)
#   - For the LLM-judge / rubric / safety / hallucination metrics in
#     eval_config.json, set GOOGLE_CLOUD_PROJECT and authenticate to Vertex AI:
#       export GOOGLE_CLOUD_PROJECT=your-project
#       export GOOGLE_CLOUD_LOCATION=us-central1
#       export GOOGLE_GENAI_USE_VERTEXAI=TRUE
#
set -euo pipefail

# --- Resolve repo-relative paths ---------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

AGENT_MODULE_DIR="${REPO_ROOT}/src/card_benefits_finder"   # dir containing the agent package
TEST_FILE="${REPO_ROOT}/evals/datasets/card_benefits_basic.test.json"
EVALSET_FILE="${REPO_ROOT}/evals/datasets/card_benefits.evalset.json"
TEST_CONFIG="${SCRIPT_DIR}/test_config.json"
EVAL_CONFIG="${SCRIPT_DIR}/eval_config.json"

# --- Guard: is the adk CLI available? ----------------------------------------
if ! command -v adk >/dev/null 2>&1; then
  echo "[run_adk_eval] 'adk' CLI not found."
  echo "  Install it with:  pip install -e \".[adk]\"  (from ${REPO_ROOT})"
  echo "  Then uncomment one of the 'adk eval' invocations in this script."
  echo
  echo "  Paths this script would use:"
  echo "    agent module dir : ${AGENT_MODULE_DIR}"
  echo "    test file        : ${TEST_FILE}"
  echo "    evalset file     : ${EVALSET_FILE}"
  echo "    simple config    : ${TEST_CONFIG}"
  echo "    full config      : ${EVAL_CONFIG}"
  exit 0
fi

echo "[run_adk_eval] adk found: $(command -v adk)"
echo "[run_adk_eval] Uncomment a block below to actually run an evaluation."

# -----------------------------------------------------------------------------
# 1) Single-session .test.json with the simple/local config
#    (tool_trajectory_avg_score + response_match_score — no Vertex needed).
# -----------------------------------------------------------------------------
# adk eval \
#   "${AGENT_MODULE_DIR}" \
#   "${TEST_FILE}" \
#   --config_file_path "${TEST_CONFIG}" \
#   --num_runs 2 \
#   --print_detailed_results

# -----------------------------------------------------------------------------
# 2) Multi-session evalset with the FULL config (LLM-judge + rubric + custom
#    metric). Requires GOOGLE_CLOUD_PROJECT / Vertex for the v2/rubric metrics.
# -----------------------------------------------------------------------------
# adk eval \
#   "${AGENT_MODULE_DIR}" \
#   "${EVALSET_FILE}" \
#   --config_file_path "${EVAL_CONFIG}" \
#   --num_runs 2 \
#   --print_detailed_results

# -----------------------------------------------------------------------------
# 3) Run only specific eval cases from an evalset (comma-separated eval_ids):
# -----------------------------------------------------------------------------
# adk eval \
#   "${AGENT_MODULE_DIR}" \
#   "${EVALSET_FILE}:world_benefits_drilldown_then_eligibility,money_back_purchase_protection" \
#   --config_file_path "${EVAL_CONFIG}" \
#   --print_detailed_results

# -----------------------------------------------------------------------------
# Web UI workflow (interactive authoring + running of eval cases):
#   1. From the repo root, launch the dev UI over the agents dir:
#        adk web src
#   2. Open the printed URL, select the 'card_benefits_finder' app, and chat.
#   3. Switch to the 'Eval' tab and click "Add current session" to capture the
#      current conversation (tool calls + final responses) as a new eval case.
#   4. Pick a metric config and click "Run Evaluation" to score the saved cases;
#      inspect per-invocation tool-trajectory and response diffs inline.
# -----------------------------------------------------------------------------
