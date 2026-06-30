#!/usr/bin/env bash
# Deploy the ADK benefits-finder agent to Vertex AI Agent Engine.
#
# Prereqs:
#   pip install -e ".[adk,vertex]"
#   gcloud auth application-default login
#   export GOOGLE_CLOUD_PROJECT=...  GOOGLE_CLOUD_LOCATION=us-central1  STAGING_BUCKET=gs://...
set -euo pipefail

: "${GOOGLE_CLOUD_PROJECT:?set GOOGLE_CLOUD_PROJECT}"
: "${STAGING_BUCKET:?set STAGING_BUCKET (gs://...)}"
LOCATION="${GOOGLE_CLOUD_LOCATION:-us-central1}"

if command -v adk >/dev/null 2>&1; then
  echo "Deploying via the adk CLI..."
  adk deploy agent_engine \
    --project="${GOOGLE_CLOUD_PROJECT}" \
    --region="${LOCATION}" \
    --staging_bucket="${STAGING_BUCKET}" \
    src/card_benefits_finder
else
  echo "adk CLI not found; deploying via the Python SDK..."
  EVAL_BACKEND=vertex python deploy/deploy_agent_engine.py
fi

echo "Done. Re-validate with:  EVAL_BACKEND=vertex python pipelines/run_all.py --layers vertex"
