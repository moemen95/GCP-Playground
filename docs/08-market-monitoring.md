# 08 · Market: Agent Observability & Monitoring

Closing the loop from offline golden sets ([06](06-preprod-gating.md)) to live production. The benefits-finder needs prod tracing of tool calls + retrieved chunks, online faithfulness/safety scoring, and drift alerts vs the release baseline.

## Platforms

| Tool | OSS? | Tracing | Online evals | Drift / monitoring | Notes |
|---|---|---|---|---|---|
| **LangSmith** | no | yes | yes (online evaluators) | yes | LangChain-native; annotation queues, datasets, experiments |
| **Arize Phoenix** | **yes** | yes (**OTel-native / OpenInference**) | yes | yes (embedding/data drift) | Self-host; strong drift + RAG analytics |
| **Langfuse** | **yes** | yes (OTel) | yes (scores/evals) | yes | Self-host; prompt mgmt, datasets, human annotation |
| **Braintrust** | no | yes | yes | yes | Eval-centric; scoring playground, CI loop |
| **Confident AI / DeepEval** | DeepEval OSS | yes | yes | yes | DeepEval (OSS) for offline + Confident AI hosted dashboard |
| **promptfoo** | **yes** | partial | offline-first | via CI | Strong CI regression gates, red-team probes ([05](05-safety-redteam.md)) |

## Standards (instrument once, ship anywhere)

| Standard | Link | What |
|---|---|---|
| **OpenTelemetry GenAI semantic conventions** | opentelemetry.io/docs/specs/semconv/gen-ai | Standard span attrs for LLM/agent calls (model, tokens, tool calls) |
| **OpenInference** | github.com/Arize-ai/openinference | OTel-compatible spec for LLM/agent/RAG traces (Phoenix) |
| **OpenLLMetry** | github.com/traceloop/openllmetry | OTel SDK/instrumentation for LLM apps |

Emit OTel GenAI spans from the agent → any of the above backends ingests them. Avoids vendor lock-in.

## Closing the offline ↔ online loop

| Stage | Action | Benefits-finder |
|---|---|---|
| 1. Trace | Capture every prod turn: query, trajectory, retrieved chunks, response | tool calls + `context_for()` output as span attrs |
| 2. **Online eval** | Score sampled live traffic with the same judges as offline | groundedness, safety, refusal on prod traffic |
| 3. **Drift** | Compare online score dist vs golden baseline; alert on regression | KB updates / model upgrades shifting faithfulness |
| 4. **Annotation queue** | Humans label low-confidence / flagged turns | calibrate judges, find new failure modes |
| 5. Feed back | Promote hard prod cases into the **golden set** | grows coverage; next CI gate catches the regression |

This makes the golden set a living asset, and online eval the early-warning system between releases.

## Where GCP fits

| GCP surface | Role | Link |
|---|---|---|
| **Agent Engine online evaluation** | Continuous scoring of deployed-agent live traffic | https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview |
| **Vertex AI Experiments** | Track eval runs/metrics across versions (`VERTEX_EXPERIMENT`) | https://cloud.google.com/vertex-ai/docs/experiments/intro-vertex-ai-experiments |
| **Unified Trace Viewer** (Vertex / Cloud Trace) | Inspect agent traces, tool calls, latency | https://cloud.google.com/trace/docs |

GCP-native covers stages 2–3 for agents already on Agent Engine; OTel export bridges to Phoenix/Langfuse if a single pane across stacks is wanted.

## Sources
- LangSmith: https://docs.smith.langchain.com · Phoenix: https://github.com/Arize-ai/phoenix · Langfuse: https://langfuse.com
- Braintrust: https://www.braintrust.dev · DeepEval/Confident AI: https://docs.confident-ai.com · promptfoo: https://promptfoo.dev
- OTel GenAI semconv · OpenInference · OpenLLMetry (links above) · Vertex Agent Engine / Experiments / Cloud Trace (links above)
