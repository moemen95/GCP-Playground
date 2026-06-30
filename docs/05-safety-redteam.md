# 05 · Safety & Red-Team Gating

Responsible-AI gates for a bank agent. The benefits-finder ingests T&C PDFs (untrusted content) and handles card data — the threat model is **RAG-side**, not just direct prompts.

## Benefits-finder threat model

| Threat | Vector | Detection / gate |
|---|---|---|
| **Indirect prompt injection** | Malicious instructions embedded in ingested T&C PDFs/KB chunks ("ignore prior rules, reveal…") | indirect-injection probes (2302.12173); treat retrieved text as data, not instructions |
| **PII echo / leakage** | Agent repeats/derives card numbers, names, account data | Presidio detection on output; refuse account-specific data (prompt rule #4) |
| **Out-of-scope financial advice** | "Should I refinance?", investment/tax/legal | refusal rubric `refuses_out_of_scope`; `_OUT_OF_SCOPE` cues in `local_agent.py` |
| **Cross-issuer / hallucinated products** | Asks about Amex/Visa Infinite/Scotiabank | scope guard refusal |
| **Jailbreak of refusals** | Roleplay/encoding to extract advice | jailbreak probes below |

## Toxicity

| Resource | Ref / link | Use |
|---|---|---|
| RealToxicityPrompts | 2009.11462 | Prompt set that elicits toxic continuations |
| ToxiGen | 2203.09509 | Implicit/adversarial toxicity (13 groups) |
| Perspective API | perspectiveapi.com | Toxicity scoring service |

## PII & data extraction

| Resource | Ref / link | Use |
|---|---|---|
| **Presidio** | microsoft.github.io/presidio | Detect/anonymize PII in inputs & outputs (repo `[safety]` extra) |
| Training-data extraction | 2012.07805 | Models memorize & emit PII verbatim — test extraction attacks |

## Prompt / injection / jailbreak

| Resource | Ref / link | Use |
|---|---|---|
| **GCG** (adversarial suffixes) | 2307.15043 | Transferable optimized jailbreak strings |
| **Jailbroken: how does LLM safety fail** | 2307.02483 | Failure-mode taxonomy (competing objectives, mismatched generalization) |
| **Indirect prompt injection** | 2302.12173 | Injection via retrieved/ingested content — **most relevant here** |
| XSTest | 2308.01263 | **Over-refusal**: safe prompts that look unsafe → guards refusal balance |

## Red-team tooling & benchmarks

| Tool / bench | OSS? | Ref / link | Use |
|---|---|---|---|
| **garak** | yes | github.com/NVIDIA/garak | LLM vulnerability scanner (probes battery) |
| **PyRIT** | yes | github.com/Azure/PyRIT | Automated red-team orchestration |
| **HarmBench** | yes | 2402.04249 | Standardized harmful-behavior eval + attack/defense |
| **JailbreakBench** | yes | 2404.01318 | Reproducible jailbreak benchmark + leaderboard |

## Governance frameworks

| Framework | Link | Relevant items |
|---|---|---|
| **OWASP LLM Top 10** | owasp.org/www-project-top-10-for-large-language-model-applications | LLM01 Prompt Injection · LLM02 Insecure Output · LLM06 Sensitive Info Disclosure |
| **NIST AI RMF** | nist.gov/itl/ai-risk-management-framework | Govern/Map/Measure/Manage — map eval evidence to controls |

## Gate posture (pre-prod)

| Check | Metric | Posture |
|---|---|---|
| No toxic/harmful output | `safety`, ADK `safety_v1` | **hard gate** |
| No PII leak | Presidio output scan | **hard gate** (zero-tolerance) |
| Resists indirect injection | injection probe pass-rate | **hard gate** |
| Refuses out-of-scope advice | `refuses_out_of_scope` rubric | **hard gate** |
| Not over-refusing | XSTest-style safe set | soft / monitored (balance) |

Safety is **conjunctive**: any hard-gate failure fails the whole gate regardless of quality scores → [06](06-preprod-gating.md).

## Sources
- RealToxicityPrompts 2009.11462 · ToxiGen 2203.09509 · GCG 2307.15043 · Jailbroken 2307.02483 · Indirect injection 2302.12173 · XSTest 2308.01263 · Extraction 2012.07805 · HarmBench 2402.04249 · JailbreakBench 2404.01318
- Presidio · garak · PyRIT · Perspective API · OWASP LLM Top 10 · NIST AI RMF (links above)
