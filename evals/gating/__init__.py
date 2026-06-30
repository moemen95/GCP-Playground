"""Layer 4 — pre-prod gating + CI/CD.

* ``deterministic_metrics`` — offline trajectory / tool-call / response-match
  scorers (the no-LLM tier; mirrors what ADK & Vertex compute live).
* ``gate`` — tiered conjunctive hard-gates + weighted soft composite.
* ``aggregate`` — composite scoring + baseline no-regression check.
* ``synth_data`` — synthetic eval-set generation from the knowledge base.
* ``report`` — unified Markdown + JSON gate report.
"""
