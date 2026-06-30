"""Evaluation layers for the card-benefits-finder agent.

Layer 1 (adk)     — ADK-native eval
Layer 2 (vertex)  — Vertex AI Gen AI Evaluation Service
Layer 3 (judges)  — custom LLM-as-judge + RAG + safety
Layer 4 (gating)  — pre-prod gating + CI/CD
common            — shared backend, text utils, dataset loaders
"""
