"""Pluggable LLM backend used by the agent's offline twin and the LLM-judges.

Three backends, chosen by ``EVAL_BACKEND``:

* ``stub``   — deterministic, offline; scores via lexical heuristics so the whole
               judge/gate pipeline runs end-to-end with no credentials.
* ``gemini`` — direct Gemini API (``google-genai`` + ``GOOGLE_API_KEY``).
* ``vertex`` — Gemini on Vertex AI / Agent Platform (``google-genai`` Vertex mode).

Judges depend only on the abstract :class:`ModelBackend` interface, so the same
judge code runs offline (stub) and live (gemini/vertex). Tests inject their own
fake backend for full control.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Sequence

from .text_utils import containment, jaccard, rouge_l_f1


@dataclass
class LLMResult:
    text: str
    # Optional score-token distribution {score: prob}; used by G-Eval prob-weighting.
    score_distribution: dict[int, float] | None = None


@dataclass
class RatingResult:
    score: float            # raw score on the requested scale
    scale: tuple[int, int]  # (low, high)
    rationale: str = ""
    distribution: dict[int, float] = field(default_factory=dict)

    @property
    def normalized(self) -> float:
        lo, hi = self.scale
        if hi == lo:
            return 0.0
        return (self.score - lo) / (hi - lo)


class ModelBackend:
    """Abstract LLM backend. Subclasses implement :meth:`complete`."""

    name = "abstract"

    def complete(self, prompt: str, *, system: str | None = None,
                 temperature: float = 0.0) -> LLMResult:
        raise NotImplementedError

    # ---- convenience judging primitives (default impls call complete) -------
    def rate(self, instruction: str, *, response: str, reference: str | None = None,
             context: str | None = None, scale: tuple[int, int] = (1, 5)) -> RatingResult:
        lo, hi = scale
        prompt = self._rating_prompt(instruction, response, reference, context, scale)
        out = self.complete(prompt, temperature=0.0)
        score = self._parse_score(out.text, scale)
        dist = out.score_distribution or {int(round(score)): 1.0}
        return RatingResult(score=score, scale=scale, rationale=out.text.strip(), distribution=dist)

    def choose(self, instruction: str, *, response_a: str, response_b: str,
               criteria: str | None = None) -> tuple[str, str]:
        prompt = self._pairwise_prompt(instruction, response_a, response_b, criteria)
        out = self.complete(prompt, temperature=0.0)
        verdict = self._parse_choice(out.text)
        return verdict, out.text.strip()

    def classify(self, text: str, labels: Sequence[str], *, instruction: str) -> tuple[str, str]:
        prompt = (f"{instruction}\n\nLabels: {', '.join(labels)}\n\nText:\n{text}\n\n"
                  f"Answer with exactly one label.")
        out = self.complete(prompt, temperature=0.0)
        chosen = self._parse_label(out.text, labels)
        return chosen, out.text.strip()

    # ---- prompt builders / parsers (shared) --------------------------------
    @staticmethod
    def _rating_prompt(instruction, response, reference, context, scale) -> str:
        lo, hi = scale
        parts = [instruction, f"\nScore on an integer scale from {lo} to {hi}."]
        if context:
            parts.append(f"\nContext (ground truth):\n{context}")
        if reference:
            parts.append(f"\nReference answer:\n{reference}")
        parts.append(f"\nResponse to score:\n{response}")
        parts.append('\nReturn your reasoning, then a final line "Rating: <number>".')
        return "\n".join(parts)

    @staticmethod
    def _pairwise_prompt(instruction, a, b, criteria) -> str:
        crit = f"\nCriteria: {criteria}" if criteria else ""
        return (f"{instruction}{crit}\n\n[Response A]\n{a}\n\n[Response B]\n{b}\n\n"
                'Decide which is better. End with a line "Verdict: A", "Verdict: B", '
                'or "Verdict: tie".')

    @staticmethod
    def _parse_score(text: str, scale: tuple[int, int]) -> float:
        lo, hi = scale
        m = re.search(r"rating\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
        if not m:
            nums = re.findall(r"\b([0-9]+(?:\.[0-9]+)?)\b", text)
            if not nums:
                return float(lo)
            val = float(nums[-1])
        else:
            val = float(m.group(1))
        return max(lo, min(hi, val))

    @staticmethod
    def _parse_choice(text: str) -> str:
        m = re.search(r"verdict\s*[:=]\s*(a|b|tie)", text, re.IGNORECASE)
        if m:
            return m.group(1).upper() if m.group(1).lower() != "tie" else "tie"
        return "tie"

    @staticmethod
    def _parse_label(text: str, labels: Sequence[str]) -> str:
        low = text.lower()
        for lab in labels:
            if lab.lower() in low:
                return lab
        return labels[0]


class StubBackend(ModelBackend):
    """Deterministic offline backend.

    ``complete`` echoes a stable hash so it never raises; the judging primitives
    are overridden with lexical heuristics so offline scores are *meaningful*
    (groundedness rewards overlap with context, correctness rewards overlap with
    the reference) rather than random.
    """

    name = "stub"

    def complete(self, prompt: str, *, system: str | None = None,
                 temperature: float = 0.0) -> LLMResult:
        h = hashlib.sha256((system or "" + prompt).encode()).hexdigest()[:8]
        return LLMResult(text=f"[stub:{h}] deterministic offline completion")

    def rate(self, instruction, *, response, reference=None, context=None,
             scale=(1, 5)) -> RatingResult:
        lo, hi = scale
        # Heuristic signal: prefer grounding context, then reference, then a
        # stable prior derived from the instruction so different metrics differ.
        if context:
            signal = containment(response, context)
        elif reference:
            signal = rouge_l_f1(response, reference)
        else:
            seed = int(hashlib.sha256(instruction.encode()).hexdigest(), 16) % 1000
            signal = 0.55 + (seed / 1000) * 0.35  # 0.55..0.90, deterministic
        score = lo + signal * (hi - lo)
        rounded = int(round(score))
        # Peaked distribution around the score (for G-Eval prob-weighting demos).
        dist = {}
        for s in range(lo, hi + 1):
            dist[s] = max(0.0, 1.0 - abs(s - score))
        total = sum(dist.values()) or 1.0
        dist = {s: p / total for s, p in dist.items()}
        return RatingResult(score=round(score, 3), scale=scale,
                            rationale=f"[stub] signal={signal:.2f}", distribution=dist)

    def choose(self, instruction, *, response_a, response_b, criteria=None) -> tuple[str, str]:
        ref = criteria or instruction
        sa, sb = jaccard(response_a, ref), jaccard(response_b, ref)
        if abs(sa - sb) < 1e-3:
            return "tie", "[stub] equal overlap"
        return ("A" if sa > sb else "B"), f"[stub] overlap A={sa:.2f} B={sb:.2f}"

    def classify(self, text, labels, *, instruction) -> tuple[str, str]:
        best = max(labels, key=lambda lab: jaccard(text, lab))
        return best, f"[stub] nearest label by overlap: {best}"


class _GenAIBackend(ModelBackend):
    """Shared implementation for the Gemini-API and Vertex backends.

    Both use the ``google-genai`` SDK; only client construction differs.
    """

    def __init__(self, model: str, *, vertexai: bool):
        from google import genai  # lazy: only when a live backend is requested

        self.model = model
        self.name = "vertex" if vertexai else "gemini"
        if vertexai:
            self._client = genai.Client(
                vertexai=True,
                project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
                location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
            )
        else:
            self._client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    def complete(self, prompt: str, *, system: str | None = None,
                 temperature: float = 0.0) -> LLMResult:
        from google.genai import types

        cfg = types.GenerateContentConfig(temperature=temperature,
                                          system_instruction=system)
        resp = self._client.models.generate_content(
            model=self.model, contents=prompt, config=cfg)
        return LLMResult(text=resp.text or "")


def get_backend(name: str | None = None, model: str | None = None) -> ModelBackend:
    """Return the configured backend. Defaults to ``EVAL_BACKEND`` (``stub``)."""
    name = (name or os.environ.get("EVAL_BACKEND", "stub")).strip().lower()
    model = model or os.environ.get("JUDGE_MODEL", "gemini-2.5-flash")
    if name == "stub":
        return StubBackend()
    if name == "gemini":
        return _GenAIBackend(model, vertexai=False)
    if name == "vertex":
        return _GenAIBackend(model, vertexai=True)
    raise ValueError(f"Unknown backend {name!r}")
