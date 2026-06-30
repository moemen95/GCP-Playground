"""A dependency-free retriever over the benefits knowledge base.

Splits ``benefits_kb.md`` on ``## `` headers into chunks and ranks them against a
query with a simple TF/containment score. Good enough to give the agent (and the
groundedness evaluators) a real, citable source of truth without pulling in a
vector DB. Swap this for Vertex AI Search / a real vector store in production.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache

from ..config import BENEFITS_KB

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "of", "for", "to", "and", "or", "is", "are", "in", "on",
    "with", "your", "you", "it", "this", "that", "be", "as", "at", "by", "i",
    "my", "do", "does", "can", "what", "which", "how", "if", "when",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    title: str
    text: str

    def as_citation(self) -> str:
        return f"[{self.chunk_id}] {self.title}: {self.text}"


@lru_cache(maxsize=1)
def load_chunks() -> tuple[Chunk, ...]:
    raw = BENEFITS_KB.read_text(encoding="utf-8")
    # Drop HTML comments.
    raw = re.sub(r"<!--.*?-->", "", raw, flags=re.DOTALL)
    chunks: list[Chunk] = []
    for i, section in enumerate(raw.split("## ")):
        section = section.strip()
        if not section:
            continue
        title, _, body = section.partition("\n")
        body = " ".join(body.split())
        if not body:
            continue
        chunks.append(Chunk(chunk_id=f"kb-{i:02d}", title=title.strip(), text=body))
    return tuple(chunks)


@lru_cache(maxsize=1)
def _idf() -> dict[str, float]:
    chunks = load_chunks()
    n = len(chunks)
    df: dict[str, int] = {}
    for c in chunks:
        for tok in set(tokenize(f"{c.title} {c.text}")):
            df[tok] = df.get(tok, 0) + 1
    return {tok: math.log((n + 1) / (d + 0.5)) for tok, d in df.items()}


def search(query: str, k: int = 3) -> list[Chunk]:
    """Return the top-``k`` chunks for ``query`` ranked by TF-IDF overlap."""
    idf = _idf()
    q_tokens = tokenize(query)
    scored: list[tuple[float, Chunk]] = []
    for c in load_chunks():
        doc_tokens = tokenize(f"{c.title} {c.title} {c.text}")  # weight the title
        tf: dict[str, int] = {}
        for t in doc_tokens:
            tf[t] = tf.get(t, 0) + 1
        score = sum(tf.get(t, 0) * idf.get(t, 0.0) for t in q_tokens)
        if score > 0:
            scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]]


def context_for(query: str, k: int = 3) -> str:
    """Concatenated citation block used as grounding context for the model."""
    hits = search(query, k=k)
    if not hits:
        return ""
    return "\n".join(c.as_citation() for c in hits)
