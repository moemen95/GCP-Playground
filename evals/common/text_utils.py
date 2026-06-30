"""Small, dependency-light text helpers shared by the judges and the stub backend.

These power the *deterministic* offline metrics (overlap, containment, ROUGE-L)
so the eval pipeline produces meaningful numbers without any model call.
"""
from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9]+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def jaccard(a: str, b: str) -> float:
    sa, sb = token_set(a), token_set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def containment(needle: str, haystack: str) -> float:
    """Fraction of ``needle`` tokens present in ``haystack`` (asymmetric overlap)."""
    sn = token_set(needle)
    if not sn:
        return 1.0
    sh = token_set(haystack)
    return len(sn & sh) / len(sn)


def split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l_f1(candidate: str, reference: str) -> float:
    """ROUGE-L F1 on the longest common subsequence of word tokens."""
    c, r = tokenize(candidate), tokenize(reference)
    if not c or not r:
        return 0.0
    lcs = _lcs_len(c, r)
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(c), lcs / len(r)
    return 2 * prec * rec / (prec + rec)
