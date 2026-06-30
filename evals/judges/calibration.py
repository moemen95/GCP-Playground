"""Judge↔human calibration metrics.

A judge is only trustworthy once it agrees with human labels on a held-out set.
Use Cohen's κ / raw agreement for binary or categorical verdicts, Spearman ρ for
ordinal scores, and TPR/TNR to see *which* side of the binary the judge gets
wrong (a judge can have high accuracy while missing the failures you care about).
"""
from __future__ import annotations

from dataclasses import dataclass


def raw_agreement(a: list, b: list) -> float:
    if not a:
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def cohen_kappa(a: list, b: list) -> float:
    """Chance-corrected agreement for two raters over categorical labels."""
    if not a:
        return 0.0
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = raw_agreement(a, b)
    # expected agreement under independence
    pe = 0.0
    for lab in labels:
        pa = sum(1 for x in a if x == lab) / n
        pb = sum(1 for x in b if x == lab) / n
        pe += pa * pb
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # average rank for ties (1-based)
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation (no scipy dependency)."""
    n = len(a)
    if n < 2:
        return 0.0
    ra, rb = _rank(a), _rank(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    if va == 0 or vb == 0:
        return 0.0
    return cov / (va * vb)


@dataclass
class BinaryCalibration:
    tpr: float          # recall on positives (caught failures, if positive=fail)
    tnr: float          # specificity
    precision: float
    accuracy: float
    n: int


def binary_calibration(y_true: list[int], y_pred: list[int]) -> BinaryCalibration:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    n = len(y_true)
    return BinaryCalibration(
        tpr=tp / (tp + fn) if (tp + fn) else 0.0,
        tnr=tn / (tn + fp) if (tn + fp) else 0.0,
        precision=tp / (tp + fp) if (tp + fp) else 0.0,
        accuracy=(tp + tn) / n if n else 0.0,
        n=n,
    )


# Landis–Koch interpretation bands for κ.
KAPPA_BANDS = [
    (0.81, "almost perfect"),
    (0.61, "substantial"),
    (0.41, "moderate"),
    (0.21, "fair"),
    (0.0, "slight"),
    (-1.0, "poor"),
]


def interpret_kappa(k: float) -> str:
    for lo, label in KAPPA_BANDS:
        if k >= lo:
            return label
    return "poor"
