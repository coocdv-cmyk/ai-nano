"""Standardised scientific metrics and the failure taxonomy."""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable

# Non-parametric 95% half-width for a normal-ish mean; with few seeds this is
# reported as *indicative only*, which is why per-seed values are preserved too.
CI_95 = 1.959963985

FAILURE_TAXONOMY = (
    "success",
    "vocabulary-limitation",
    "representation-limitation",
    "search-explosion",
    "incorrect-candidate",
    "validation-failure",
    "overgeneralization",
    "undergeneralization",
    "insufficient-evidence",
    "non-identifiable",
    "time-exhaustion",
    "memory-exhaustion",
    "concept-interference",
    "execution-error",
)


def mean_ci(values: Iterable[float], *, ci: float = CI_95) -> dict[str, float | None]:
    xs = [float(v) for v in values]
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "ci95": None}
    mu = statistics.fmean(xs)
    sd = statistics.pstdev(xs) if n > 1 else 0.0
    half = ci * sd / math.sqrt(n)
    return {"n": n, "mean": mu, "std": sd, "ci95": half,
            "min": min(xs), "max": max(xs), "median": statistics.median(xs)}


def quantiles(values: Iterable[float], qs=(0.25, 0.5, 0.75, 0.9)) -> dict[str, float | None]:
    xs = sorted(float(v) for v in values)
    n = len(xs)
    if n == 0:
        return {f"q{q}": None for q in qs}

    def _q(q: float) -> float:
        pos = q * (n - 1)
        lo = int(math.floor(pos))
        hi = min(n - 1, lo + 1)
        frac = pos - lo
        return xs[lo] * (1.0 - frac) + (xs[hi] * frac)

    return {f"q{int(q * 100)}": _q(q) for q in qs}


def aggregate(values: Iterable[float], *, ci: float = CI_95) -> dict[str, Any]:
    xs = [float(v) for v in values]
    out = mean_ci(xs, ci=ci)
    out.update(quantiles(xs))
    return out


def failure_counts(rows: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {k: 0 for k in FAILURE_TAXONOMY}
    for r in rows:
        key = r.get("failure_category") if isinstance(r, dict) else getattr(r, "failure_category", None)
        counts[key if key in counts else "incorrect-candidate"] += 1
    return {k: v for k, v in counts.items() if v}


def mean(values: Iterable[float]) -> float | None:
    xs = list(values)
    return statistics.fmean(xs) if xs else None


def safe_div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return num / den
