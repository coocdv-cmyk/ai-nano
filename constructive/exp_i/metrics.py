"""Standardised scientific metrics for EXP-I.

Every comparison reports n, mean, median, std, 95% CI, baseline, new result,
absolute/relative improvement and failure count. Skipped experiments stay
skipped; missing measurements are never fabricated.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable

CI_Z = 1.959963985


def mean_ci(values: Iterable[float | None], *, z: float = CI_Z) -> dict[str, Any]:
    xs = [float(v) for v in values if v is not None]
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "ci95": None,
                "median": None, "min": None, "max": None}
    mu = statistics.fmean(xs)
    sd = statistics.pstdev(xs) if n > 1 else 0.0
    half = z * sd / math.sqrt(n)
    return {
        "n": n, "mean": round(mu, 6), "std": round(sd, 6),
        "ci95": round(half, 6), "median": round(statistics.median(xs), 6),
        "min": round(min(xs), 6), "max": round(max(xs), 6),
    }


def rates(flags: Iterable[bool]) -> dict[str, Any]:
    xs = [bool(v) for v in flags]
    return mean_ci([1.0 if x else 0.0 for x in xs])


def accuracy_coverage(correct: int, answered: int, total: int) -> dict[str, Any]:
    return {
        "total": total, "answered": answered, "correct": correct,
        "accuracy": correct / total if total else None,
        "coverage": answered / total if total else None,
        "selective_accuracy": correct / answered if answered else None,
    }


def compare(baseline: Iterable[float], new: Iterable[float], *,
            lower_is_better: bool = False) -> dict[str, Any]:
    b = mean_ci(baseline)
    n = mean_ci(new)
    out: dict[str, Any] = {"baseline": b, "new": n, "lower_is_better": lower_is_better}
    if b["mean"] is not None and n["mean"] is not None:
        diff = n["mean"] - b["mean"]
        out["absolute_improvement"] = round(diff, 6)
        out["relative_improvement"] = (round(diff / abs(b["mean"]), 6)
                                       if b["mean"] else None)
        out["improved"] = diff < 0 if lower_is_better else diff > 0
    else:
        out["absolute_improvement"] = None
        out["relative_improvement"] = None
        out["improved"] = None
    return out


def count_failures(rows: list[dict[str, Any]], *, state_key: str = "state") -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        key = str(r.get(state_key, r.get("failure_category", "unknown")))
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def taxonomy(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Failure taxonomy aligned with EXP-H categories plus EXP-I states."""
    cats = [
        "success", "wrong-acceptance", "false-rejection", "non-identifiable",
        "insufficient-evidence", "contradictory", "resource-limited",
        "representation-limitation", "compute-limited", "search-explosion",
        "false-retrieval", "concept-interference", "false-repair",
        "false-repair-detected", "unverifiable", "no-repair", "false-acceptance",
        "undergeneralization", "false-abstraction", "time-exhaustion",
        "execution-error", "skipped",
    ]
    out = {c: 0 for c in cats}
    for r in rows:
        key = r.get("failure_category") or r.get("state") or "execution-error"
        out[key if key in out else "execution-error"] += 1
    return {k: v for k, v in out.items() if v}
