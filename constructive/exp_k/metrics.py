"""EXP-K metrics: dispersion, confidence intervals and discrimination measures."""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable

CI_Z = 1.959963985


def stats(values: Iterable[float | None]) -> dict[str, Any]:
    xs = [float(v) for v in values if v is not None]
    n = len(xs)
    if not n:
        return {"n": 0, "mean": None, "median": None, "std": None,
                "ci95": None, "min": None, "max": None}
    sd = statistics.pstdev(xs) if n > 1 else 0.0
    return {
        "n": n, "mean": round(statistics.fmean(xs), 6),
        "median": round(statistics.median(xs), 6), "std": round(sd, 6),
        "ci95": round(CI_Z * sd / math.sqrt(n), 6),
        "min": round(min(xs), 6), "max": round(max(xs), 6),
    }


def rate(flags: Iterable[bool]) -> dict[str, Any]:
    return stats([1.0 if bool(f) else 0.0 for f in flags])


def ece(confidences: list[float], correct: list[int], n_bins: int = 10) -> float:
    if not confidences:
        return 0.0
    total = 0.0
    for b in range(n_bins):
        idx = [i for i, c in enumerate(confidences)
               if min(n_bins - 1, int(c * n_bins)) == b]
        if not idx:
            continue
        avg_conf = sum(confidences[i] for i in idx) / len(idx)
        avg_corr = sum(correct[i] for i in idx) / len(idx)
        total += len(idx) / len(confidences) * abs(avg_conf - avg_corr)
    return round(total, 6)


def brier(confidences: list[float], correct: list[int]) -> float | None:
    if not confidences:
        return None
    return round(sum((c - y) ** 2 for c, y in zip(confidences, correct)) / len(confidences), 6)


def nll(confidences: list[float], correct: list[int], eps: float = 1e-9) -> float | None:
    if not confidences:
        return None
    total = 0.0
    for c, y in zip(confidences, correct):
        p = min(1 - eps, max(eps, c))
        total += -(math.log(p) if y else math.log(1 - p))
    return round(total / len(confidences), 6)


def accuracy_coverage(correct: int, answered: int, total: int) -> dict[str, Any]:
    return {
        "total": total, "answered": answered, "correct": correct,
        "accuracy": round(correct / total, 6) if total else None,
        "coverage": round(answered / total, 6) if total else None,
        "selective_accuracy": round(correct / answered, 6) if answered else None,
    }


def confusion(rows: list[dict[str, Any]], key: str = "outcome") -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        k = str(r.get(key, "unknown"))
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def compare(baseline: Iterable[float], treatment: Iterable[float], *,
            lower_is_better: bool = False) -> dict[str, Any]:
    b, t = stats(baseline), stats(treatment)
    out = {"baseline": b, "treatment": t, "lower_is_better": lower_is_better}
    if b["mean"] is not None and t["mean"] is not None:
        d = t["mean"] - b["mean"]
        out["absolute_delta"] = round(d, 6)
        out["improved"] = d < 0 if lower_is_better else d > 0
    else:
        out["absolute_delta"] = None
        out["improved"] = None
    return out
