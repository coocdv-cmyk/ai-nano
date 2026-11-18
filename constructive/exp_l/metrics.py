"""Shared, auditable metric definitions for EXP-L.

Every metric here is a pure function of declared inputs.  No oracle label is
ever passed to an operational decision function; ground truth is used only by
the evaluation functions in this module and in the stage runners.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence


def safe_rate(num: int, den: int) -> float | None:
    """Return None (undefined) for a zero denominator, never 0 or 1."""
    return None if den == 0 else num / den


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    precision = safe_rate(tp, tp + fp)
    recall = safe_rate(tp, tp + fn)
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """Rank-based AUROC with average ranks for ties (Mann-Whitney form)."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    combined = sorted((s, y) for s, y in zip(scores, labels))
    ranks: dict[int, float] = {}
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j + 2) / 2.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_sum = sum(ranks[k] for k, (_, y) in enumerate(combined) if y == 1)
    n1, n0 = len(pos), len(neg)
    return (rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def detection_result(flags: Sequence[bool], drift_idx: int,
                     recover_idx: int) -> dict[str, Any]:
    """Detection outcome for one stream with explicit phase boundaries."""
    drift_flags = list(flags[drift_idx:recover_idx])
    base_flags = list(flags[:drift_idx])
    detected = any(drift_flags)
    false_alarm = any(base_flags)
    delay = None
    for i, f in enumerate(drift_flags):
        if f:
            delay = i
            break
    return {
        "detected": detected,
        "false_alarm": false_alarm,
        "detection_delay": delay,
        "drift_hit_in_window": detected,
    }


def ece(confidences: Sequence[float], correct: Sequence[int],
        n_bins: int = 10) -> float:
    """Equal-width binned expected calibration error (evaluation only)."""
    n = len(confidences)
    if n == 0:
        return 0.0
    total = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, c in enumerate(confidences)
               if (lo <= c < hi or (b == n_bins - 1 and c == hi))]
        if not idx:
            continue
        mc = sum(confidences[i] for i in idx) / len(idx)
        ma = sum(correct[i] for i in idx) / len(idx)
        total += (len(idx) / n) * abs(mc - ma)
    return total


def brier(confidences: Sequence[float], correct: Sequence[int]) -> float:
    if not confidences:
        return 0.0
    return sum((c - y) ** 2 for c, y in zip(confidences, correct)) / len(confidences)


def nll(confidences: Sequence[float], correct: Sequence[int],
        eps: float = 1e-9) -> float:
    if not confidences:
        return 0.0
    total = 0.0
    for c, y in zip(confidences, correct):
        p = min(1.0 - eps, max(eps, c))
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / len(confidences)


def summarize(values: Iterable[float]) -> dict[str, float | None]:
    vals = [float(v) for v in values]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    s = sorted(vals)
    n = len(s)
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    return {"n": n, "mean": sum(vals) / n, "median": median,
            "min": s[0], "max": s[-1]}
