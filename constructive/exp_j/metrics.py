"""EXP-J standardised metrics.

Every principal comparison reports n, mean, median, std, 95% CI, baseline,
new value and failure counts. Missing measurements are never fabricated.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable

from constructive.exp_i.validation import Calibrator

CI_Z = 1.959963985


def stats(values: Iterable[float | None]) -> dict[str, Any]:
    xs = [float(v) for v in values if v is not None]
    n = len(xs)
    if not n:
        return {"n": 0, "mean": None, "median": None, "std": None,
                "ci95": None, "min": None, "max": None}
    sd = statistics.pstdev(xs) if n > 1 else 0.0
    return {
        "n": n,
        "mean": round(statistics.fmean(xs), 6),
        "median": round(statistics.median(xs), 6),
        "std": round(sd, 6),
        "ci95": round(CI_Z * sd / math.sqrt(n), 6),
        "min": round(min(xs), 6),
        "max": round(max(xs), 6),
    }


def rate(flags: Iterable[bool]) -> dict[str, Any]:
    return stats([1.0 if bool(f) else 0.0 for f in flags])


def accuracy_coverage(correct: int, answered: int, total: int) -> dict[str, Any]:
    return {
        "total": total, "answered": answered, "correct": correct,
        "accuracy": round(correct / total, 6) if total else None,
        "coverage": round(answered / total, 6) if total else None,
        "selective_accuracy": round(correct / answered, 6) if answered else None,
    }


def calibration_ece(confidences: list[float], correct: list[int], *,
                    n_bins: int = 10, fit_fraction: float = 0.5) -> dict[str, Any]:
    """Fit calibrator on the first split, report ECE on the disjoint second split."""
    pairs = list(zip(confidences, correct))
    cut = max(1, int(len(pairs) * fit_fraction))
    fit, eval_ = pairs[:cut], pairs[cut:]
    cal = Calibrator(n_bins=n_bins)
    cal.fit([c for c, _ in fit], [y for _, y in fit])
    ece = cal.ece([c for c, _ in eval_], [y for _, y in eval_]) if eval_ else None
    bins = []
    for b in range(n_bins):
        if cal.bin_count[b]:
            bins.append({"bin": b, "n": cal.bin_count[b],
                         "empirical_accuracy": round(cal.bin_correct[b] / cal.bin_count[b], 4)})
    return {
        "ece_independent": round(ece, 6) if ece is not None else None,
        "n_fit": len(fit), "n_eval": len(eval_), "bins": bins,
        "note": "Calibration split never used for candidate selection or test measurement.",
    }


def retrieval_pr(relevant: set[str], ranked: list[str], k: int) -> dict[str, Any]:
    top = ranked[:k]
    hits = sum(1 for cid in top if cid in relevant)
    return {
        "k": k, "hits": hits,
        "precision": round(hits / len(top), 6) if top else None,
        "recall": round(hits / len(relevant), 6) if relevant else None,
        "silent_substitution": bool(top and hits == 0),
    }


def confusion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the required evidence-engine outcome counts."""
    out = {
        "n": len(rows), "correct_acceptance": 0, "wrong_acceptance": 0,
        "abstention": 0, "false_rejection": 0, "non_identifiable": 0,
        "contradictory": 0, "insufficient": 0,
    }
    for r in rows:
        out["correct_acceptance"] += int(bool(r.get("correct_acceptance")))
        out["wrong_acceptance"] += int(bool(r.get("wrong_acceptance")))
        out["abstention"] += int(bool(r.get("abstained")))
        out["false_rejection"] += int(bool(r.get("false_rejection")))
        out["non_identifiable"] += int(bool(r.get("non_identifiable")))
        out["contradictory"] += int(bool(r.get("contradictory")))
        out["insufficient"] += int(bool(r.get("insufficient")))
    return out


def compare(baseline: Iterable[float], new: Iterable[float], *,
            lower_is_better: bool = False) -> dict[str, Any]:
    b, n = stats(baseline), stats(new)
    out = {"baseline": b, "new": n, "lower_is_better": lower_is_better}
    if b["mean"] is not None and n["mean"] is not None:
        diff = n["mean"] - b["mean"]
        out["absolute_improvement"] = round(diff, 6)
        out["relative_improvement"] = (round(diff / abs(b["mean"]), 6)
                                       if b["mean"] else None)
        out["improved"] = diff < 0 if lower_is_better else diff > 0
    else:
        out.update(absolute_improvement=None, relative_improvement=None, improved=None)
    return out


def detection_delay(flags: list[bool], event_index: int) -> int | None:
    """Episodes after ``event_index`` until the first True detection flag."""
    for i in range(event_index, len(flags)):
        if flags[i]:
            return i - event_index
    return None


def taxonomy(rows: list[dict[str, Any]], key: str = "failure_category") -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        k = str(r.get(key, r.get("outcome", "unknown")))
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))
