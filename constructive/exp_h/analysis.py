"""Cross-experiment analysis: curves, failure analysis, baselines table."""

from __future__ import annotations

from typing import Any

from .metrics import aggregate, mean_ci

THRESHOLD = 0.8


def _experiments_of(latest: dict[str, Any]) -> dict[str, Any]:
    if isinstance(latest, dict) and "experiments" in latest and isinstance(latest["experiments"], dict):
        return latest["experiments"]
    return latest


def experiment_rows(latest: dict[str, Any], exp_id: str) -> list[dict[str, Any]]:
    return (_experiments_of(latest).get(exp_id) or {}).get("episodes", [])


def learning_curve_by_k(latest: dict[str, Any], exp_id: str,
                        family: str | None = None) -> dict[str, Any]:
    rows = experiment_rows(latest, exp_id)
    buckets: dict[int, list[float]] = {}
    for r in rows:
        if family and r.get("family") != family:
            continue
        k = r.get("k")
        if k is None:
            continue
        buckets.setdefault(k, []).append(r.get("test_accuracy", 0.0) or 0.0)
    return {
        str(k): {"n": len(v), **aggregate(v)}
        for k, v in sorted(buckets.items())
    }


def learning_curve_by_depth(latest: dict[str, Any], exp_id: str,
                            family: str | None = None) -> dict[str, Any]:
    rows = experiment_rows(latest, exp_id)
    buckets: dict[int, list[float]] = {}
    for r in rows:
        if family and r.get("family") != family:
            continue
        d = r.get("depth")
        if d is None:
            continue
        buckets.setdefault(d, []).append(r.get("test_accuracy", 0.0) or 0.0)
    return {
        str(d): {"n": len(v), **aggregate(v)}
        for d, v in sorted(buckets.items())
    }


def failure_taxonomy(latest: dict[str, Any]) -> dict[str, dict[str, int]]:
    latest = _experiments_of(latest)
    out: dict[str, dict[str, int]] = {}
    for exp_id in ("H1", "H2", "H3", "H4", "H5", "H7", "H8"):
        rows = experiment_rows(latest, exp_id)
        counts: dict[str, int] = {}
        for r in rows:
            cat = r.get("failure_category") or "unknown"
            counts[cat] = counts.get(cat, 0) + 1
        if counts:
            out[exp_id] = counts
    return out


def examples_to_competence(latest: dict[str, Any], exp_id: str,
                           family: str) -> dict[str, Any]:
    curve = learning_curve_by_k(latest, exp_id, family)
    hits = [int(k) for k, v in curve.items()
            if v.get("mean") is not None and v["mean"] >= THRESHOLD]
    return {"curve": curve, "examples_to_competence": min(hits) if hits else None}


def summary_table(latest: dict[str, Any]) -> dict[str, Any]:
    latest = _experiments_of(latest)
    out: dict[str, Any] = {}
    for exp_id in ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"):
        block = latest.get(exp_id) or {}
        episodes = block.get("episodes", [])
        acc = [r["test_accuracy"] for r in episodes
               if r.get("test_accuracy") is not None]
        out[exp_id] = {
            "n_episodes": len(episodes),
            "accuracy_mean": (sum(acc) / len(acc)) if acc else None,
            "wall_s": block.get("wall_s"),
            "summary_keys": list((block.get("summary") or {}).keys()),
        }
    return out
