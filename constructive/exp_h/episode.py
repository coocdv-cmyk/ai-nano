"""Shared episode runner: fit a system on a task, measure everything."""

from __future__ import annotations

import time
from typing import Any

from ..baselines import MLP, Memorizer, NearestNeighbor
from ..core import same_value
from ..model import ConstructiveModel
from ..worlds import Task
from .profiling import profile_region

# failure-label hints from families (used only as corroborating signal)
_VOCAB_GAP_FAMILIES = {"exclusion", "conditional"}


def classify_failure(rep: Any, task: Task) -> str:
    if rep.accepted:
        return "success"
    timed_out = bool(getattr(rep, "timed_out", False))
    reason = getattr(rep, "reason", "") or ""
    candidates = getattr(rep, "candidates", [])
    family = task.meta.get("family_tag") or task.family or ""
    if reason == "fits all examples but fails held-out validation":
        return "validation-failure"
    if timed_out:
        return "time-exhaustion"
    if reason == "no program reproduces all examples":
        if family in _VOCAB_GAP_FAMILIES and not candidates:
            return "vocabulary-limitation"
        return "search-explosion"
    if "ambiguous" in reason or "hypothesis" in reason:
        return "non-identifiable"
    if reason == "" or reason == "one inconsistent example quarantined":
        return "success"
    if reason:
        if family in _VOCAB_GAP_FAMILIES and not candidates:
            return "representation-limitation"
        return "validation-failure"
    return "incorrect-candidate"


def ccm_episode(task: Task, *, seed: int, max_cost: int = 3, beam: int = 45,
                max_nodes: int = 250_000, time_budget_s: float = 4.0) -> dict[str, Any]:
    model = ConstructiveModel(max_cost=max_cost, beam=beam, max_nodes=max_nodes,
                              time_budget_s=time_budget_s)
    before_params = model.n_parameters()
    before_nodes = model.n_stored_nodes()
    try:
        prof = profile_region(lambda: model.fit(task.train, task.input_type,
                                                task.output_type,
                                                validation=task.val,
                                                probes=task.probes),
                              count_nodes=lambda r: r.nodes)
    except Exception as exc:
        return {
            "system": "ccm",
            "task": task.name,
            "family": task.meta.get("family_tag", task.family),
            "k": len(task.train),
            "seed": seed,
            "accepted": False,
            "abstained": True,
            "failure_category": "execution-error",
            "error": f"{type(exc).__name__}: {exc}",
            "fit_s": None,
            "nodes": None,
            "peak_bytes": None,
            "test_n": len(task.test),
            "test_answered": 0,
            "test_correct": 0,
            "test_accuracy": 0.0,
            "coverage": 0.0,
        }
    rep = prof.value
    correct = answered = 0
    for x, y in task.test:
        v = model.predict_episode(task.train, x, task.input_type)
        if v is None:
            continue
        answered += 1
        if same_value(v, y):
            correct += 1
    n = len(task.test)
    category = classify_failure(rep, task)
    if category == "success" and answered > 0:
        if correct / n < 0.5:
            category = "overgeneralization"
        elif correct / n < 0.8:
            category = "undergeneralization"
    return {
        "system": "ccm",
        "task": task.name,
        "family": task.meta.get("family_tag", task.family),
        "k": len(task.train),
        "seed": seed,
        "accepted": rep.accepted,
        "abstained": rep.abstained,
        "reason": rep.reason,
        "quarantined": rep.quarantined if hasattr(rep, "quarantined") else [],
        "failure_category": category,
        "fit_s": prof.wall_s,
        "nodes": rep.nodes,
        "peak_bytes": prof.traced_peak_bytes,
        "n_concepts": len(model.concepts) + len(getattr(model, "hypotheses", {})),
        "new_nodes": model.n_stored_nodes() - before_nodes,
        "new_params": model.n_parameters() - before_params,
        "test_n": n,
        "test_answered": answered,
        "test_correct": correct,
        "test_accuracy": correct / n if n else 0.0,
        "coverage": answered / n if n else 0.0,
        "confidence": rep.confidence,
        "margin": rep.margin,
        "candidates": rep.candidates[:3],
    }


def baseline_episode(name: str, task: Task, *, seed: int, **kw: Any) -> dict[str, Any]:
    try:
        data = list(task.train) + list(task.val)
        if name == "memorizer":
            b: Any = Memorizer()
        elif name == "nearest_neighbor":
            b = NearestNeighbor()
        elif name == "mlp":
            b = MLP()
        else:
            raise ValueError(name)
        prof = profile_region(lambda: b.fit(data, task.input_type, task.output_type,
                                            validation=task.val))
    except Exception as exc:  # NotSupported and others are reported, never hidden
        return {"system": name, "task": task.name,
                "family": task.meta.get("family_tag", task.family),
                "k": len(task.train), "seed": seed,
                "unsupported": True, "error": str(exc),
                "failure_category": "representation-limitation",
                "test_accuracy": None, "coverage": None}
    correct = answered = 0
    for x, y in task.test:
        try:
            v = b.predict(x)
        except Exception:
            v = None
        if v is None:
            continue
        answered += 1
        if same_value(v, y):
            correct += 1
    n = len(task.test)
    return {"system": name, "task": task.name,
            "family": task.meta.get("family_tag", task.family),
            "k": len(task.train), "seed": seed,
            "unsupported": False,
            "fit_s": prof.wall_s, "peak_bytes": prof.traced_peak_bytes,
            "failure_category": "success" if correct else ("undergeneralization" if answered else "non-identifiable"),
            "test_n": n, "test_answered": answered, "test_correct": correct,
            "test_accuracy": correct / n if n else 0.0,
            "coverage": answered / n if n else 0.0}


def one_line(row: dict[str, Any]) -> dict[str, Any]:
    """Slim dict for logs/prints; keeps every evidence field."""
    return {k: v for k, v in row.items() if k not in ("candidates",)}
