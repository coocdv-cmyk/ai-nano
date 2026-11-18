"""H3 — Structural OOD and compositional generalization."""

from __future__ import annotations

import time
from typing import Any

from .episode import ccm_episode
from .generators import (
    build_ood_list_task,
    build_ood_numeric_task,
    build_ood_scene_task,
)
from .config import probe_hardware
from .metrics import mean_ci

LENGTH_RULES = ("sum_of_evens", "distinct_count", "sum_top3")
NUMERIC_RULES = ("square", "sum_digits", "fib")
SCENE_RULES = ("nearest", "same_shape", "left_of")


def run_h3(cfg, *, seeds=None, n_test: int = 20, time_budget_s: float = 4.0,
           max_nodes: int = 250_000) -> dict[str, Any]:
    seeds = tuple(seeds or cfg.seeds)
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for rule in LENGTH_RULES:
        for seed in seeds:
            task = build_ood_list_task(rule, seed, k=4, train_len=(2, 8),
                                       test_len=(17, 60), n_test=min(n_test, 12))
            rows.append(ccm_episode(task, seed=seed,
                                    max_nodes=max_nodes,
                                    time_budget_s=time_budget_s) | {"split_kind": "length-ood"})
    for rule in NUMERIC_RULES:
        for seed in seeds:
            task = build_ood_numeric_task(rule, seed, k=3, lo=0, hi=9,
                                          test_hi=70, n_test=n_test)
            rows.append(ccm_episode(task, seed=seed,
                                    max_nodes=max_nodes,
                                    time_budget_s=time_budget_s) | {"split_kind": "numeric-ood"})
    for rule in SCENE_RULES:
        for seed in seeds:
            task = build_ood_scene_task(rule, seed, k=4, train_size=(4, 6),
                                        test_size=(7, 12), n_test=n_test)
            rows.append(ccm_episode(task, seed=seed,
                                    max_nodes=max_nodes,
                                    time_budget_s=time_budget_s) | {"split_kind": "structure-ood"})

    summary: dict[str, Any] = {}
    for split_kind in ("length-ood", "numeric-ood", "structure-ood"):
        sel = [r for r in rows if r["split_kind"] == split_kind]
        summary[split_kind] = {
            "n": len(sel),
            "accuracy_mean": mean_ci(r["test_accuracy"] for r in sel),
            "accepted_rate": _mean(r["accepted"] for r in sel),
            "coverage_mean": _mean(r["coverage"] for r in sel),
            "nodes_median": _median(r["nodes"] for r in sel),
            "failure_categories": _counts(r["failure_category"] for r in sel),
            "by_rule": {
                rule: {
                    "n": len(sel2),
                    "accuracy_mean": mean_ci(r["test_accuracy"] for r in sel2),
                    "accepted_rate": _mean(r["accepted"] for r in sel2),
                }
                for rule in sorted({_rule_of(r) for r in sel})
                for sel2 in [[r for r in sel if _rule_of(r) == rule]]
            },
        }
    return {
        "experiment_id": "H3",
        "experiment_name": "Structural OOD and Compositional Generalization",
        "status": "completed",
        "episodes": rows,
        "summary": summary,
        "hardware": probe_hardware().as_dict(),
        "wall_s": time.perf_counter() - t0,
    }


def _rule_of(row: dict[str, Any]) -> str:
    return row["task"].split("-")[1]


def _mean(xs):
    xs = list(xs)
    return round(sum(xs) / len(xs), 6) if xs else None


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    mid = n // 2
    return xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2


def _counts(items):
    out: dict[str, int] = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out
