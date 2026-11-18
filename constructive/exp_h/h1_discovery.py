"""H1 — Open-world primitive and concept discovery."""

from __future__ import annotations

import time
from typing import Any

from .episode import baseline_episode, ccm_episode
from .generators import (
    build_h1_task,
    build_list_twin_task,
    build_twin_task,
)
from .config import probe_hardware
from .metrics import aggregate, mean_ci

FAMILIES = ("vocab", "exclusion", "conditional")
K_GRID = (1, 2, 3, 5, 8, 16)


def run_h1(cfg, *, seeds=None, k_grid=K_GRID, time_budget_s: float = 4.0,
           max_nodes: int = 250_000) -> dict[str, Any]:
    seeds = tuple(seeds or cfg.seeds)
    t0 = time.perf_counter()
    episodes: list[dict[str, Any]] = []
    skipped = 0
    for family in FAMILIES:
        for k in k_grid:
            for seed in seeds:
                try:
                    task = build_h1_task(seed, k, family)
                except ValueError:
                    skipped += 1
                    continue
                episodes.append(ccm_episode(task, seed=seed, max_cost=3, beam=45,
                                            max_nodes=max_nodes,
                                            time_budget_s=time_budget_s))
                episodes.append(baseline_episode("nearest_neighbor", task, seed=seed))
    # twin (non-identifiability) episodes
    twin_rows: list[dict[str, Any]] = []
    for seed in seeds:
        for k in (3, 5, 8):
            for builder in (build_twin_task, build_list_twin_task):
                try:
                    task = builder(seed, k)
                except ValueError:
                    skipped += 1
                    continue
                twin_rows.append(ccm_episode(task, seed=seed, max_cost=3, beam=45,
                                             max_nodes=max_nodes,
                                             time_budget_s=time_budget_s))
    # --- aggregate ---
    by_family: dict[str, Any] = {}
    for family in FAMILIES:
        rows = [r for r in episodes if r["system"] == "ccm" and r["family"] == family]
        by_family[family] = {
            "n": len(rows),
            "accepted_rate": _mean(r["accepted"] for r in rows),
            "accuracy_mean": mean_ci(r["test_accuracy"] for r in rows),
            "coverage_mean": _mean(r["coverage"] for r in rows),
            "perfect_rate": _mean(1.0 if r["test_accuracy"] == 1.0 else 0.0 for r in rows),
            "failure_categories": _counts(r["failure_category"] for r in rows),
            "k_learning_curve": {str(k): mean_ci(
                r["test_accuracy"] for r in rows if r["k"] == k) for k in k_grid},
            "nodes_median": _median(r["nodes"] for r in rows),
            "fit_s_median": _median(r["fit_s"] for r in rows),
            "nodes_sum": sum(r["nodes"] for r in rows),
        }
    twin_summary = {
        "n": len(twin_rows),
        "accepted_rate": _mean(r["accepted"] for r in twin_rows),
        "abstain_rate": _mean(1.0 if r["abstained"] else 0.0 for r in twin_rows),
        "accuracy_mean": mean_ci(r["test_accuracy"] for r in twin_rows),
        "failure_categories": _counts(r["failure_category"] for r in twin_rows),
    }
    baseline_summary = {}
    for system in ("nearest_neighbor",):
        rows = [r for r in episodes if r["system"] == system]
        baseline_summary[system] = {
            "n": len(rows),
            "accuracy_mean": mean_ci(r["test_accuracy"] for r in rows if r["test_accuracy"] is not None),
        }
    return {
        "experiment_id": "H1",
        "experiment_name": "Open-World Concept Discovery",
        "status": "completed",
        "episodes": episodes,
        "twin_episodes": twin_rows,
        "skipped_degenerate": skipped,
        "summary": {
            "by_family": by_family,
            "twin_summary": twin_summary,
            "baseline_summary": baseline_summary,
            "examples_to_competence": {fam: _e2c(by_family[fam]["k_learning_curve"]) for fam in FAMILIES},
        },
        "by_family": by_family,
        "twin_summary": twin_summary,
        "baseline_summary": baseline_summary,
        "examples_to_competence": {
            fam: _e2c(by_family[fam]["k_learning_curve"]) for fam in FAMILIES},
        "hardware": probe_hardware().as_dict(),
        "wall_s": time.perf_counter() - t0,
    }


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


def _e2c(curve: dict[str, dict | None]) -> int | None:
    hits = []
    for k, stats in curve.items():
        if stats and stats.get("mean") is not None and stats["mean"] >= 0.8:
            hits.append(int(k))
    return min(hits) if hits else None
