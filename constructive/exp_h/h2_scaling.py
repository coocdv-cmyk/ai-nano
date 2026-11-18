"""H2 — Deep composition / reasoning scaling."""

from __future__ import annotations

import time
from typing import Any

from .episode import ccm_episode
from .generators import build_h2_task
from .config import probe_hardware
from .metrics import mean_ci

DEPTHS = (2, 3, 5, 8, 12, 16, 20, 32)


def run_h2(cfg, *, seeds=None, depths=DEPTHS, time_budget_s: float = 3.0,
           max_nodes: int = 200_000) -> dict[str, Any]:
    seeds = tuple(seeds or cfg.seeds[:3])
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    unary_depths = tuple(d for d in depths)
    # pair trees are exponentially sized in depth; depth > 12 exceeds the
    # engine's safety caps, so deeper pair trees are reported as not running
    pair_depths = tuple(d for d in depths if d <= 12)
    for depth in unary_depths:
        for seed in seeds:
            try:
                task = build_h2_task(seed, depth, "unary_chain", k=4)
            except ValueError:
                continue
            row = ccm_episode(task, seed=seed, max_cost=min(depth + 1, 4), beam=45,
                              max_nodes=max_nodes, time_budget_s=time_budget_s)
            row["depth"] = depth
            rows.append(row)
    for depth in pair_depths:
        for seed in seeds:
            try:
                task = build_h2_task(seed, depth, "pair_tree", k=4)
            except ValueError:
                continue
            row = ccm_episode(task, seed=seed, max_cost=min(depth + 1, 4), beam=45,
                              max_nodes=max_nodes, time_budget_s=time_budget_s)
            row["depth"] = depth
            rows.append(row)
    # one-seed conditional depth sweep (negative control, unreachable by design)
    for depth in (2, 4, 6):
        try:
            task = build_h2_task(0, depth, "conditional_tree", k=4)
        except ValueError:
            continue
        row = ccm_episode(task, seed=0, max_cost=min(depth + 1, 4), beam=45,
                          max_nodes=max_nodes, time_budget_s=time_budget_s)
        row["depth"] = depth
        rows.append(row)

    summary: dict[str, Any] = {}
    for family in ("h2-unary_chain", "h2-pair_tree", "h2-conditional_tree"):
        fam = [r for r in rows if r["family"] == family]
        if not fam:
            continue
        by_depth = {}
        for depth in sorted({r["depth"] for r in fam}):
            sel = [r for r in fam if r["depth"] == depth]
            acc = [r["test_accuracy"] for r in sel]
            by_depth[str(depth)] = {
                "n": len(sel),
                "accuracy_mean": mean_ci(acc),
                "accepted_rate": _mean(r["accepted"] for r in sel),
                "nodes_median": _median(r["nodes"] for r in sel),
                "nodes_sum": sum(r["nodes"] for r in sel),
                "fit_s_median": _median(r["fit_s"] for r in sel),
                "timed_out_rate": _mean(r.get("failure_category") == "time-exhaustion" for r in sel),
                "spurious_accept_rate": _mean(
                    1.0 if (r["accepted"] and r["test_accuracy"] < 0.8) else 0.0 for r in sel),
                "failure_categories": _counts(r["failure_category"] for r in sel),
            }
        summary[family.replace("h2-", "")] = {"by_depth": by_depth}
    return {
        "experiment_id": "H2",
        "experiment_name": "Deep Composition and Reasoning Scaling",
        "status": "completed",
        "episodes": rows,
        "summary": summary,
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
