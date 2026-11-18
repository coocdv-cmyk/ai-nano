"""H8 — Computational efficiency and architectural alternatives."""

from __future__ import annotations

import time
from typing import Any

from ..core import same_value
from .episode import ccm_episode
from ..model import ConstructiveModel
from .config import probe_hardware
from .profiling import profile_region
from .generators import build_h1_task
from .lexical_h8 import alternative_search
from .metrics import mean_ci
from ..core import build_prims

_PRIMS13 = build_prims(tuple(range(0, 13)))

ARMS = {
    "ccm-beam-006": {"beam": 6},
    "ccm-beam-015": {"beam": 15},
    "ccm-beam-045": {"beam": 45},   # reference = stock CCM
    "ccm-beam-120": {"beam": 120},
    "ccm-cost-002": {"max_cost": 2},
    "ccm-cost-003": {"max_cost": 3},
    "ccm-cost-004": {"max_cost": 4},
    "beam-elite":   {"arm": "beam_elite"},
    "eager-prune":  {"arm": "eager_prune"},
}


def _run_alternative(task, *, beam_elite: bool, eager_prune: bool,
                     max_nodes: int, max_cost: int, time_budget_s: float):
    prof = profile_region(
        lambda: alternative_search(task.train, task.input_type, task.output_type,
                                   beam=45, max_nodes=max_nodes,
                                   time_budget_s=time_budget_s,
                                   beam_elite=beam_elite,
                                   eager_prune=eager_prune,
                                   max_cost=max_cost),
        count_nodes=lambda r: r[1])
    cands, nodes = prof.value
    # normalise to the same gate CCM uses: best exact candidate, val-checked
    accepted = False
    program = None
    if cands:
        exact = [c for c in cands if c.n_ok == len(task.train)]
        if exact:
            best = min(exact, key=lambda c: (c.cost, c.term.size() if hasattr(c.term, "size") else 0))
            # validation gate (same info budget as CCM)
            ok = True
            try:
                from ..core import eval_term, EvalError
                for x, y in task.val:
                    try:
                        if eval_term(best.term, {"x": x}, _PRIMS13) != y:
                            ok = False
                            break
                    except EvalError:
                        ok = False
                        break
            except Exception:
                ok = False
            if ok:
                accepted = True
                program = str(best.term)
    correct = 0
    answered = 1 if accepted else 0
    if accepted:
        from ..core import eval_term, EvalError
        for x, y in task.test:
            try:
                v = eval_term(best.term, {"x": x}, _PRIMS13)
            except EvalError:
                continue
            if same_value(v, y):
                correct += 1
    return {
        "system": "alt",
        "accepted": accepted,
        "program": program,
        "fit_s": prof.wall_s,
        "nodes": nodes,
        "test_accuracy": correct / len(task.test) if task.test else 0.0,
        "coverage": answered / len(task.test) if task.test else 0.0,
        "failure_category": ("success" if accepted else "search-explosion"),
        "beam_elite": beam_elite,
        "eager_prune": eager_prune,
    }


def run_h8(cfg, *, seeds=None, k_grid=(2, 3, 5, 8), family="vocab",
           time_budget_s: float = 3.0, max_nodes: int = 150_000) -> dict[str, Any]:
    seeds = tuple(seeds or cfg.seeds[:3])
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for arm, spec in ARMS.items():
        for k in k_grid:
            for seed in seeds:
                try:
                    task = build_h1_task(seed, k, family)
                except ValueError:
                    continue
                if "arm" in spec:
                    row = _run_alternative(task, beam_elite=spec["arm"] == "beam_elite",
                                           eager_prune=spec["arm"] == "eager_prune",
                                           max_nodes=max_nodes,
                                           max_cost=min(3, k + 1),
                                           time_budget_s=time_budget_s)
                    row.update({"arm": arm, "k": k, "seed": seed, "family": family})
                else:
                    kw = {"max_nodes": max_nodes, "time_budget_s": time_budget_s}
                    if "beam" in spec:
                        kw["beam"] = spec["beam"]
                    if "max_cost" in spec:
                        kw["max_cost"] = spec["max_cost"]
                    row = ccm_episode(task, seed=seed, **kw)
                    row.update({"arm": arm, "k": k, "seed": seed, "family": family})
                rows.append(row)

    summary: dict[str, Any] = {}
    for arm in sorted({r["arm"] for r in rows}):
        sel = [r for r in rows if r["arm"] == arm]
        summary[arm] = {
            "n": len(sel),
            "accepted_rate": _mean(r["accepted"] for r in sel),
            "accuracy_mean": mean_ci(r["test_accuracy"] for r in sel),
            "fit_s_median": _median(r["fit_s"] for r in sel),
            "nodes_median": _median(r["nodes"] for r in sel),
            "failure_categories": _counts(r["failure_category"] for r in sel),
        }
    return {
        "experiment_id": "H8",
        "experiment_name": "Computational Efficiency and Architectural Alternatives",
        "status": "completed",
        "arms": ARMS,
        "episodes": rows,
        "summary": summary,
        "hardware": probe_hardware().as_dict(),
        "wall_s": time.perf_counter() - t0,
    }


def _mean(xs) -> float | None:
    xs = list(xs)
    return round(sum(xs) / len(xs), 6) if xs else None


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    mid = n // 2
    return xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2


def _counts(items) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out
