"""H4 — Large-scale continual learning and interference."""

from __future__ import annotations

import statistics
import time
from typing import Any

from ..core import same_value
from .episode import ccm_episode
from ..model import ConstructiveModel
from ..worlds import Task
from .config import probe_hardware
from .profiling import profile_region
from .h4_pool import build_pool
from .metrics import mean_ci

# The pool defines what exists inside this vocabulary; the 250-concept tier
# is explicitly NOT executed (reported as not run, not as zero interference).
N_EXECUTE = 10**9  # i.e. "as many as the vocabulary can express"


def _accuracy_on(model: ConstructiveModel, task: Task) -> float:
    correct = 0
    for x, y in task.test:
        v = model.predict_episode(task.train, x, task.input_type)
        if v is not None and same_value(v, y):
            correct += 1
    return correct / len(task.test) if task.test else 0.0


def run_h4(cfg, *, seeds=None, n_concepts: int = N_EXECUTE,
           time_budget_s: float = 2.5, max_nodes: int = 150_000) -> dict[str, Any]:
    seeds = tuple(seeds or cfg.seeds[:2])
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        pool = build_pool()
        n_avail = len(pool)
        n_use = min(n_concepts, n_avail)
        pool = pool[:n_use]
        model = ConstructiveModel(max_cost=3, beam=45, max_nodes=max_nodes,
                                  time_budget_s=time_budget_s)
        learned: list[tuple[str, Task, int]] = []  # (pool_item, task, step_id)
        acc_after_learn: dict[int, float] = {}
        for i, (item, make) in enumerate(pool):
            try:
                task = make(seed)
            except ValueError as exc:
                rows.append({"seed": seed, "step": i, "pool_item": item,
                             "failure_category": "generation-error",
                             "error": str(exc), "accepted": False,
                             "task": "(generation failed)"})
                continue
            before_params = model.n_parameters()
            before_nodes = model.n_stored_nodes()
            prof = profile_region(
                lambda: model.fit(task.train, task.input_type, task.output_type,
                                  validation=task.val, probes=task.probes),
                count_nodes=lambda r: r.nodes)
            rep = prof.value
            accepted = rep.accepted
            acc_own = _accuracy_on(model, task)
            if accepted:
                learned.append((item, task, i))
                acc_after_learn[i] = acc_own
            row = {
                "seed": seed,
                "step": i,
                "pool_item": item,
                "task": task.name,
                "accepted": accepted,
                "own_accuracy_after_learn": acc_own,
                "failure_category": "success" if accepted else classify_own(model, rep),
                "fit_s": prof.wall_s,
                "nodes": rep.nodes,
                "peak_bytes": prof.traced_peak_bytes,
                "new_nodes": model.n_stored_nodes() - before_nodes,
                "new_params": model.n_parameters() - before_params,
                "n_concepts": len(model.concepts),
                "n_guard_nodes": _count_guard_nodes(model),
                "false_acceptance": accepted and acc_own < 0.8,
                "quarantined": getattr(rep, "quarantined", []),
            }
            if i + 1 in (9, 25, 50, 100):
                # accuracy of every concept learned so far at this checkpoint
                row["checkpoint"] = {
                    str(step_j): _accuracy_on(model, task_j)
                    for j, (item_j, task_j, step_j) in enumerate(learned)
                }
            rows.append(row)
        # final per-concept evaluation
        final_accs: dict[str, float] = {}
        for item_j, task_j, step_j in learned:
            final_accs[item_j] = _accuracy_on(model, task_j)

        rows.append({
            "seed": seed, "experiment": "h4-final",
            "n_learned": len(learned),
            "n_pool": len(pool),
            "n_final_concepts": len(model.concepts),
            "n_stored_nodes": model.n_stored_nodes(),
            "n_params": model.n_parameters(),
            "final_accuracies": final_accs,
            "lifetime_fit_s": sum(r.get("fit_s", 0.0) for r in rows
                                  if r.get("seed") == seed and "fit_s" in r),
            "lifetime_search_nodes": sum(r.get("nodes", 0) for r in rows
                                         if r.get("seed") == seed and "nodes" in r),
            "guard_count": _count_guards(model),
        })
        # forgetting per concept per seed
        for item_j, task_j, step_j in learned:
            drop = acc_after_learn.get(step_j, 0.0) - final_accs.get(item_j, 0.0)
            rows.append({"seed": seed, "experiment": "forgetting",
                         "pool_item": item_j, "step": step_j,
                         "after_learn": acc_after_learn.get(step_j),
                         "final": final_accs.get(item_j),
                         "drop": drop,
                         "interference": drop > 0.0})

    steps = [r for r in rows if r.get("experiment") is None and "step" in r]
    summary: dict[str, Any] = {}
    for seed in seeds:
        srows = [r for r in steps if r["seed"] == seed]
        learns = [r for r in srows if r.get("accepted")]
        summary[f"seed_{seed}"] = {
            "steps": len(srows),
            "accepted_rate": _mean(bool(r["accepted"]) for r in srows),
            "own_acc_mean": mean_ci(r["own_accuracy_after_learn"] for r in learns),
            "lifetime_search_nodes": sum(r.get("nodes", 0) for r in srows),
            "lifetime_fit_s": sum(r.get("fit_s", 0.0) for r in srows),
            "n_guards": _sum(srows, "n_guard_nodes"),
            "false_acceptances": sum(1 for r in srows if r.get("false_acceptance")),
        }
    all_forget = [r for r in rows if r.get("experiment") == "forgetting"]
    summary["forgetting"] = {
        "n": len(all_forget),
        "mean_drop": mean_ci(r["drop"] for r in all_forget),
        "worst": max((r["drop"] for r in all_forget), default=None),
        "interference_count": sum(1 for r in all_forget if r["interference"]),
    }
    return {
        "experiment_id": "H4",
        "experiment_name": "Large-Scale Continual Learning and Interference",
        "status": "completed",
        "requested_corpus_size": 250,
        "executed_corpus_size": max((r.get("n_pool", 0) for r in rows
                                     if r.get("experiment") == "h4-final"), default=0),
        "pool_available_note": ("N is bounded by concepts this vocabulary can express; "
                                "the full 250-tier requires more parametrized concepts "
                                "than the frozen DSL generates. Reported as not run, "
                                "never as zero-interference."),
        "episodes": [r for r in rows if r.get("experiment") is None],
        "forgetting_rows": [r for r in rows if r.get("experiment") == "forgetting"],
        "final_evals": [r for r in rows if r.get("experiment") == "h4-final"],
        "summary": summary,
        "seed_summary": summary,
        "hardware": probe_hardware().as_dict(),
        "wall_s": time.perf_counter() - t0,
    }


def classify_own(model, rep):
    if rep.accepted:
        return "success"
    if getattr(rep, "timed_out", False):
        return "time-exhaustion"
    if "ambiguous" in (rep.reason or ""):
        return "non-identifiable"
    if "fails held-out validation" in (rep.reason or ""):
        return "validation-failure"
    return "search-explosion"


def _count_guard_nodes(model) -> int:
    return sum(1 for c in model.concepts.values() if c.guard is not None)


def _count_guards(model):
    return len({str(c.guard) for c in model.concepts.values() if c.guard is not None})


def _sum(rows, key):
    return sum(int(r.get(key, 0)) for r in rows)


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
