"""J4 — Abstraction economics.

EXP-I discovered 81 abstractions and 29 reuse events yet reported identical
expansions and accuracy. This module tests whether that is a cost-model,
retrieval, useless-abstraction, benchmark or measurement problem by charging an
explicit total lifecycle cost to five arms on shared-core and unrelated streams.
"""

from __future__ import annotations

import time
from typing import Any

from constructive.exp_i import abstraction as AB
from constructive.exp_i import dsl
from constructive.exp_i.resources import ResourceGovernor
from constructive.exp_i.search import Searcher

from . import generators as G
from . import metrics as M
from .config import ExpJConfig
from .resources import Monitor
from .types import Arm

# Disclosed model coefficients. Costs are wall-seconds unless noted.
STORAGE_COST_PER_BYTE = 1e-7       # amortised storage charge per byte
COST_PER_EXPANSION = 1e-4          # charge per search expansion (from measured A)


def _keys(node: Any) -> set[str]:
    return {dsl.canonical_key(s) for s in [node] + AB.subtrees(node)}


def _reused_ids(winner: Any, store: AB.AbstractionStore) -> list[str]:
    ks = _keys(winner) if winner is not None else set()
    return [aid for aid, a in store.items.items()
            if a.active and a.validated and dsl.canonical_key(a.body) in ks]


def _agree_fraction(body: Any, task: Any) -> float:
    ok = 0
    for ex in task.train:
        try:
            if dsl.evaluate(body, ex.x) == ex.y:
                ok += 1
        except Exception:
            pass
    return ok / max(1, len(task.train))


def _bodies_for(arm: Arm, store: AB.AbstractionStore, task: Any,
                reuse_counts: dict[str, int]) -> tuple[list[Any], float]:
    t0 = time.perf_counter()
    active = [a for a in store.items.values() if a.active and a.validated]
    if arm is Arm.NONE:
        bodies: list[Any] = []
    elif arm is Arm.CURRENT:
        bodies = [a.body for a in active]
    elif arm is Arm.REUSE_CREDIT:
        bodies = [a.body for a in active if reuse_counts.get(a.id, 0) > 0]
    elif arm is Arm.RETRIEVAL_RANKED:
        bodies = [a.body for a in active if _agree_fraction(a.body, task) >= 0.5]
    else:  # selective retention
        bodies = [a.body for a in active if a.reusability >= 0.3]
        for a in active:
            if a.reusability < 0.3:
                store.retire(a.id, "below selective-retention threshold")
    return bodies, time.perf_counter() - t0


def _fit(task: Any, bodies: list[Any], cfg: ExpJConfig) -> tuple[Any, float]:
    gov = ResourceGovernor(global_deadline_s=cfg.fit_deadline_s,
                           node_budget=cfg.search_node_budget)
    s = Searcher(governor=gov, beam=cfg.search_beam, max_cost=3.5,
                 max_depth=5, max_nodes=64, abstractions=bodies)
    t0 = time.perf_counter()
    result = s.search(task)
    return result, time.perf_counter() - t0


def _accuracy(winner: Any, task: Any) -> float:
    if winner is None:
        return 0.0
    correct = 0
    for e in task.test:
        try:
            if dsl.evaluate(winner, e.x) == e.y:
                correct += 1
        except Exception:
            pass
    return correct / max(1, len(task.test))


def run_abstraction_economics(cfg: ExpJConfig, *,
                              seeds: tuple[int, ...] | None = None) -> dict[str, Any]:
    seeds = tuple(seeds if seeds is not None else cfg.seeds[:5])
    mon = Monitor()
    rows: list[dict[str, Any]] = []
    baseline: dict[tuple[int, str, int], tuple[int, float]] = {}

    # Arm A first: provides the per-task reference expansions and time.
    for seed in seeds:
        for family in ("shared-core", "unrelated"):
            for idx, task in enumerate(G.abstraction_stream_j(seed, family)):
                res, fit_s = _fit(task, [], cfg)
                baseline[(seed, family, idx)] = (res.expansions, fit_s)
                winner = res.candidates[0].program if res.candidates else None
                rows.append({"arm": Arm.NONE.value, "seed": seed, "family": family,
                             "index": idx, "expansions": res.expansions,
                             "fit_s": fit_s, "accuracy": _accuracy(winner, task),
                             "accepted": bool(res.candidates), "timed_out": res.timed_out,
                             "n_abstractions": 0, "reuse": 0})

    for arm in (Arm.CURRENT, Arm.REUSE_CREDIT, Arm.RETRIEVAL_RANKED, Arm.SELECTIVE):
        for seed in seeds:
            for family in ("shared-core", "unrelated"):
                store = AB.AbstractionStore()
                reuse_counts: dict[str, int] = {}
                prior_winners: list[Any] = []
                for idx, task in enumerate(G.abstraction_stream_j(seed, family)):
                    bodies, retrieval_s = _bodies_for(arm, store, task, reuse_counts)
                    res, fit_s = _fit(task, bodies, cfg)
                    winner = res.candidates[0].program if res.candidates else None

                    t0 = time.perf_counter()
                    harvested: list[str] = []
                    if winner is not None:
                        pool = prior_winners + [c.program for c in res.candidates[:3]]
                        harvested = store.harvest(winner, [task.name], pool)
                        for aid in harvested:
                            store.validate(aid, G.abstraction_stream_j(
                                seed + 100, family)[:3])
                        for aid in _reused_ids(winner, store):
                            reuse_counts[aid] = reuse_counts.get(aid, 0) + 1
                    discovery_s = time.perf_counter() - t0
                    if winner is not None:
                        prior_winners.append(winner)

                    base_exp, base_fit = baseline[(seed, family, idx)]
                    storage_bytes = sum(
                        len(str(a.provenance)) + len(str(a.body))
                        for a in store.items.values())
                    storage_s = storage_bytes * STORAGE_COST_PER_BYTE
                    instantiation_s = max(0.0, fit_s - base_fit)
                    savings_s = max(0.0, base_exp - res.expansions) * COST_PER_EXPANSION
                    total = discovery_s + storage_s + retrieval_s + instantiation_s
                    rows.append({
                        "arm": arm.value, "seed": seed, "family": family,
                        "index": idx, "expansions": res.expansions,
                        "fit_s": fit_s, "accuracy": _accuracy(winner, task),
                        "accepted": bool(res.candidates), "timed_out": res.timed_out,
                        "n_abstractions": len(store.items),
                        "n_harvested": len(harvested),
                        "reuse": bool(_reused_ids(winner, store)) if winner else False,
                        "discovery_s": discovery_s, "storage_s": storage_s,
                        "retrieval_s": retrieval_s, "instantiation_s": instantiation_s,
                        "savings_s": savings_s, "total_cost_s": total,
                        "net_cost_s": total - savings_s,
                        "storage_bytes": storage_bytes,
                    })
                    mon.charge(expansions=res.expansions)

    summary: dict[str, Any] = {}
    for arm in Arm:
        for family in ("shared-core", "unrelated"):
            sel = [r for r in rows if r["arm"] == arm.value and r["family"] == family]
            summary[f"{arm.value}/{family}"] = {
                "n": len(sel),
                "accuracy": M.stats(r["accuracy"] for r in sel),
                "acceptance": M.rate(r["accepted"] for r in sel),
                "expansions_total": sum(int(r["expansions"]) for r in sel),
                "discovery_s": M.stats(r.get("discovery_s", 0.0) for r in sel),
                "storage_s": M.stats(r.get("storage_s", 0.0) for r in sel),
                "retrieval_s": M.stats(r.get("retrieval_s", 0.0) for r in sel),
                "instantiation_s": M.stats(r.get("instantiation_s", 0.0) for r in sel),
                "savings_s": M.stats(r.get("savings_s", 0.0) for r in sel),
                "total_cost_s": M.stats(r.get("total_cost_s", 0.0) for r in sel),
                "net_cost_s": M.stats(r.get("net_cost_s", 0.0) for r in sel),
                "abstractions_created": sum(int(r["n_abstractions"]) for r in sel),
                "reuse_events": sum(1 for r in sel if r.get("reuse")),
                "timed_out": sum(1 for r in sel if r.get("timed_out")),
            }

    verdict = {
        "shared-core": {
            arm.value: summary[f"{arm.value}/shared-core"]["net_cost_s"]["mean"]
            for arm in Arm},
        "unrelated": {
            arm.value: summary[f"{arm.value}/unrelated"]["net_cost_s"]["mean"]
            for arm in Arm},
        "note": "Lower net cost is better. A negative value means savings exceeded cost.",
    }
    out = {
        "experiment_id": "J4", "experiment_name": "Abstraction Economics",
        "status": "completed", "seeds": list(seeds),
        "n_runs": len(rows), "episodes": rows, "summary": summary,
        "cost_model": {
            "storage_cost_per_byte": STORAGE_COST_PER_BYTE,
            "cost_per_expansion": COST_PER_EXPANSION,
            "components": ["discovery", "storage", "retrieval", "instantiation",
                           "validation (inside discovery)", "search savings"],
            "formula": "net_cost = discovery + storage + retrieval + instantiation - savings",
        },
        "arms": {a.value: a.name for a in Arm},
        "verdict": verdict,
        "limitations": [
            "Cost coefficients are disclosed and model-based, not hardware calibration.",
            "Searcher treats abstractions as atoms; reuse credit is via proven reuse filter.",
        ],
    }
    out.update(mon.finish())
    return out
