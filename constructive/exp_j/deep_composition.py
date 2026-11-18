"""J3 — Deep composition reliability.

Independently generated deep chains at depths 2, 3, 4, 5, 8, 12, 16. Outcomes
are classified into the six required categories. Strategies are compared on
incorrect acceptance vs correct discovery, with hard budgets that are never
silently raised.
"""

from __future__ import annotations

import random
from typing import Any, Callable

from constructive.exp_i import dsl
from constructive.exp_i.dsl import Node, call, var
from constructive.exp_i.resources import ResourceGovernor
from constructive.exp_i.search import Searcher, cegis_refine
from constructive.exp_i.types import Candidate, Example, Task, ValidationState

from . import generators as G
from . import metrics as M
from .config import ExpJConfig
from .engine import probe_pool_for, run_validation
from .resources import Monitor
from .types import Outcome

SAFE_UNARY = ("add1", "sub1", "neg", "abs", "sum_digits", "digital_root")


class FactorizedSearcher(Searcher):
    """Search-space factorization: only consider a declared operator subset."""

    def __init__(self, allowed_ops: tuple[str, ...], **kw: Any):
        super().__init__(**kw)
        self.allowed_ops = set(allowed_ops)

    def _consider(self, child, probes, seen, next_frontier, pool, target, parent_cost):
        if child.op not in self.allowed_ops and child.op != "var":
            return
        return super()._consider(child, probes, seen, next_frontier, pool,
                                 target, parent_cost)


def make_deep_task(seed: int, depth: int) -> tuple[Task, Node]:
    """Build a chain of exactly ``depth`` safe unary operations."""
    rng = random.Random(seed * 131 + depth)
    node: Node = var("int")
    for _ in range(depth):
        op = rng.choice(SAFE_UNARY)
        node = call(op, node)
    label = lambda x: dsl.evaluate(node, x)
    tx = G._uniq_ints(rng, 4, 0, 9)
    vx = G._uniq_ints(rng, 4, 10, 25, set(tx))
    ex = G._uniq_ints(rng, 12, 30, 80, set(tx + vx))
    train = [Example(x, label(x)) for x in tx]
    val = [Example(x, label(x)) for x in vx]
    test = [Example(x, label(x)) for x in ex]
    return Task(
        name=f"deep-d{depth}-s{seed}", family=f"deep-{depth}",
        train=train, validation_pool=val, test=test,
        input_type="int", output_type="int",
        meta={"hidden": node, "hidden_str": str(node), "depth": depth,
              "hidden_ops": sorted({c.op for c in _walk(node)})},
    ), node


def _walk(node: Node) -> list[Node]:
    out = [node]
    for c in node.children:
        out.extend(_walk(c))
    return out


def _equivalent(a: Node, b: Node, probes: list[Any]) -> bool:
    for x in probes:
        try:
            if dsl.evaluate(a, x) != dsl.evaluate(b, x):
                return False
        except Exception:
            return False
    return True


def _validation_fn(task: Task) -> Callable[[Candidate], Example | None]:
    def witness(cand: Candidate) -> Example | None:
        for ex in task.validation_pool:
            try:
                if dsl.evaluate(cand.program, ex.x) != ex.y:
                    return ex
            except Exception:
                return ex
        return None
    return witness


def _classify(task: Task, hidden: Node, result: Any, val_result: Any,
              found_correct: bool) -> Outcome:
    if result.timed_out:
        return Outcome.BUDGET_EXHAUSTION
    if val_result.state is ValidationState.NON_IDENTIFIABLE:
        return Outcome.NON_IDENTIFIABLE
    if val_result.accepted:
        prog = next((c.program for c in result.candidates
                     if c.id == val_result.candidate_id), None)
        if prog is not None and _equivalent(prog, hidden,
                                            [e.x for e in task.test]):
            return Outcome.CORRECT_DISCOVERED
        return Outcome.INCORRECT_ACCEPTED
    return Outcome.CORRECT_NOT_DISCOVERED


def _run_one(task: Task, hidden: Node, cfg: ExpJConfig, strategy: str,
             searcher: Searcher) -> dict[str, Any]:
    gov = ResourceGovernor(global_deadline_s=cfg.fit_deadline_s,
                           node_budget=cfg.search_node_budget)
    probes = [e.x for e in task.train + task.validation_pool + task.test]
    try:
        if strategy == "cegis":
            result, injected = cegis_refine(
                task, searcher, _validation_fn(task), rounds=1,
                node_budget=cfg.search_node_budget, deadline_s=cfg.fit_deadline_s)
        else:
            result = searcher.search(task, node_budget=cfg.search_node_budget,
                                     deadline_s=cfg.fit_deadline_s)
            injected = []
    except Exception as exc:  # noqa: BLE001 - isolation, never fabricate success
        return {"strategy": strategy, "depth": task.meta["depth"],
                "seed": int(task.name.split("-s")[-1]),
                "timed_out": False, "error": f"{type(exc).__name__}: {exc}",
                "outcome": Outcome.BUDGET_EXHAUSTION.value, "status": "error"}

    found = any(_equivalent(c.program, hidden, probes) for c in result.candidates)
    task.meta["candidates"] = result.candidates
    val_result, fields = run_validation(task, cfg, probe_pool=probe_pool_for(task))
    outcome = _classify(task, hidden, result, val_result, found)
    return {
        "strategy": strategy, "depth": task.meta["depth"],
        "seed": int(task.name.split("-s")[-1]),
        "hidden": task.meta["hidden_str"],
        "n_candidates": len(result.candidates),
        "found_correct": found,
        "accepted": val_result.accepted,
        "state": val_result.state.value,
        "outcome": outcome.value,
        "expansions": result.expansions,
        "nodes": result.nodes_generated,
        "cache_hits": result.cache_hits,
        "elapsed_s": round(result.elapsed_s, 6),
        "timed_out": bool(result.timed_out),
        "n_injected": len(injected),
        "verification_calls": len(probes) * max(1, len(result.candidates)),
        "status": "completed",
    }


STRATEGIES = ("plain", "factorized", "cegis", "cached")


def run_deep(cfg: ExpJConfig, *, seeds: tuple[int, ...] | None = None,
             strategies: tuple[str, ...] = STRATEGIES) -> dict[str, Any]:
    seeds = tuple(seeds if seeds is not None else cfg.seeds)
    mon = Monitor()
    rows: list[dict[str, Any]] = []
    for depth in cfg.depths:
        # 'cached' shares one searcher (and its memo) across seeds at a depth.
        cached_searcher = Searcher(
            beam=cfg.search_beam, max_cost=min(cfg.max_cost, depth + 1.0),
            max_depth=depth, max_nodes=64)
        for seed in seeds:
            task, hidden = make_deep_task(seed, depth)
            for strategy in strategies:
                if strategy == "plain":
                    s = Searcher(beam=cfg.search_beam,
                                 max_cost=min(cfg.max_cost, depth + 1.0),
                                 max_depth=depth, max_nodes=64)
                elif strategy == "factorized":
                    s = FactorizedSearcher(
                        allowed_ops=tuple(task.meta["hidden_ops"]),
                        beam=cfg.search_beam,
                        max_cost=min(cfg.max_cost, depth + 1.0),
                        max_depth=depth, max_nodes=64)
                elif strategy == "cached":
                    s = cached_searcher
                else:  # cegis uses a fresh searcher
                    s = Searcher(beam=cfg.search_beam,
                                 max_cost=min(cfg.max_cost, depth + 1.0),
                                 max_depth=depth, max_nodes=64)
                row = _run_one(task, hidden, cfg, strategy, s)
                rows.append(row)
                mon.charge(expansions=row.get("expansions", 0),
                           nodes=row.get("nodes", 0),
                           cache_hits=row.get("cache_hits", 0),
                           verification_calls=row.get("verification_calls", 0))
                if row.get("timed_out"):
                    mon.mark_timeout(f"depth {depth} seed {seed} {strategy}")

    by_depth_strategy: dict[str, Any] = {}
    for depth in cfg.depths:
        for strategy in strategies:
            sel = [r for r in rows if r["depth"] == depth and r["strategy"] == strategy]
            outcomes = M.taxonomy(sel, key="outcome")
            by_depth_strategy[f"d{depth}/{strategy}"] = {
                "n": len(sel),
                "outcomes": outcomes,
                "incorrect_acceptance": M.rate(r["outcome"] == Outcome.INCORRECT_ACCEPTED.value
                                               for r in sel),
                "correct_discovery": M.rate(r["outcome"] == Outcome.CORRECT_DISCOVERED.value
                                            for r in sel),
                "found_correct": M.rate(r.get("found_correct", False) for r in sel),
                "budget_exhaustion": M.rate(r.get("timed_out", False) for r in sel),
                "expansions": M.stats(r.get("expansions") for r in sel),
                "nodes": M.stats(r.get("nodes") for r in sel),
                "cache_hits": M.stats(r.get("cache_hits") for r in sel),
                "elapsed_s": M.stats(r.get("elapsed_s") for r in sel),
                "verification_calls": M.stats(r.get("verification_calls") for r in sel),
                "n_errors": sum(1 for r in sel if r.get("status") == "error"),
            }

    out = {
        "experiment_id": "J3",
        "experiment_name": "Deep Composition Reliability",
        "status": "completed",
        "seeds": list(seeds), "depths": list(cfg.depths),
        "strategies": list(strategies),
        "n_runs": len(rows),
        "episodes": rows,
        "by_depth_strategy": by_depth_strategy,
        "outcome_totals": M.taxonomy(rows, key="outcome"),
        "budgets": {
            "node_budget": cfg.search_node_budget,
            "fit_deadline_s": cfg.fit_deadline_s,
            "note": "Hard caps; never raised to force a success.",
        },
        "limitations": [
            "Safe unary chain pool keeps magnitudes bounded; not all DSL ops appear.",
            "factorized strategy is oracle-guided (uses hidden op set) and is an upper bound.",
        ],
    }
    out.update(mon.finish())
    return out
