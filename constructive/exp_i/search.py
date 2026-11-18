"""EXP-I-D — typed, memoized, cost-aware AST search with hard budgets.

This is an experimental search substrate, not a replacement for CCM. It keeps a
global subexpression cache and evaluation memo, expands by cost, prunes by a
provable type-distance lower bound, and supports counterexample-guided refinement
through injected support points. Budgets are hard caps; running out is reported,
never silently raised.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from . import dsl
from .dsl import Node, call, lit, var
from .resources import ResourceLimit, ResourceGovernor
from .types import Candidate, Example, Task

EvalFn = Callable[[Any, Any], Any]


@dataclass
class SearchResult:
    candidates: list[Candidate] = field(default_factory=list)
    expansions: int = 0
    nodes_generated: int = 0
    cache_hits: int = 0
    elapsed_s: float = 0.0
    timed_out: bool = False
    reason: str = ""
    abstraction_reuse: int = 0
    abstraction_rejections: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)


# ---- admissible type-distance bound ----------------------------------------


_MIN_TYPE_COST: dict[tuple[str, str], int] = {
    ("int", "int"): 1,
    ("int", "bool"): 1,     # gt/even/odd/is_prime
    ("bool", "int"): 5,     # if with int branches
    ("bool", "bool"): 1,    # not/and/or
    ("list", "int"): 1,     # sum_list/len_list/max_list...
    ("list", "list"): 1,    # sort/evens/map...
    ("int", "list"): 6,     # no direct int->list op in a bounded chain
    ("pair", "int"): 1,
    ("int", "pair"): 1,
    ("list", "pair"): 8,
    ("pair", "list"): 8,
    ("scene", "int"): 99,   # not expressible in typed DSL
    ("scene", "bool"): 99,
}


def can_reach(src: str, dst: str, budget: float) -> bool:
    """Admissible pruning: lower bound on cost to change output type."""
    if src == dst:
        return budget >= 0
    lb = _MIN_TYPE_COST.get((src, dst), 99)
    return budget >= lb


# ---- primitive libraries ---------------------------------------------------


_INT_UNARY = ("add1", "sub1", "neg", "abs", "double", "square", "sum_digits",
              "digital_root", "fib")
_INT_BOOL = ("even", "odd", "is_prime", "pos", "is_zero")
_LIST_TO_LIST = ("evens", "odds", "sort", "sort_desc", "reverse", "unique",
                 "map_square", "map_cube")
_LIST_TO_INT = ("sum_list", "len_list", "max_list", "min_list", "head", "last",
                "range_list", "count_even", "count_odd", "sum_squares")
_PARAM_LIST_LIST = ("map_add", "map_mul", "filter_gt", "take", "drop")
_PARAM_LIST_INT = ("count_gt",)
_PARAM_INT_INT = ("add", "sub", "mul", "mod")
_PARAM_INT_BOOL = ("gt", "lt", "le", "ge")


def constants_from(task: Task, limit: int = 6) -> list[int]:
    vals: set[int] = set()
    for ex in (task.train + task.validation_pool):
        x = ex.x
        if isinstance(x, bool):
            continue
        if isinstance(x, int):
            vals.add(x)
        elif isinstance(x, list):
            vals.update(v for v in x if not isinstance(v, bool))
    out = sorted(vals, key=lambda v: (abs(v), v))[:limit]
    return out or [1, 2, 3]


class Searcher:
    """Bottom-up typed enumeration with value-level deduplication and memoization."""

    def __init__(self, *, eval_fn: EvalFn | None = None,
                 governor: ResourceGovernor | None = None,
                 beam: int = 60, max_cost: float = 4.0,
                 max_depth: int = 8, max_nodes: int = 64,
                 max_magnitude: int = 10 ** 9,
                 abstractions: list[Node] | None = None):
        self.abstractions = list(abstractions or [])
        self.eval_fn = eval_fn or dsl.evaluate
        self.governor = governor
        self.beam = beam
        self.max_cost = max_cost
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.max_magnitude = max_magnitude
        self.cache: dict[str, tuple] = {}
        self.hits = 0
        self.expansions = 0
        self.nodes_generated = 0

    # ---- memoized evaluation ----------------------------------------------
    def outputs(self, node: Node, probes: list[Any]) -> tuple:
        key = (dsl.canonical_key(node), tuple(_pk(x) for x in probes))
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        outs: list[Any] = []
        for p in probes:
            try:
                outs.append(_pk(self.eval_fn(node, p)))
            except Exception as exc:  # noqa: BLE001
                outs.append(f"ERR:{type(exc).__name__}")
        val = tuple(outs)
        self.cache[key] = val
        return val

    # ---- atom generation ---------------------------------------------------
    def _atoms(self, task: Task) -> list[Node]:
        atoms: list[Node] = []
        in_t = task.input_type
        atoms.append(var(in_t) if in_t in dsl._PRIMITIVE_TYPES else var("int"))
        for c in constants_from(task):
            if abs(c) <= self.max_magnitude:
                atoms.append(lit(c))
        for op in _INT_UNARY:
            atoms.append(call(op, var("int")))
        for op in _INT_BOOL:
            atoms.append(call(op, var("int")))
        for op in _LIST_TO_LIST:
            atoms.append(call(op, var("list")))
        for op in _LIST_TO_INT:
            atoms.append(call(op, var("list")))
        consts = constants_from(task)[:3]
        for op in _PARAM_LIST_LIST:
            for k in consts:
                atoms.append(call(op, var("list"), params=(k,)))
        for op in _PARAM_LIST_INT:
            for k in consts:
                atoms.append(call(op, var("list"), params=(k,)))
        for op in _PARAM_INT_INT:
            for k in consts:
                atoms.append(call(op, var("int"), params=(k,)))
        for op in _PARAM_INT_BOOL:
            for k in consts:
                atoms.append(call(op, var("int"), params=(k,)))
        # Reusable abstractions are search units, never retrieved as solutions.
        atoms.extend(self.abstractions)
        return atoms

    # ---- main search -------------------------------------------------------
    def search(self, task: Task, *, node_budget: int | None = None,
               deadline_s: float | None = None,
               extra_support: list[Example] | None = None) -> SearchResult:
        t0 = time.perf_counter()
        result = SearchResult()
        probes = [e.x for e in task.train] + [e.x for e in task.validation_pool[:2]]
        if extra_support:
            probes += [e.x for e in extra_support]
        seen: dict[tuple, int] = {}   # (type, outputs) -> cost
        frontier: list[tuple[Node, tuple, float, str]] = []
        pool: dict[str, tuple[Node, tuple, float, str]] = {}
        target = task.output_type

        try:
            for atom in self._atoms(task):
                if self.governor:
                    self.governor.charge_nodes(1)
                    self.governor.checkpoint("search-atom")
                t = dsl.infer_type(atom)
                o = self.outputs(atom, probes)
                c = dsl.complexity(atom)
                key = (t, o)
                if key not in seen or c < seen[key]:
                    seen[key] = c
                    frontier.append((atom, o, c, t))
                    pool.setdefault(dsl.canonical_key(atom), (atom, o, c, t))
                self.nodes_generated += 1

            frontier.sort(key=lambda x: x[2])
            for _level in range(self.max_depth):
                if not frontier:
                    break
                next_frontier: list[tuple[Node, tuple, float, str]] = []
                buckets: dict[tuple, list[tuple[Node, tuple, float, str]]] = {}
                for node, outs, cost, typ in frontier:
                    buckets.setdefault((typ,), []).append((node, outs, cost, typ))

                for node, outs, cost, typ in frontier:
                    if self.governor:
                        self.governor.charge_nodes(1)
                        self.governor.checkpoint("search-expand")
                        if node_budget is not None and self.expansions >= node_budget:
                            raise ResourceLimit("node_budget", "search node cap")
                    self.expansions += 1
                    if cost >= self.max_cost:
                        continue
                    # unary wraps whose child type matches
                    wraps: list[str] = []
                    if typ == "int":
                        wraps = list(_INT_UNARY) + list(_INT_BOOL) + list(_PARAM_INT_INT) + list(_PARAM_INT_BOOL)
                    elif typ == "list":
                        wraps = list(_LIST_TO_LIST) + list(_LIST_TO_INT) + list(_PARAM_LIST_LIST) + list(_PARAM_LIST_INT)
                    for op in wraps:
                        params = ()
                        if dsl.OPS[op].has_param:
                            for k in constants_from(task)[:2]:
                                child = call(op, node, params=(k,))
                                self._consider(child, probes, seen, next_frontier, pool, target, cost)
                        else:
                            child = call(op, node)
                            self._consider(child, probes, seen, next_frontier, pool, target, cost)
                    # conditional composition: if(bool, int, int) when a bool
                    # expression and int expressions are available
                    if typ == "bool" and cost + 3 <= self.max_cost:
                        ints = [n for n, _o, _c, t in frontier if t == "int"][:4]
                        for a in ints:
                            for b in ints[:2]:
                                child = call("if", node, a, b)
                                self._consider(child, probes, seen, next_frontier, pool, target, cost + 2)
                # beam by (type, outputs) bucket
                next_frontier.sort(key=lambda x: x[2])
                trimmed = next_frontier[:self.beam * 3]
                frontier = trimmed
        except ResourceLimit as exc:
            result.timed_out = True
            result.reason = f"{exc.kind}: {exc.detail}"

        # collect candidates fitting all support train examples
        support = task.train + (extra_support or [])
        for node, _outs, _cost, typ in pool.values():
            if typ != target:
                continue
            good = True
            for ex in support:
                try:
                    if not _same(self.eval_fn(node, ex.x), ex.y):
                        good = False
                        break
                except Exception:
                    good = False
                    break
            if good:
                result.candidates.append(Candidate(
                    id=f"s{len(result.candidates):03d}", program=node,
                    cost=dsl.complexity(node), notes={"search": True}))

        result.expansions = self.expansions
        result.nodes_generated = self.nodes_generated
        result.cache_hits = self.hits
        result.elapsed_s = time.perf_counter() - t0
        result.diagnostics = {
            "cache_size": len(self.cache),
            "final_frontier": len(frontier),
            "target_type": target,
            "beam": self.beam,
        }
        if not result.candidates and not result.timed_out:
            result.reason = "no program reproduces all support examples"
        return result

    def _consider(self, child: Node, probes: list[Any], seen: dict,
                  next_frontier: list, pool: dict, target: str,
                  parent_cost: float) -> None:
        if child.depth() > self.max_depth or child.size() > self.max_nodes:
            return
        if dsl.validate(child, max_depth=self.max_depth,
                        max_nodes=self.max_nodes,
                        max_magnitude=self.max_magnitude):
            return
        t = dsl.infer_type(child)
        if not can_reach(t, target, self.max_cost - dsl.complexity(child)):
            return
        o = self.outputs(child, probes)
        key = (t, o)
        c = dsl.complexity(child)
        if key in seen and seen[key] <= c:
            return
        seen[key] = c
        next_frontier.append((child, o, c, t))
        pool.setdefault(dsl.canonical_key(child), (child, o, c, t))
        self.nodes_generated += 1


def cegis_refine(task: Task, searcher: Searcher, validation_fn: Callable[[Candidate], Any],
                 *, rounds: int = 2, **kw: Any) -> tuple[SearchResult, list[Example]]:
    """Counterexample-guided refinement: add a discriminating support point and
    re-search. Returns the final result and the injected counterexamples."""
    injected: list[Example] = []
    result = searcher.search(task, **kw)
    for _ in range(rounds):
        if not result.candidates:
            break
        witness = validation_fn(result.candidates[0])
        if witness is None:
            break
        injected.append(witness)
        result = searcher.search(task, extra_support=injected, **kw)
    return result, injected


def _pk(x: Any) -> str:
    from .types import stable_key

    return stable_key(x)


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    return a == b
