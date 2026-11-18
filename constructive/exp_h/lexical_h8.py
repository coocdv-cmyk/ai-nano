"""Independent alternative search strategies for H8.

This module deliberately does NOT copy ``constructive.model.Searcher`` back;
it re-implements the same enumeration semantics from the same mathematical
primitives with two named, measurable deltas:

1. ``beam_score`` — replaces the beam-pruning heuristic (CCM keeps exact
   fits, then *diversity*, then fit: ``beam-elite`` keeps fit first).
2. ``eager_prune`` — intermediate candidates with zero training matches are
   dropped once below the last cost level (CCM keeps building the doorway).

Everything else — the environment, DSL, seeds, typing rule, dedup, validation
gate — is identical to CCM.  The CCM baseline is never bypassed: it runs
untouched in the runner.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..core import (
    EvalError,
    Term,
    V,
    L,
    build_prims,
    eval_term,
    infer_type,
    same_value,
    term_cost,
    value_key,
    well_typed,
)

UNDEF = object()

# ---------------------------------------------------------------- candidates


@dataclass
class AltCandidate:
    term: Term
    outputs: tuple[Any, ...]
    cost: int
    n_ok: int
    variation: int
    posterior: float = 0.0
    validation_accuracy: float | None = None

    @property
    def key(self):
        return (id(self),)


def _n_ok(vals, targets):
    return sum(1 for v, y in zip(vals, targets) if v is not UNDEF and same_value(v, y))


def _variation(vals):
    return len({("U",) if v is UNDEF else value_key(v) for v in vals})


def alternative_search(examples, input_type, output_type, *, beam: int = 45,
                       max_cost: int = 3, max_nodes: int = 250_000,
                       time_budget_s: float = 4.0, beam_elite: bool = False,
                       eager_prune: bool = False) -> tuple[list[AltCandidate], int]:
    """Bottom-up enumerative search with independent variant knobs.

    ``beam_elite``  pods order by (n_ok, cost, -variation) before diversity.
    ``eager_prune`` drop candidates whose train match count is zero, except at
                    the cheapest level.
    """
    t0 = time.time()
    prims = build_prims(tuple(range(0, 13)))
    inputs = [x for x, _ in examples]
    targets = [y for _, y in examples]
    n = len(inputs)

    levels: dict[tuple[str, int], list[AltCandidate]] = {}
    seen: set[tuple] = set()
    nodes = 0
    timed_out = False

    mid_cap = max(beam, 4000)

    def add(term, vals):
        nonlocal nodes
        typ = infer_type(term, {"x": input_type}, prims, {})
        key = (typ, tuple(("U",) if v is UNDEF else value_key(v) for v in vals))
        if key in seen:
            return
        seen.add(key)
        cost = term_cost(term)
        if cost == max_cost and typ != output_type:
            return
        v = (variation := _variation(vals))

        if eager_prune and cost < max_cost:
            if _n_ok(vals, targets) == 0 and v <= 1 and cost >= 2:
                return  # uninformative intermediate
        c = AltCandidate(term=term, outputs=vals, cost=cost,
                         n_ok=_n_ok(vals, targets), variation=v)
        levels.setdefault((typ, cost), []).append(c)

    # seeds
    add(V("x"), tuple(inputs))
    for lit in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1):
        add(L(lit), (lit,) * n)
    for lit in (True, False):
        add(L(lit), (lit,) * n)

    useful = {0: {output_type}}
    for d in range(1, max_cost + 1):
        acc = set(useful[d - 1])
        for p in prims.values():
            if p.ret in acc:
                acc.update(p.args)
        useful[d] = acc

    def ev(term):
        out = []
        for x in inputs:
            try:
                out.append(eval_term(term, {"x": x}, prims))
            except EvalError:
                out.append(UNDEF)
        return tuple(out)

    for cost in range(1, max_cost + 1):
        for prim in prims.values():
            if nodes >= max_nodes:
                timed_out = True
                break
            if prim.ret not in useful.get(max_cost - cost, set()):
                continue
            if len(prim.args) == 1:
                for e in list(levels.get((prim.args[0], cost - prim.cost), [])):
                    nodes += 1
                    vals = ev(Term(prim.name, (e.term,)))
                    add(Term(prim.name, (e.term,)), vals)
            elif len(prim.args) == 2:
                for c1 in range(0, cost - prim.cost):
                    c2 = cost - prim.cost - c1
                    for a in levels.get((prim.args[0], c1), [])[:beam]:
                        for b in levels.get((prim.args[1], c2), [])[:beam]:
                            if time.time() - t0 > time_budget_s or nodes >= max_nodes:
                                timed_out = True
                                break
                            nodes += 1
                            vals = ev(Term(prim.name, (a.term, b.term)))
                            add(Term(prim.name, (a.term, b.term)), vals)
                        if timed_out:
                            break
                    if timed_out:
                        break

        # prune buckets
        for key in list(levels):
            bucket = levels[key]
            cap = beam if key[1] >= max_cost else mid_cap
            if len(bucket) > cap:
                if beam_elite:
                    bucket.sort(key=lambda e: (-e.n_ok, e.cost, -e.variation, str(e.term)))
                else:
                    bucket.sort(key=lambda e: (-int(e.n_ok == n), -e.variation, -e.n_ok,
                                               str(e.term)))
                levels[key] = bucket[:cap]

        exact = [e for e in levels.get((output_type, cost), []) if e.n_ok == n]
        if exact:
            exact.sort(key=lambda e: (e.cost, str(e.term)))
            exact = exact[:8]
            k_out = max(2, len({value_key(y) for y in targets}))
            # uniform-ish posterior, matches CCM shape
            for i, e in enumerate(exact):
                e.posterior = 1.0 / len(exact)
            return exact, nodes
    # take the best partial solutions
    pool = [e for (typ, _), bucket in levels.items() if typ == output_type for e in bucket]
    pool.sort(key=lambda e: (-e.n_ok, e.cost))
    return pool[:8], nodes
