"""EXP-K shared constructors for deep chains, drift streams and retrieval pools.

These builders are used by both baseline and treatment arms so that every
comparison is made on matched, independently generated conditions.
"""

from __future__ import annotations

import random
from typing import Any

from constructive.exp_i import dsl
from constructive.exp_i.dsl import Node, call, var
from constructive.exp_i.retrieval import ConceptStore
from constructive.exp_i.types import Example, Task

from . import generators as G
from .types import DriftKind

SAFE_UNARY = ("add1", "sub1", "neg", "abs", "sum_digits", "digital_root")


def deep_chain_task(seed: int, depth: int) -> tuple[Task, Node]:
    """Chain of exactly `depth` safe unary operations, with trace metadata."""
    rng = random.Random(seed * 131 + depth)
    node: Node = var("int")
    steps = []
    for i in range(depth):
        op = rng.choice(SAFE_UNARY)
        node = call(op, node)
        steps.append({"step": i + 1, "op": op})
    label = lambda x: dsl.evaluate(node, x)
    tx = G._uniq_ints(rng, 4, 0, 9)
    vx = G._uniq_ints(rng, 5, 10, 25, set(tx))
    ex = G._uniq_ints(rng, 12, 30, 80, set(tx + vx))
    task = Task(
        name=f"kchain-d{depth}-s{seed}", family=f"chain-{depth}",
        train=[Example(x, label(x)) for x in tx],
        validation_pool=[Example(x, label(x)) for x in vx],
        test=[Example(x, label(x)) for x in ex],
        input_type="int", output_type="int",
        meta={"hidden": node, "hidden_str": str(node), "depth": depth,
              "steps": steps, "candidates": []},
    )
    return task, node


def _shift_domain(base: dict[str, Any], step: int, gradual: bool) -> dict[str, Any]:
    d = dict(base)
    if "lo" in d:
        move = step * (4 if gradual else 40)
        d["lo"] += move
        d["hi"] += move
    if "len" in d:
        d["len"] += step * (1 if gradual else 4)
        d["len_hi"] += step * (1 if gradual else 4)
    return d


def drift_stream(rule: str, seed: int, kind: DriftKind | str, *,
                 n_base: int = 3, n_drift: int = 4, n_recover: int = 3,
                 n_val: int = 4, n_test: int = 10) -> tuple[list[Task], int, int]:
    """base -> drift -> recover phases with explicit indices."""
    kind = DriftKind(kind)
    tr, base_te = G.domains(rule, "none")
    tasks: list[Task] = []
    idx = 0
    for _ in range(n_base):
        tasks.append(_drift_task(rule, seed, idx, 0, tr, tr, base_te,
                                 n_val, n_test, "base", kind))
        idx += 1
    drift_idx = idx
    for i in range(n_drift):
        step = (i + 1) if kind in (DriftKind.GRADUAL, DriftKind.CYCLIC,
                                    DriftKind.RECURRING) else 2
        moved = base_te if kind is DriftKind.STABLE else _shift_domain(
            base_te, step, kind in (DriftKind.GRADUAL, DriftKind.CYCLIC,
                                    DriftKind.RECURRING))
        tasks.append(_drift_task(rule, seed, idx, step, tr, moved, moved,
                                 n_val, n_test, "drift", kind))
        idx += 1
    recover_idx = idx
    for _ in range(n_recover):
        tasks.append(_drift_task(rule, seed, idx, 0, tr, tr, base_te,
                                 n_val, n_test, "recover", kind))
        idx += 1
    return tasks, drift_idx, recover_idx


def _drift_task(rule, seed, idx, step, tr, vd, td, n_val, n_test, phase, kind) -> Task:
    t = G.build_task(rule, seed * 101 + idx, tr, vd, td,
                     n_val=n_val, n_test=n_test, tag=f"{kind.value}-{phase}")
    t.meta.update({"phase": phase, "drift_step": step, "drift_kind": kind.value})
    return t


def concept_pool(learned_rules: tuple[str, ...], seed: int = 0) -> ConceptStore:
    """Store concepts learned from the declared (in-family) rules only."""
    store = ConceptStore(eval_fn=dsl.evaluate)
    for rule in learned_rules:
        t = G.make_task(rule, seed)
        prog = dict((c.id, c.program) for c in t.meta["candidates"])["true"]
        store.add(prog, input_type=t.input_type, output_type=t.output_type,
                  id=f"c-{rule}", context=(rule,))
    return store
