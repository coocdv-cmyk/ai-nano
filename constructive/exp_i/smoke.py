"""Fast Stage-A smoke checks for EXP-I (no long experiments)."""

from __future__ import annotations

from typing import Any

from . import dsl, generators as G
from . import retrieval as RT
from .resources import ResourceGovernor, ResourceLimit
from .search import Searcher
from .validation import ValidationEngine
from .types import ValidationState


def run_stage_a() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, fn) -> None:
        try:
            ok = bool(fn())
            checks.append({"name": name, "pass": ok})
        except BaseException as exc:  # noqa: BLE001
            checks.append({"name": name, "pass": False,
                           "error": f"{type(exc).__name__}: {exc}"})

    # DSL: conditional + exclusion + typed composition + safety
    check("dsl_conditional", lambda: dsl.evaluate(
        dsl.call("if", dsl.call("gt", dsl.var("int"), params=(5,)),
                 dsl.call("double", dsl.var("int")),
                 dsl.call("neg", dsl.var("int"))), 7) == 14)
    check("dsl_exclusion", lambda: dsl.evaluate(
        dsl.call("bxor", dsl.var("int"), params=(3,)), 5) == 6)
    def _ill_typed_rejected() -> bool:
        bad = dsl.Node("add1", (), (dsl.call("even", dsl.var("int")),))
        return not dsl.well_typed(bad)

    check("dsl_type_safety", _ill_typed_rejected)
    bomb = dsl.var("int")
    for _ in range(12):
        bomb = dsl.call("add1", bomb)
    check("dsl_depth_cap", lambda: bool(dsl.validate(bomb, max_depth=8)))
    check("dsl_magnitude_cap", lambda: bool(dsl.validate(
        dsl.call("add", dsl.var("int"), params=(10 ** 12,)),
        max_magnitude=10 ** 9)))

    # Generators: disjoint splits
    check("twin_splits_disjoint", lambda: G.disjointify(
        [e.x for e in G.twin_task(3).train],
        [e.x for e in G.twin_task(3).validation_pool],
        [e.x for e in G.twin_task(3).test]))

    # Validation: twin accepted, ambiguous abstained
    eng = ValidationEngine(budget=4, eval_fn=dsl.evaluate)
    disc = G.twin_task(1, k=3)
    amb = G.twin_task(1, k=3, validation_range=(1, 9), test_range=(10, 30))
    check("validation_twin_accepts_correct",
          lambda: eng.validate(G.twin_candidates(), disc).candidate_id == "sum_digits")
    check("validation_ambiguous_abstains",
          lambda: not eng.validate(G.twin_candidates(), amb).accepted)
    check("validation_contradictory_state", lambda: eng.validate(
        G.twin_candidates(), G.contradictory_support(disc)).state is ValidationState.CONTRADICTORY)

    # Retrieval: versioned history + ambiguity-free basic retrieval
    store = RT.ConceptStore()
    store.add(G.collision_program("count_gt", 3), input_type="list",
              output_type="int", id="cg3")
    store.refine("cg3", G.collision_program("count_gt", 4))
    check("retrieval_immutable_history", lambda: len(store.records["cg3"].history) == 1)

    # Search: finds a simple composition, respects node budget
    task = G.numeric_ood_task("sum_of_evens", 0, k=3)
    s = Searcher(beam=40, max_cost=3.5, max_depth=4)
    res = s.search(task)
    check("search_finds_composition", lambda: len(res.candidates) >= 1)
    g = ResourceGovernor(node_budget=5)
    s2 = Searcher(governor=g, beam=40, max_cost=3.5)
    r2 = s2.search(task)
    check("search_node_budget", lambda: r2.timed_out)

    # Resource: deadline + cancellation
    g3 = ResourceGovernor(global_deadline_s=0.01)
    hit = False
    try:
        while True:
            g3.checkpoint("probe")
    except ResourceLimit:
        hit = True
    checks.append({"name": "resource_deadline", "pass": hit})

    passed = sum(1 for c in checks if c["pass"])
    return {"stage": "A", "n_checks": len(checks), "passed": passed,
            "failures": len(checks) - passed, "checks": checks}
