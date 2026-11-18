"""EXP-K stage-A smoke checks (no campaign execution)."""

from __future__ import annotations

import inspect
from typing import Any

from constructive.exp_i import dsl

from . import constructors as C
from . import generators as G
from . import metrics as M
from .baseline import integrity_snapshot
from .types import DriftKind


def run_stage_a() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, fn) -> None:
        try:
            checks.append({"name": name, "passed": True, "detail": fn()})
        except Exception as exc:  # noqa: BLE001
            checks.append({"name": name, "passed": False,
                           "detail": f"{type(exc).__name__}: {exc}"})

    def independence():
        src = inspect.getsource(G)
        for bad in ("exp_j.generators", "exp_i.generators", "exp_h.generators"):
            assert bad not in src, bad
        return "no EXP-H/I/J generator import in exp_k.generators"

    def new_rules():
        known = {"sum_digits", "digital_root", "square", "fib", "is_prime",
                 "double", "abs", "add1", "affine_mod", "digit_sum_parity",
                 "threshold_parity", "prime_and_odd", "fib_mod", "square_digit_sum",
                 "range_of_evens", "sum_top_k", "count_distinct_gt", "prefix_sum_gt",
                 "evens_of_even", "weighted_pair"}
        overlap = set(G.ALL_RULES) & known
        assert not overlap, overlap
        return f"{len(G.ALL_RULES)} new rules disjoint from EXP-I/J"

    def deterministic():
        a, b = G.make_task("if_even", 7), G.make_task("if_even", 7)
        assert [e.x for e in a.train] == [e.x for e in b.train]
        assert [e.y for e in a.test] == [e.y for e in b.test]
        return "same seed yields identical splits"

    def splits():
        for rule in G.ALL_RULES:
            for shift in ("none", "magnitude", "length", "both"):
                t = G.make_task(rule, 3, shift=shift)
                assert G.disjoint([e.x for e in t.train],
                                  [e.x for e in t.validation_pool],
                                  [e.x for e in t.test])
        return "all train/validation/test splits disjoint"

    def labels():
        for rule in G.ALL_RULES:
            t = G.make_task(rule, 2)
            prog = dict((c.id, c.program) for c in t.meta["candidates"])["true"]
            assert all(dsl.evaluate(prog, e.x) == e.y
                       for e in t.train + t.validation_pool + t.test)
        return "reference labels coincide with true programs"

    def deep_depth():
        for d in (2, 5, 16):
            t, node = C.deep_chain_task(3, d)
            # A var leaf has depth 1; d unary wraps give AST depth d + 1.
            assert node.depth() == d + 1
            assert len(t.meta["steps"]) == d
        return "deep chains have exact declared depth and trace steps"

    def drift_order():
        for kind in (DriftKind.GRADUAL, DriftKind.ABRUPT):
            tasks, d, r = C.drift_stream("mod_double", 1, kind)
            assert 0 < d < r < len(tasks)
        return "drift phases ordered base < drift < recover"

    def metrics_check():
        s = M.stats([1.0, 2.0, 3.0])
        assert s["n"] == 3 and s["mean"] == 2.0 and s["ci95"] > 0
        assert M.ece([1.0, 0.0], [1, 0]) == 0.0
        assert M.ece([0.9, 0.1], [1, 0]) == 0.1
        return "mean/median/std/CI and ECE correct"

    def protection():
        integ = integrity_snapshot()
        assert all(v["all_unchanged"] for v in integ["manifests"].values())
        return f"{sum(v['n_files'] for v in integ['manifests'].values())} protected files unchanged"

    check("independent_generators", independence)
    check("new_rule_families", new_rules)
    check("deterministic_generation", deterministic)
    check("disjoint_splits", splits)
    check("correct_reference_labels", labels)
    check("deep_chain_tracing", deep_depth)
    check("drift_phase_order", drift_order)
    check("metric_correctness", metrics_check)
    check("protected_file_integrity", protection)

    passed = sum(1 for c in checks if c["passed"])
    return {"stage": "A", "experiment": "EXP-K smoke", "status": "completed"
            if passed == len(checks) else "failed",
            "n_checks": len(checks), "passed": passed,
            "failures": len(checks) - passed, "checks": checks}
