"""EXP-J fast stage-A smoke checks (no campaign execution)."""

from __future__ import annotations

from typing import Any

from constructive.exp_i import dsl

from . import generators as G
from . import noise as NZ
from .config import PROTECTED, RESULTS_DIR, sha256_file, source_revision
from .types import NoiseKind
from . import drift as DR
from .types import DriftKind

_ROOT = RESULTS_DIR.resolve().parents[2]


def run_stage_a() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, fn):
        try:
            detail = fn()
            checks.append({"name": name, "passed": True, "detail": detail})
        except Exception as exc:  # noqa: BLE001
            checks.append({"name": name, "passed": False,
                           "detail": f"{type(exc).__name__}: {exc}"})

    def gen_independence():
        import inspect
        src = inspect.getsource(G)
        assert "exp_i.generators" not in src and "exp_h.generators" not in src
        assert "REFERENCE_LABELS" not in src
        return "no EXP-H/EXP-I generator import or label table in exp_j.generators"

    def new_rules():
        known = {"sum_digits", "digital_root", "square", "fib", "is_prime",
                 "double", "abs", "add1", "sum_of_evens", "distinct_count",
                 "sum_squares", "count_gt2", "nearest", "same_shape", "left_of"}
        overlap = set(G.ALL_RULES) & known
        assert not overlap, overlap
        return f"{len(G.ALL_RULES)} new rules, disjoint from EXP-I/EXP-H rule names"

    def disjoint_splits():
        for rule in G.ALL_RULES:
            for shift in ("none", "magnitude", "length"):
                t = G.make_task(rule, 3, shift=shift)
                assert G.disjoint([e.x for e in t.train],
                                  [e.x for e in t.validation_pool],
                                  [e.x for e in t.test])
        return "all train/validation/test splits disjoint"

    def correct_label():
        for rule in G.ALL_RULES:
            t = G.make_task(rule, 2)
            prog = dict((c.id, c.program) for c in t.meta["candidates"])["true"]
            assert all(dsl.evaluate(prog, e.x) == e.y
                       for e in t.train + t.validation_pool + t.test)
        return "independent reference labels coincide with true programs"

    def safety():
        n = dsl.call("square", dsl.var("int"))
        assert dsl.validate(n, max_depth=8, max_nodes=64) == []
        assert dsl.evaluate(n, 3) == 9
        return "DSL validation and bounded evaluation operate"

    def noise_invariants():
        t = G.make_task("threshold_parity", 1)
        for kind in NoiseKind:
            n = NZ.apply_noise(t, kind, 0.2, 1)
            assert len(n.test) == len(t.test)
            assert [e.y for e in n.test] == [e.y for e in t.test]
        return "test labels unchanged under all noise kinds"

    def drift_phases():
        tasks, d, r = DR.make_drift_stream("affine_mod", 1, DriftKind.ABRUPT)
        assert 0 < d < r < len(tasks)
        return f"drift phases ordered base<drift<recover ({d},{r})"

    def protected_hashes():
        changed = []
        for f in PROTECTED:
            p = _ROOT / f
            if not p.exists() or sha256_file(p) == "":
                changed.append(f)
        assert not changed, changed
        return f"{len(PROTECTED)} protected files present and hashable"

    check("independent_generators", gen_independence)
    check("new_rule_families", new_rules)
    check("disjoint_splits", disjoint_splits)
    check("correct_reference_labels", correct_label)
    check("dsl_safety", safety)
    check("noise_test_invariance", noise_invariants)
    check("drift_phase_order", drift_phases)
    check("protected_manifest", protected_hashes)

    passed = sum(1 for c in checks if c["passed"])
    return {
        "stage": "A", "experiment": "EXP-J smoke", "n_checks": len(checks),
        "passed": passed, "failures": len(checks) - passed,
        "checks": checks, "source_revision": source_revision(),
        "status": "completed" if passed == len(checks) else "failed",
    }
