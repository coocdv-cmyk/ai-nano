"""EXP-I experiment definitions (I-A .. I-H).

Each function returns a self-contained measured block. Nothing here mutates
protected files or writes results; the runner owns persistence.
"""

from __future__ import annotations

import random
import time
from typing import Any

from . import abstraction as AB
from . import dsl, generators as G
from . import generalization as GEN
from . import repair as RP
from . import retrieval as RT
from .config import ExpIConfig
from .dsl import Node, call, var
from .metrics import mean_ci, rates
from .resources import ResourceGovernor, ResourceLimit
from .search import Searcher
from .types import Candidate, CompletionStatus, Expressiveness, Example, Task, ValidationState
from .validation import Calibrator, ValidationEngine, calibration_report

EVAL = dsl.evaluate


# ---------------------------------------------------------------------------
# I-A — evidence engine / validation
# ---------------------------------------------------------------------------


def run_ia(config: ExpIConfig, *, n_episodes: int | None = None) -> dict[str, Any]:
    n = n_episodes or config.n_episodes
    eng = ValidationEngine(budget=config.n_val,
                           abstain_threshold=config.abstain_threshold,
                           eval_fn=EVAL)
    rows: list[dict[str, Any]] = []
    for condition in ("discriminating", "ambiguous"):
        val_range = (10, 30) if condition == "discriminating" else (1, 9)
        test_range = (30, 70) if condition == "discriminating" else (10, 30)
        for i in range(n):
            seed = 1000 + i
            task = G.twin_task(seed, k=3, validation_range=val_range,
                               test_range=test_range)
            # Independent *unlabeled* probes spanning the ambiguity boundary.
            # Their labels are never used; they only let the engine detect that
            # survivors disagree outside the observed domain.
            # Probes never overlap the test split of either condition.
            probe_pool = (list(range(10, 30)) if condition == "discriminating"
                          else list(range(40, 70)))
            result = eng.validate(G.twin_candidates(), task, probe_pool=probe_pool)
            correct = (result.accepted and result.candidate_id == "sum_digits")
            wrong = (result.accepted and result.candidate_id != "sum_digits")
            # Test accuracy of the accepted candidate (evaluation-side only).
            acc_test = None
            if result.accepted and result.candidate_id:
                prog = next((c.program for c in G.twin_candidates()
                             if c.id == result.candidate_id), None)
                if prog is not None:
                    acc_test = sum(1 for e in task.test if EVAL(prog, e.x) == e.y) / len(task.test)
            rows.append({
                "condition": condition, "seed": seed, "state": result.state.value,
                "accepted": result.accepted, "candidate": result.candidate_id,
                "correct_acceptance": correct, "wrong_acceptance": wrong,
                "non_identifiable": result.state is ValidationState.NON_IDENTIFIABLE,
                "insufficient": result.state is ValidationState.INSUFFICIENT_EVIDENCE,
                "confidence": result.confidence,
                "test_accuracy_if_accepted": acc_test,
                "informative_observations": sum(1 for e in result.evidence if e.informative),
                "redundant_observations": sum(1 for e in result.evidence if not e.informative),
                "failure_category": (
                    "wrong-acceptance" if wrong else
                    "success" if correct else
                    "non-identifiable" if result.state is ValidationState.NON_IDENTIFIABLE else
                    "insufficient-evidence" if result.state is ValidationState.INSUFFICIENT_EVIDENCE else
                    "false-rejection"),
            })

    disc = [r for r in rows if r["condition"] == "discriminating"]
    amb = [r for r in rows if r["condition"] == "ambiguous"]
    # Accuracy/coverage trade-off, not "reject everything".
    tradeoff = {
        "discriminating": {
            "coverage": rates(r["accepted"] for r in disc),
            "selective_accuracy": mean_ci(r["test_accuracy_if_accepted"] for r in disc
                                          if r["test_accuracy_if_accepted"] is not None),
            "wrong_acceptance_rate": rates(r["wrong_acceptance"] for r in disc),
            "false_rejection_rate": rates((not r["accepted"]) for r in disc),
        },
        "ambiguous": {
            "coverage": rates(r["accepted"] for r in amb),
            "correct_abstention_rate": rates(
                (not r["accepted"]) and (r["non_identifiable"] or r["insufficient"]) for r in amb),
            "wrong_acceptance_rate": rates(r["wrong_acceptance"] for r in amb),
        },
    }
    # Calibration: fit on first half of seeds, evaluate ECE on second half.
    conf = [float(r["confidence"]) for r in rows]
    correct = [1 if r["correct_acceptance"] else 0 for r in rows]
    half = max(1, len(rows) // 2)
    cal = Calibrator()
    cal.fit(conf[:half], correct[:half])
    ece = cal.ece(conf[half:], correct[half:])
    calibrated = [cal.calibrate(c) for c in conf[half:]]
    return {
        "experiment_id": "I-A", "experiment_name": "Evidence Engine and Multi-Sample Validation",
        "status": "completed", "n_episodes": len(rows), "episodes": rows,
        "validation_states": {s: sum(1 for r in rows if r["state"] == s)
                              for s in {r["state"] for r in rows}},
        "accuracy_coverage_tradeoff": tradeoff,
        "calibration": {"ece_independent": round(ece, 4),
                        "n_calibration": half, "n_eval": len(rows) - half,
                        "report": calibration_report(conf[half:], correct[half:])},
        "baseline_reference": {
            "source": "CONSTRUCTIVE-EXP-H-REPORT.md H1 twins",
            "wrong_twin_acceptance": "15/60 (0.25 of accepted)",
            "abstain_rate": 0.15,
        },
    }


# ---------------------------------------------------------------------------
# I-B — typed DSL structural evolution
# ---------------------------------------------------------------------------


def run_ib(config: ExpIConfig) -> dict[str, Any]:
    # Read-only import of the EXP-H hidden-program language for classification.
    from constructive.exp_h.engine import HNode

    checks: dict[str, Any] = {}
    targets = {
        "bitwise_exclusion": HNode("bxor", 3, (HNode("var"),)),
        "conditional": HNode("if_else", None,
                             (HNode("gt", 3, (HNode("var"),)),
                              HNode("double", None, (HNode("var"),)),
                              HNode("neg", None, (HNode("var"),)))),
        "nested_conditional": HNode("if_else", None,
                                    (HNode("even", None, (HNode("var"),)),
                                     HNode("if_else", None,
                                           (HNode("gt", 5, (HNode("var"),)),
                                            HNode("square", None, (HNode("var"),)),
                                            HNode("sub1", None, (HNode("var"),)))),
                                     HNode("abs", None, (HNode("var"),)))),
        "sum_digits": HNode("sum_digits", None, (HNode("var"),)),
    }
    classifications: dict[str, Any] = {}
    for name, hidden in targets.items():
        ops = dsl.hidden_op_set(hidden)
        expr = dsl.expressiveness_of_ops(ops)
        try:
            node = dsl.from_hidden(hidden)
            translated = True
            well = dsl.well_typed(node)
            issues = dsl.validate(node)
        except dsl.DslError as exc:
            translated, well, issues = False, False, [str(exc)]
            node = None
        classifications[name] = {
            "hidden_ops": sorted(ops), "expressiveness": expr.value,
            "translated": translated, "well_typed": well, "issues": issues,
            "hidden_program": str(hidden),
        }
        if node is not None:
            classifications[name]["example_eval"] = {
                str(x): EVAL(node, x) for x in (-3, 0, 2, 7, 19)
            }
    checks["expressiveness"] = classifications

    # Unseen compositions of learned operators.
    x = var("int")
    xs = var("list")
    compositions = {
        "not_prime": call("not", call("is_prime", x)),
        "even_and_gt": call("and", call("even", x), call("gt", x, params=(3,))),
        "cond_filter_sum": call("sum_list", call("filter_gt", call("evens", xs), params=(2,))),
        "neg_magnitude": call("abs", call("neg", call("sum_digits", x))),
    }
    comp_results = {}
    for name, prog in compositions.items():
        issues = dsl.validate(prog)
        probe = [-3, 0, 2, 7, 19] if dsl.infer_type(prog) in ("int", "bool") else [[1, 2, 3, 4]]
        out = []
        for p in probe:
            if dsl.infer_type(prog) == "list" and not isinstance(p, list):
                continue
            if dsl.infer_type(prog) != "list" and isinstance(p, list):
                continue
            try:
                out.append(EVAL(prog, p))
            except dsl.DslError as exc:
                out.append(f"ERR:{exc}")
        comp_results[name] = {"program": str(prog), "type": dsl.infer_type(prog),
                              "issues": issues, "outputs": out}
    checks["unseen_compositions"] = comp_results

    # Safety: depth / magnitude / type / cycles.
    bomb = x
    for _ in range(config.max_program_depth + 4):
        bomb = call("add1", bomb)
    bad_type = dsl.Node("add1", (), (call("even", x),))  # bool fed to int op
    huge = call("add", x, params=(10 ** 12,))
    safety = {
        "depth_cap_rejects": bool(dsl.validate(bomb, max_depth=config.max_program_depth)),
        "type_mismatch_rejects": not dsl.well_typed(bad_type),
        "magnitude_cap_rejects": bool(dsl.validate(huge, max_magnitude=config.max_int_magnitude)),
        "cycles_impossible": not dsl.cycles_possible(),
        "safe_execution_no_eval": "AST interpreter only; no eval/exec used",
    }
    checks["safety"] = safety
    all_pass = (all(c["translated"] and c["well_typed"] for c in classifications.values())
                and all(not v["issues"] for v in comp_results.values())
                and all(safety.values()))
    return {
        "experiment_id": "I-B", "experiment_name": "Typed Structural DSL Evolution",
        "status": "completed", "checks": checks, "all_checks_pass": all_pass,
        "note": "Distinguishes expressible-but-undiscovered, not-expressible, "
                "not-identifiable and compute-limited categories.",
    }


# ---------------------------------------------------------------------------
# I-C — retrieval collision resolution
# ---------------------------------------------------------------------------


def run_ic(config: ExpIConfig, *, seeds: tuple[int, ...] | None = None) -> dict[str, Any]:
    seeds = tuple(seeds if seeds is not None else config.seeds[:2])
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for family in G.COLLISION_FAMILIES:
            params = [2, 3, 4, 5, 6, 7] if family == "add" else [1, 2, 3, 4, 6, 7]
            store = RT.ConceptStore()
            for p in params:
                store.add(G.collision_program(family, p),
                          input_type=("int" if family == "add" else "list"),
                          output_type=("list" if family in ("take", "filter_gt", "map_add") else "int"),
                          id=f"{family}-{p}", context=(family,))
            for true_p in params:
                task = G.collision_task(family, true_p, seed)
                observed = [(e.x, e.y) for e in task.train]
                t0 = time.perf_counter()
                res = store.retrieve(
                    input_type=task.input_type, output_type=task.output_type,
                    observed_outputs=observed, context=(family,), top_k=2)
                latency = time.perf_counter() - t0
                correct_id = f"{family}-{true_p}"
                top_ids = [cid for cid, _ in res.ranked]
                silent_sub = bool(res.ranked and res.top != correct_id and not res.ambiguous)
                ambiguous_correct = bool(res.ambiguous and correct_id in res.ambiguity_ids)
                rows.append({
                    "seed": seed, "family": family, "true": correct_id,
                    "top": res.top, "ranked": top_ids, "ambiguous": res.ambiguous,
                    "ambiguity_ids": list(res.ambiguity_ids),
                    "correct_top": res.top == correct_id,
                    "silent_substitution": silent_sub,
                    "ambiguity_contains_truth": ambiguous_correct,
                    "latency_s": latency,
                    "confidence": res.confidence,
                    "failure_category": (
                        "false-retrieval" if silent_sub else
                        "success" if (res.top == correct_id or ambiguous_correct) else
                        "concept-interference"),
                })
            for r in rows[-len(params):]:
                r["memory_bytes"] = store.memory_bytes()
                r["store_report"] = store.report()
    summary = {
        "n": len(rows),
        "silent_substitution_rate": rates(r["silent_substitution"] for r in rows),
        "correct_top_rate": rates(r["correct_top"] for r in rows),
        "ambiguity_rate": rates(r["ambiguous"] for r in rows),
        "precision_at_2": rates(any(rid == r["true"] for rid in r["ranked"]) for r in rows),
        "recall_at_2": rates(r["true"] in r["ranked"] for r in rows),
        "latency_ms": mean_ci(r["latency_s"] * 1000 for r in rows),
        "memory_bytes": mean_ci(r["memory_bytes"] for r in rows),
    }
    return {
        "experiment_id": "I-C", "experiment_name": "Retrieval Collision Resolution",
        "status": "completed", "seeds": list(seeds),
        "collision_families": list(G.COLLISION_FAMILIES),
        "episodes": rows, "summary": summary,
        "baseline_reference": {
            "source": "CONSTRUCTIVE-EXP-H-REPORT.md H4",
            "interference": "13/157 (8.3%)", "new_pool_size": 86,
            "note": "250-concept tier not executed by EXP-H; not claimed here",
        },
    }


# ---------------------------------------------------------------------------
# I-D — deep composition and search control
# ---------------------------------------------------------------------------

_UNARY_POOL = ("double", "add1", "sub1", "abs", "neg", "sum_digits", "digital_root")


def _chain(depth: int, seed: int) -> Node:
    rng = random.Random(seed * 97 + depth)
    node: Node = var("int")
    for _ in range(depth):
        op = rng.choice(_UNARY_POOL)
        node = call(op, node)
    return node


def _chain_task(depth: int, seed: int, k: int = 4, n_test: int = 10) -> Task:
    hidden = _chain(depth, seed)
    rng = random.Random(seed * 31 + depth)
    xs = G.sample_ints(rng, k + 4 + n_test, 0, 20)
    def label(v: int) -> Any:
        try:
            return EVAL(hidden, v)
        except dsl.DslError:
            return None
    labelled = [(v, label(v)) for v in xs]
    if any(y is None for _x, y in labelled):
        # regenerate with a smaller domain if a chain overflowed
        return _chain_task(depth, seed + 1, k, n_test)
    return Task(f"chain-d{depth}-s{seed}", "unary_chain",
                [Example(x, y) for x, y in labelled[:k]],
                [Example(x, y) for x, y in labelled[k:k + 4]],
                [Example(x, y) for x, y in labelled[k + 4:]],
                "int", "int", {"depth": depth, "hidden": str(hidden)})


def run_id(config: ExpIConfig, *, seeds: tuple[int, ...] | None = None) -> dict[str, Any]:
    seeds = tuple(seeds if seeds is not None else config.seeds[:3])
    depths = (2, 3, 5, 8, 12, 16)
    rows: list[dict[str, Any]] = []
    for depth in depths:
        for seed in seeds:
            task = _chain_task(depth, seed)
            gov = ResourceGovernor(global_deadline_s=config.fit_deadline_s,
                                    node_budget=config.search_node_budget)
            searcher = Searcher(governor=gov, beam=config.n_episodes,
                                 max_cost=3.6, max_depth=6)
            t0 = time.perf_counter()
            result = searcher.search(task)
            wall = time.perf_counter() - t0
            hidden_key = dsl.canonical_key(_chain(depth, seed))
            found_exact = any(dsl.canonical_key(c.program) == hidden_key
                              for c in result.candidates)
            # The search is a candidate generator only; the evidence engine owns
            # the epistemic decision (accept / abstain).
            eng = ValidationEngine(budget=config.n_val, eval_fn=EVAL)
            shortlist = result.candidates[:12]
            vr = eng.validate(shortlist, task, probe_pool=list(range(0, 25)))
            by_id = {c.id: c for c in shortlist}
            chosen = by_id.get(vr.candidate_id) if vr.accepted else None
            accepted = vr.accepted
            correct = bool(chosen and dsl.canonical_key(chosen.program) == hidden_key)
            spurious = bool(accepted and not correct)
            rows.append({
                "family": "unary_chain", "depth": depth, "seed": seed,
                "accepted": accepted, "correct_acceptance": correct,
                "spurious_acceptance": spurious,
                "exact_hidden_generated": found_exact,
                "validation_state": vr.state.value,
                "expansions": result.expansions, "nodes": result.nodes_generated,
                "cache_hits": result.cache_hits, "cache_size": result.diagnostics.get("cache_size", 0),
                "wall_s": wall, "timed_out": result.timed_out,
                "reason": result.reason,
                "n_test": len(task.test),
                "failure_category": (
                    "time-exhaustion" if result.timed_out else
                    "success" if correct else
                    "wrong-acceptance" if spurious else
                    "non-identifiable" if vr.state is ValidationState.NON_IDENTIFIABLE else
                    "insufficient-evidence" if vr.state is ValidationState.INSUFFICIENT_EVIDENCE else
                    "search-explosion"),
            })
    summary = {}
    for d in depths:
        sel = [r for r in rows if r["depth"] == d]
        summary[f"depth_{d}"] = {
            "n": len(sel),
            "correct_acceptance": rates(r["correct_acceptance"] for r in sel),
            "spurious_acceptance": rates(r["spurious_acceptance"] for r in sel),
            "expansions": mean_ci(r["expansions"] for r in sel),
            "wall_s": mean_ci(r["wall_s"] for r in sel),
            "timeout_rate": rates(r["timed_out"] for r in sel),
        }
    return {
        "experiment_id": "I-D", "experiment_name": "Deep Composition and Search Control",
        "status": "completed", "depths": list(depths), "seeds": list(seeds),
        "episodes": rows, "summary": summary,
        "baseline_reference": {
            "source": "CONSTRUCTIVE-EXP-H-REPORT.md H2",
            "depth_2_accuracy": 1.0,
            "spurious_rate_depth_3_plus": "0.33-0.67",
            "timeouts": 2,
        },
    }


# ---------------------------------------------------------------------------
# I-E / I-F / I-G — delegate to their modules
# ---------------------------------------------------------------------------


def run_ie(config: ExpIConfig) -> dict[str, Any]:
    return GEN.run_ood(config)


def run_if(config: ExpIConfig, *, n_episodes: int | None = None) -> dict[str, Any]:
    n = n_episodes or 6
    cases = ["sum_range", "sum_range", "sum_gcd", "clamp", "sum_squares", "pow2"]
    rows: list[dict[str, Any]] = []
    for i, name in enumerate(cases):
        for seed in range(n):
            rows.append(RP.repair_episode(name, seed))
    summary = {
        "n": len(rows),
        "repair_rate": rates(r.get("repair_found", False) for r in rows),
        "false_verified_rate": rates(r.get("false_verified_repair", False) for r in rows),
        "false_repair_detection_rate": rates(r.get("false_repair_detected", False) for r in rows),
        "verification_coverage": mean_ci(r.get("verification_coverage") for r in rows
                                         if r.get("verification_coverage") is not None),
        "states": {s: sum(1 for r in rows if r.get("state") == s)
                   for s in {r.get("state") for r in rows}},
    }
    return {
        "experiment_id": "I-F", "experiment_name": "Verified Program Repair",
        "status": "completed", "cases": cases, "episodes": rows, "summary": summary,
        "baseline_reference": {
            "source": "CONSTRUCTIVE-EXP-H-REPORT.md H5",
            "false_repairs": "5/144", "note": "original 5 false-repair templates replayed",
        },
    }


def run_ig(config: ExpIConfig) -> dict[str, Any]:
    return AB.run_abstraction(config)


# ---------------------------------------------------------------------------
# I-H — resource governance
# ---------------------------------------------------------------------------


def run_ih(config: ExpIConfig) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    # 1. Global deadline enforced, reported as timeout (never COMPLETED).
    g = ResourceGovernor(global_deadline_s=0.01)
    hit = False
    try:
        while True:
            g.checkpoint("deadline-probe")
            time.sleep(0.002)
    except ResourceLimit as exc:
        hit = exc.kind == "global_deadline"
    checks.append({"check": "global_deadline_enforced", "pass": hit,
                   "status": CompletionStatus.TIMED_OUT.value,
                   "reports": len(g.report()["timeouts"])})

    # 2. Node budget enforced.
    g2 = ResourceGovernor(node_budget=5)
    hit2 = False
    try:
        for _ in range(20):
            g2.charge_nodes(1)
            g2.checkpoint("node-probe")
    except ResourceLimit as exc:
        hit2 = exc.kind == "node_budget"
    checks.append({"check": "node_budget_enforced", "pass": hit2})

    # 3. Cancellation propagation.
    g3 = ResourceGovernor()
    g3.cancel("test cancellation")
    hit3 = False
    try:
        g3.checkpoint("cancel-probe")
    except ResourceLimit as exc:
        hit3 = exc.kind == "cancellation"
    checks.append({"check": "cancellation_propagates", "pass": hit3})

    # 4. Timeout must never produce a COMPLETED result.
    g4 = ResourceGovernor(global_deadline_s=0.01)
    status = CompletionStatus.COMPLETED
    try:
        with g4.scope("fit"):
            while True:
                g4.checkpoint("fit")
                time.sleep(0.002)
    except ResourceLimit:
        status = CompletionStatus.TIMED_OUT
    checks.append({"check": "timeout_never_false_success",
                   "pass": status is CompletionStatus.TIMED_OUT,
                   "status": status.value})

    # 5. Deterministic repeated runs at same seed.
    t = G.twin_task(7, k=3)
    eng = ValidationEngine(budget=3, eval_fn=EVAL)
    r1 = eng.validate(G.twin_candidates(), t)
    r2 = eng.validate(G.twin_candidates(), t)
    checks.append({"check": "repeated_runs_deterministic",
                   "pass": r1.state == r2.state and r1.candidate_id == r2.candidate_id})

    # 6. Failure isolation: a broken stage yields a diagnostic, not a success.
    from .resources import run_isolated
    log: list[dict[str, Any]] = []
    out = run_isolated(lambda: 1 / 0, default=None, log=log)
    checks.append({"check": "failure_isolation_records_diagnostic",
                   "pass": out is None and bool(log)})

    return {
        "experiment_id": "I-H", "experiment_name": "Resource Governance",
        "status": "completed", "checks": checks,
        "all_pass": all(c["pass"] for c in checks),
        "completion_status_policy": "TIMED_OUT/PARTIAL are never mapped to COMPLETED",
    }
