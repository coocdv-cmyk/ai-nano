"""K1 — Deep-composition root-cause investigation.

Matched arms isolate which safeguard reduces incorrect acceptance: shifted
validation, counterexample-guided refinement, oracle intermediate contracts
(an explicit non-deployable upper bound), and conservative abstention. Every
incorrectly accepted program is traced to its first incorrect intermediate step.
"""

from __future__ import annotations

import random
from typing import Any

from constructive.exp_i import dsl
from constructive.exp_i.resources import ResourceGovernor
from constructive.exp_i.search import Searcher, cegis_refine
from constructive.exp_i.types import Candidate, Example, Task, ValidationState

from . import constructors as C
from . import generators as G
from . import metrics as M
from .config import ExpKConfig
from .engine import run_validation
from .resources import Monitor
from .types import Completion, DeepOutcome

ARMS = ("baseline", "shifted_validation", "cegis", "oracle_contract",
        "conservative_abstention")


def _walk(node):
    out = [node]
    for c in node.children:
        out.extend(_walk(c))
    return out


def _equivalent(program, hidden, probes) -> bool:
    for x in probes:
        try:
            if dsl.evaluate(program, x) != dsl.evaluate(hidden, x):
                return False
        except Exception:
            return False
    return True


def trace_first_error(program, hidden, probes) -> dict[str, Any]:
    """First composition level at which the candidate's outputs diverge.

    For each hidden subexpression depth, look for a candidate subexpression of
    the same depth whose outputs match the hidden intermediate on all probes.
    The first depth where none matches is the first incorrect intermediate step.
    """
    # Innermost-first order: the first divergence is the earliest composition step.
    h_nodes = sorted((n for n in _walk(hidden) if n.op != "var"),
                     key=lambda n: n.depth())
    c_by_depth: dict[int, list] = {}
    for n in _walk(program):
        if n.op == "var":
            continue
        c_by_depth.setdefault(n.depth(), []).append(n)
    first = None
    for h in h_nodes:
        d = h.depth()
        want = [dsl.evaluate(h, x) for x in probes]
        found = False
        for cand in c_by_depth.get(d, []):
            try:
                if [dsl.evaluate(cand, x) for x in probes] == want:
                    found = True
                    break
            except Exception:
                continue
        if not found:
            first = d
            break
    final_depth = max((n.depth() for n in h_nodes), default=0)
    return {
        "first_error_depth": first,
        "hidden_steps": len(h_nodes),
        "final_depth": final_depth,
        "error_propagated": bool(first is not None and first < final_depth),
        "candidate": str(program), "hidden": str(hidden),
    }


def _shifted_validation(task: Task, hidden, *, n: int = 5) -> Task:
    rng = random.Random(sum(ord(c) for c in task.name) + 77)
    xs = G._uniq_ints(rng, n, 100, 160, [e.x for e in task.test])
    extra = [Example(x, dsl.evaluate(hidden, x)) for x in xs]
    return Task(name=task.name + "-shiftval", family=task.family,
                train=list(task.train),
                validation_pool=list(task.validation_pool) + extra,
                test=list(task.test), input_type=task.input_type,
                output_type=task.output_type, meta=dict(task.meta))


def _validation_fn(task: Task):
    def witness(cand: Candidate):
        for ex in task.validation_pool:
            try:
                if dsl.evaluate(cand.program, ex.x) != ex.y:
                    return ex
            except Exception:
                return ex
        return None
    return witness


def _classify(fields, hidden_ok: bool, timed_out: bool) -> DeepOutcome:
    if timed_out:
        return DeepOutcome.BUDGET
    if fields["wrong"]:
        return DeepOutcome.INCORRECT
    if fields["correct"]:
        return DeepOutcome.CORRECT
    if fields["non_identifiable"]:
        return DeepOutcome.NON_IDENTIFIABLE
    if fields["abstained"]:
        return (DeepOutcome.CORRECT_ABSTENTION if not hidden_ok
                else DeepOutcome.INCORRECT_ABSTENTION)
    return DeepOutcome.INCORRECT_ABSTENTION


def run_k1(cfg: ExpKConfig, *, seeds=None, arms=ARMS) -> dict[str, Any]:
    seeds = tuple(seeds if seeds is not None else cfg.seeds)
    mon = Monitor()
    rows: list[dict[str, Any]] = []
    for arm in arms:
        for depth in cfg.depths:
            for seed in seeds:
                task, hidden = C.deep_chain_task(seed, depth)
                gov = ResourceGovernor(global_deadline_s=cfg.fit_deadline_s,
                                       node_budget=cfg.search_node_budget)
                s = Searcher(beam=cfg.search_beam,
                             max_cost=min(cfg.max_cost, depth + 1.0),
                             max_depth=depth, max_nodes=64, governor=gov)
                try:
                    if arm == "cegis":
                        result, injected = cegis_refine(
                            task, s, _validation_fn(task), rounds=1,
                            node_budget=cfg.search_node_budget,
                            deadline_s=cfg.fit_deadline_s)
                    else:
                        result = s.search(task, node_budget=cfg.search_node_budget,
                                          deadline_s=cfg.fit_deadline_s)
                        injected = []
                except Exception as exc:  # noqa: BLE001
                    rows.append({"arm": arm, "depth": depth, "seed": seed,
                                 "outcome": DeepOutcome.BUDGET.value,
                                 "error": f"{type(exc).__name__}: {exc}",
                                 "status": "error"})
                    mon.mark_error(str(exc))
                    continue

                eval_task = _shifted_validation(task, hidden) if arm in (
                    "shifted_validation", "cegis", "conservative_abstention") else task
                eval_task.meta["candidates"] = result.candidates
                _r, fields = run_validation(eval_task, cfg)

                probes = [e.x for e in task.train + task.validation_pool + task.test]
                hidden_ok = any(_equivalent(c.program, hidden, probes)
                                for c in result.candidates)

                # Oracle-contract arm: reject a candidate whose intermediate
                # values disagree with the hidden chain (requires ground truth;
                # reported only as a non-deployable upper bound).
                oracle_reject = False
                trace = None
                if fields["accepted"]:
                    prog = next((c.program for c in result.candidates
                                 if c.id == _r.candidate_id), None)
                    if prog is not None:
                        trace = trace_first_error(prog, hidden, probes)
                        oracle_reject = trace["first_error_depth"] is not None
                if arm == "oracle_contract" and oracle_reject:
                    fields = dict(fields)
                    fields.update(accepted=False, correct=False, wrong=False,
                                  abstained=True, state="ORACLE_CONTRACT_REJECT",
                                  test_accuracy=0.0)

                outcome = _classify(fields, hidden_ok, bool(result.timed_out))
                rows.append({
                    "arm": arm, "depth": depth, "seed": seed,
                    "outcome": outcome.value, **fields,
                    "expansions": result.expansions,
                    "nodes": result.nodes_generated,
                    "cache_hits": result.cache_hits,
                    "elapsed_s": round(result.elapsed_s, 6),
                    "timed_out": bool(result.timed_out),
                    "n_candidates": len(result.candidates),
                    "n_injected": len(injected),
                    "n_val_examples": len(eval_task.validation_pool),
                    "hidden_discovered": hidden_ok,
                    "first_error_depth": trace["first_error_depth"] if trace else None,
                    "error_propagated": trace["error_propagated"] if trace else False,
                    "status": "completed",
                })
                mon.charge(expansions=result.expansions, nodes=result.nodes_generated,
                           validation_calls=len(eval_task.validation_pool))
                if result.timed_out:
                    mon.mark_timeout(f"{arm}/d{depth}/s{seed}")

    by_arm: dict[str, Any] = {}
    for arm in arms:
        sel = [r for r in rows if r["arm"] == arm]
        wrong = [r for r in sel if r["outcome"] == DeepOutcome.INCORRECT.value]
        by_arm[arm] = {
            "n": len(sel), "confusion": M.confusion(sel),
            "incorrect_acceptance": M.rate(r["outcome"] == DeepOutcome.INCORRECT.value
                                           for r in sel),
            "correct_acceptance": M.rate(r["outcome"] == DeepOutcome.CORRECT.value
                                         for r in sel),
            "coverage": M.rate(r["accepted"] for r in sel),
            "abstention": M.rate(r["abstained"] for r in sel),
            "false_rejection": M.rate(r["false_rejection"] for r in sel),
            "first_error_depth": M.stats(r["first_error_depth"] for r in wrong),
            "error_propagation": M.rate(r["error_propagated"] for r in wrong),
            "hidden_discovered": M.rate(r["hidden_discovered"] for r in sel),
            "expansions": M.stats(r.get("expansions") for r in sel),
            "validation_calls": M.stats(r.get("n_val_examples", 0) for r in sel),
            "elapsed_s": M.stats(r.get("elapsed_s") for r in sel),
            "timed_out": sum(1 for r in sel if r.get("timed_out")),
        }
    by_depth_arm = {}
    for arm in arms:
        for d in cfg.depths:
            sel = [r for r in rows if r["arm"] == arm and r["depth"] == d]
            by_depth_arm[f"{arm}/d{d}"] = {
                "n": len(sel),
                "incorrect_acceptance": M.rate(
                    r["outcome"] == DeepOutcome.INCORRECT.value for r in sel),
                "correct_acceptance": M.rate(
                    r["outcome"] == DeepOutcome.CORRECT.value for r in sel),
                "abstention": M.rate(r["abstained"] for r in sel),
            }

    key = lambda r: (r["depth"], r["seed"])
    baseline_by_key = {key(r): r["outcome"] == DeepOutcome.INCORRECT.value
                       for r in rows if r["arm"] == "baseline"}
    comparison: dict[str, Any] = {}
    for arm in arms:
        if arm == "baseline":
            continue
        treatment_by_key = {key(r): r["outcome"] == DeepOutcome.INCORRECT.value
                            for r in rows if r["arm"] == arm}
        shared = sorted(set(baseline_by_key) & set(treatment_by_key))
        comparison[arm] = M.compare(
            [baseline_by_key[k] for k in shared],
            [treatment_by_key[k] for k in shared], lower_is_better=True)
        comparison[arm]["n_paired"] = len(shared)
    out = {
        "stage": "K1", "experiment_id": "K1",
        "experiment_name": "Deep Composition Root-Cause Investigation",
        "status": "completed", "seeds": list(seeds), "depths": list(cfg.depths),
        "arms": list(arms), "n_runs": len(rows), "episodes": rows,
        "by_arm": by_arm, "by_depth_arm": by_depth_arm,
        "comparison_to_baseline": comparison,
        "oracle_note": ("oracle_contract uses the hidden chain and is NOT deployable; "
                        "it is reported as an upper bound on what perfect intermediate "
                        "verification could achieve."),
        "budgets": {"node_budget": cfg.search_node_budget,
                    "fit_deadline_s": cfg.fit_deadline_s},
        "limitations": ["Shifted validation inputs are generator-side validation data, not test data."],
    }
    out.update(mon.finish())
    return out
