"""J5 — Cross-mechanism interaction ablations.

Each combination is run with the mechanism enabled and disabled in a controlled
fashion. A regression is reported, never hidden. No claim is made that
individually successful components remain successful when combined.
"""

from __future__ import annotations

import random
from typing import Any

from constructive.exp_i import abstraction as AB
from constructive.exp_i import dsl, repair
from constructive.exp_i.resources import ResourceGovernor
from constructive.exp_i.search import Searcher
from constructive.exp_i.types import Candidate, Example, Task, ValidationState

from . import deep_composition as DC
from . import generators as G
from . import metrics as M
from . import noise as NZ
from .config import ExpJConfig
from .engine import probe_pool_for, run_validation
from .resources import Monitor
from .types import Arm, NoiseKind


def _regression(base: dict[str, Any], comb: dict[str, Any], key: str, *,
                higher_better: bool = True) -> dict[str, Any]:
    b, c = base.get(key), comb.get(key)
    if b is None or c is None:
        return {"key": key, "baseline": b, "combined": c, "regression": None}
    if higher_better:
        reg = c < b
    else:
        reg = c > b
    return {"key": key, "baseline": b, "combined": c, "regression": bool(reg)}


# --- 1. evidence validation + deep search ----------------------------------


def _evidence_deep(cfg: ExpJConfig, seeds: tuple[int, ...]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        task, hidden = DC.make_deep_task(seed, 5)
        s = Searcher(beam=cfg.search_beam, max_cost=5.0, max_depth=5)
        res = s.search(task)
        task.meta["candidates"] = res.candidates
        probes = [e.x for e in task.train + task.validation_pool + task.test]
        for arm in ("search_only", "with_validation"):
            if arm == "search_only":
                cand = res.candidates[0] if res.candidates else None
                accepted = cand is not None
                correct = bool(accepted and DC._equivalent(
                    cand.program, hidden, [e.x for e in task.test]))
                state = "SEARCH_ONLY"
            else:
                vr, fields = run_validation(task, cfg, probe_pool=probe_pool_for(task))
                accepted = fields["accepted"]
                correct = fields["correct_acceptance"]
                state = fields["state"]
            rows.append({"seed": seed, "arm": arm, "accepted": accepted,
                         "correct_acceptance": correct,
                         "wrong_acceptance": bool(accepted and not correct),
                         "state": state, "expansions": res.expansions})
    base = {"wrong_acceptance": M.rate(r["wrong_acceptance"] for r in rows
                                       if r["arm"] == "search_only")["mean"],
            "correct_acceptance": M.rate(r["correct_acceptance"] for r in rows
                                         if r["arm"] == "search_only")["mean"],
            "coverage": M.rate(r["accepted"] for r in rows
                               if r["arm"] == "search_only")["mean"]}
    comb = {"wrong_acceptance": M.rate(r["wrong_acceptance"] for r in rows
                                       if r["arm"] == "with_validation")["mean"],
            "correct_acceptance": M.rate(r["correct_acceptance"] for r in rows
                                         if r["arm"] == "with_validation")["mean"],
            "coverage": M.rate(r["accepted"] for r in rows
                               if r["arm"] == "with_validation")["mean"]}
    return {"episodes": rows, "search_only": base, "with_validation": comb,
            "deltas": [
                _regression(base, comb, "wrong_acceptance", higher_better=False),
                _regression(base, comb, "correct_acceptance"),
                _regression(base, comb, "coverage"),
            ]}


# --- 2. retrieval + abstraction --------------------------------------------


def _retrieval_abstraction(cfg: ExpJConfig, seeds: tuple[int, ...]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        store = AB.AbstractionStore()
        for idx, task in enumerate(G.abstraction_stream_j(seed, "shared-core")[:4]):
            gov = ResourceGovernor(global_deadline_s=cfg.fit_deadline_s)
            bodies = store.active_bodies()
            s = Searcher(governor=gov, beam=cfg.search_beam, max_cost=3.5,
                         max_depth=5, abstractions=bodies)
            res = s.search(task)
            winner = res.candidates[0].program if res.candidates else None
            if winner is not None:
                store.harvest(winner, [task.name], [c.program for c in res.candidates[:3]])
                for a in store.items.values():
                    if not a.validated:
                        store.validate(a.id, G.abstraction_stream_j(seed + 1)[:2])
            for arm in ("all_abstractions", "retrieval_ranked"):
                if arm == "all_abstractions":
                    sel = store.active_bodies()
                else:
                    sel = [a.body for a in store.items.values()
                           if a.active and a.validated and _agree(a.body, task) >= 0.5]
                s2 = Searcher(beam=cfg.search_beam, max_cost=3.5,
                              max_depth=5, abstractions=sel)
                res2 = s2.search(task)
                rows.append({"seed": seed, "index": idx, "arm": arm,
                             "expansions": res2.expansions,
                             "n_injected": len(sel),
                             "accepted": bool(res2.candidates)})
    base = {"expansions": M.stats(r["expansions"] for r in rows
                                  if r["arm"] == "all_abstractions")["mean"]}
    comb = {"expansions": M.stats(r["expansions"] for r in rows
                                  if r["arm"] == "retrieval_ranked")["mean"]}
    return {"episodes": rows, "all_abstractions": base, "retrieval_ranked": comb,
            "deltas": [_regression(base, comb, "expansions", higher_better=False)]}


def _agree(body: Any, task: Task) -> float:
    ok = 0
    for ex in task.train:
        try:
            ok += int(dsl.evaluate(body, ex.x) == ex.y)
        except Exception:
            pass
    return ok / max(1, len(task.train))


# --- 3. concept drift + retrieval ------------------------------------------


def _drift_retrieval(cfg: ExpJConfig, seeds: tuple[int, ...]) -> dict[str, Any]:
    from . import drift as DR
    from constructive.exp_i.retrieval import ConceptStore

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        store = ConceptStore(eval_fn=dsl.evaluate)
        for rule in ("affine_mod", "threshold_parity", "sum_top_k"):
            src = G.make_task(rule, 0)
            tp = dict((c.id, c.program) for c in src.meta["candidates"])["true"]
            store.add(tp, input_type=src.input_type, output_type=src.output_type,
                      id=f"c-{rule}", context=(rule,))
        tasks, d_idx, _ = DR.make_drift_stream("affine_mod", seed, "abrupt",
                                               n_base=2, n_drift=3, n_recover=2)
        for t in tasks:
            obs = [(e.x, e.y) for e in t.train]
            r = store.retrieve(input_type=t.input_type, output_type=t.output_type,
                               observed_outputs=obs, top_k=1)
            rows.append({"seed": seed, "phase": t.meta["phase"],
                         "top": r.top, "expected": "c-affine_mod",
                         "hit": bool(r.top == "c-affine_mod"),
                         "ambiguous": bool(r.ambiguous)})
    base = {"recall@1": M.rate(r["hit"] for r in rows if r["phase"] == "base")["mean"]}
    comb = {"recall@1": M.rate(r["hit"] for r in rows if r["phase"] == "drift")["mean"]}
    return {"episodes": rows, "base": base, "drift": comb,
            "deltas": [_regression(base, comb, "recall@1")]}


# --- 4. noisy evidence + abstraction ---------------------------------------


def _noisy_abstraction(cfg: ExpJConfig, seeds: tuple[int, ...]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for level in (0.0, 0.2):
            for arm in ("no_abstraction", "with_abstraction"):
                task = NZ.apply_noise(
                    G.make_task("sum_top_k", seed * 7, n_val=cfg.n_val),
                    NoiseKind.RANDOM_LABEL, level, seed)
                bodies = []
                if arm == "with_abstraction":
                    src = G.make_task("sum_top_k", seed)
                    bodies = [dsl.call("evens", dsl.var("list"))]
                s = Searcher(beam=cfg.search_beam, max_cost=3.5,
                             max_depth=5, abstractions=bodies)
                res = s.search(task)
                task.meta["candidates"] = res.candidates
                vr, fields = run_validation(task, cfg, probe_pool=probe_pool_for(task))
                rows.append({"seed": seed, "level": level, "arm": arm,
                             "accepted": fields["accepted"],
                             "wrong_acceptance": fields["wrong_acceptance"],
                             "correct_acceptance": fields["correct_acceptance"],
                             "expansions": res.expansions})
    out: dict[str, Any] = {"episodes": rows}
    for level in (0.0, 0.2):
        b = {"wrong": M.rate(r["wrong_acceptance"] for r in rows
                             if r["level"] == level and r["arm"] == "no_abstraction")["mean"],
             "exp": M.stats(r["expansions"] for r in rows
                            if r["level"] == level and r["arm"] == "no_abstraction")["mean"]}
        c = {"wrong": M.rate(r["wrong_acceptance"] for r in rows
                             if r["level"] == level and r["arm"] == "with_abstraction")["mean"],
             "exp": M.stats(r["expansions"] for r in rows
                            if r["level"] == level and r["arm"] == "with_abstraction")["mean"]}
        out[f"level_{level}"] = {"no_abstraction": b, "with_abstraction": c,
                                "deltas": [_regression(b, c, "wrong", higher_better=False),
                                           _regression(b, c, "exp", higher_better=False)]}
    return out


# --- 5. conditional DSL + deep composition ---------------------------------


def _conditional_deep(cfg: ExpJConfig, seeds: tuple[int, ...]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        rng = random.Random(seed)
        k = rng.randint(0, 8)
        hidden = dsl.call("if", dsl.call("gt", dsl.var("int"), params=(k,)),
                          dsl.call("add1", dsl.var("int")),
                          dsl.call("double", dsl.var("int")))
        label = lambda x: dsl.evaluate(hidden, x)
        tx = G._uniq_ints(rng, 4, 0, 9)
        vx = G._uniq_ints(rng, 4, 10, 25, set(tx))
        ex = G._uniq_ints(rng, 12, 30, 80, set(tx + vx))
        task = Task(name=f"cond-s{seed}", family="conditional",
                    train=[Example(x, label(x)) for x in tx],
                    validation_pool=[Example(x, label(x)) for x in vx],
                    test=[Example(x, label(x)) for x in ex],
                    input_type="int", output_type="int",
                    meta={"hidden": hidden, "hidden_str": str(hidden), "depth": 3})
        for arm in ("conditional_enabled", "conditional_disabled"):
            s = Searcher(beam=cfg.search_beam, max_cost=4.5, max_depth=4)
            if arm == "conditional_disabled":
                orig = s._consider

                def blocked(child, probes, seen, nf, pool, target, pc, _o=orig):
                    if child.op == "if":
                        return None
                    return _o(child, probes, seen, nf, pool, target, pc)
                s._consider = blocked  # type: ignore[method-assign]
            res = s.search(task)
            task.meta["candidates"] = res.candidates
            vr, fields = run_validation(task, cfg, probe_pool=probe_pool_for(task))
            found = any(DC._equivalent(c.program, hidden,
                                       [e.x for e in task.test])
                        for c in res.candidates)
            rows.append({"seed": seed, "arm": arm, "found_correct": found,
                         "accepted": fields["accepted"],
                         "correct_acceptance": fields["correct_acceptance"],
                         "state": fields["state"], "expansions": res.expansions})
    b = {"correct_acceptance": M.rate(r["correct_acceptance"] for r in rows
                                      if r["arm"] == "conditional_disabled")["mean"]}
    c = {"correct_acceptance": M.rate(r["correct_acceptance"] for r in rows
                                      if r["arm"] == "conditional_enabled")["mean"]}
    return {"episodes": rows, "disabled": b, "enabled": c,
            "deltas": [_regression(b, c, "correct_acceptance")]}


# --- 6. program repair + adversarial evidence ------------------------------


def _repair_adversarial(cfg: ExpJConfig, seeds: tuple[int, ...]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    names = tuple(repair.TEMPLATES)
    for seed in seeds:
        for name in names:
            for arm, nv in (("limited_visible", 2), ("rich_visible", 8)):
                r = repair.repair_episode(name, seed + 100, n_visible=nv)
                rows.append({"seed": seed, "name": name, "arm": arm,
                             "status": r.get("status"),
                             "false_verified": bool(r.get("false_verified_repair")),
                             "false_detected": bool(r.get("false_repair_detected")),
                             "state": r.get("state"),
                             "coverage": r.get("verification_coverage")})
    out: dict[str, Any] = {"episodes": rows}
    for arm in ("limited_visible", "rich_visible"):
        sel = [r for r in rows if r["arm"] == arm]
        out[arm] = {
            "false_verified": M.rate(r["false_verified"] for r in sel),
            "false_detected": M.rate(r["false_detected"] for r in sel),
            "n": len(sel),
        }
    out["deltas"] = [_regression(
        {"false_verified": out["limited_visible"]["false_verified"]["mean"]},
        {"false_verified": out["rich_visible"]["false_verified"]["mean"]},
        "false_verified", higher_better=False)]
    return out


COMBINATIONS = {
    "evidence_validation+deep_search": _evidence_deep,
    "retrieval+abstraction": _retrieval_abstraction,
    "concept_drift+retrieval": _drift_retrieval,
    "noisy_evidence+abstraction": _noisy_abstraction,
    "conditional_dsl+deep_composition": _conditional_deep,
    "program_repair+adversarial_evidence": _repair_adversarial,
}


def run_interactions(cfg: ExpJConfig) -> dict[str, Any]:
    mon = Monitor()
    out = {"experiment_id": "J5", "experiment_name": "Cross-Mechanism Interaction",
           "status": "completed", "seeds": list(cfg.seeds), "combinations": {}}
    for name, fn in COMBINATIONS.items():
        try:
            out["combinations"][name] = fn(cfg, cfg.seeds)
        except Exception as exc:  # noqa: BLE001 - isolate, record, never fabricate
            out["combinations"][name] = {
                "status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
            mon.mark_error(f"{name}: {exc}")
    regressions = []
    for name, block in out["combinations"].items():
        for d in block.get("deltas", []):
            if d.get("regression"):
                regressions.append({"combination": name, **d})
    out["regressions"] = regressions
    out["n_regressions"] = len(regressions)
    out.update(mon.finish())
    return out
