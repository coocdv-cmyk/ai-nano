"""K5 — Cross-mechanism interaction analysis.

Each named pair is evaluated in a 2x2 factorial design (neither, A only,
B only, both) on matched episodes. Additive interaction effects and regressions
are reported; an improvement in one metric is never treated as an overall win.
"""

from __future__ import annotations

from typing import Any

from constructive.exp_i import dsl
from constructive.exp_i.resources import ResourceGovernor
from constructive.exp_i.search import Searcher, cegis_refine
from constructive.exp_i.types import Candidate, Example, Task

from . import calibration as K4
from . import constructors as C
from . import drift_detection as K3
from . import generators as G
from . import metrics as M
from . import retrieval_ood as K2
from .config import ExpKConfig
from .engine import run_validation
from .resources import Monitor

# (name, mechanism A, mechanism B, episode source)
COMBINATIONS = (
    ("evidence_validation+deep_search", "evidence", "deep_search", "deep"),
    ("retrieval+deep_search", "retrieval", "deep_search", "deep"),
    ("retrieval+ood_abstention", "retrieval", "ood_abstention", "ood"),
    ("drift_detection+retrieval", "drift_detection", "retrieval", "drift"),
    ("drift_detection+calibration", "drift_detection", "calibration", "drift"),
    ("noisy_evidence+deep_search", "noisy_evidence", "deep_search", "deep"),
    ("contradictory_evidence+retrieval", "contradictory_evidence", "retrieval", "ood"),
    ("calibration+abstention", "calibration", "abstention", "normal"),
    ("type_checking+composition", "type_checking", "deep_search", "deep"),
    ("cegis+resource_governance", "cegis", "resource_governance", "deep"),
)


def _episodes(source: str, seeds: tuple[int, ...], cfg) -> list[dict[str, Any]]:
    out = []
    for seed in seeds:
        if source == "deep":
            for depth in (2, 5, 12):
                t, hidden = C.deep_chain_task(seed, depth)
                out.append({"kind": "task", "task": t, "hidden": hidden,
                            "seed": seed, "label": f"d{depth}"})
        elif source == "normal":
            for rule in G.ALL_RULES[:4]:
                t = G.make_task(rule, seed * 7 + 1, n_val=cfg.n_val)
                out.append({"kind": "task", "task": t, "hidden": None,
                            "seed": seed, "label": rule})
        elif source == "ood":
            for rule in G.ALL_RULES[12:]:
                t = G.make_task(rule, seed * 11 + 2, n_val=cfg.n_val)
                out.append({"kind": "task", "task": t, "hidden": None,
                            "seed": seed, "label": rule})
        else:  # drift
            for rule in G.ALL_RULES[:2]:
                tasks, d, r = C.drift_stream(rule, seed * 13, "abrupt",
                                             n_val=cfg.n_val)
                for t in tasks[d:d + 2]:
                    out.append({"kind": "task", "task": t, "hidden": None,
                                "seed": seed, "label": f"{rule}-{t.meta['phase']}"})
    return out


def _noise(task, kind):
    if kind == "none":
        return task
    train = list(task.train)
    if kind == "noisy_evidence":
        train = [Example(e.x, e.y + 5 if isinstance(e.y, int)
                         and not isinstance(e.y, bool) else e.y) for e in train]
    elif kind == "contradictory_evidence":
        ex = train[0]
        train = train + [Example(ex.x, ex.y + 999 if isinstance(ex.y, int)
                                 else not ex.y)]
    return Task(name=task.name + "-" + kind, family=task.family, train=train,
                validation_pool=list(task.validation_pool), test=list(task.test),
                input_type=task.input_type, output_type=task.output_type,
                meta=dict(task.meta))


def _candidates(ep, flags, cfg):
    task = ep["task"]
    if "deep_search" not in flags and "cegis" not in flags:
        cands = list(task.meta.get("candidates", []))
        if not cands and task.output_type == "int" and task.input_type == "int":
            # Fixed shallow hypothesis set for tasks without a built-in pool.
            x = dsl.var("int")
            cands = [Candidate(id=f"fixed{i}", program=p, cost=1.0)
                     for i, p in enumerate((x, dsl.call("add1", x),
                                            dsl.call("double", x),
                                            dsl.call("square", x)))]
        return cands, 0, False
    gov = ResourceGovernor(
        global_deadline_s=(1.0 if "resource_governance" in flags else cfg.fit_deadline_s),
        node_budget=(2000 if "resource_governance" in flags else cfg.search_node_budget))
    depth = task.meta.get("depth", 4)
    s = Searcher(beam=cfg.search_beam, max_cost=min(cfg.max_cost, depth + 1.0),
                 max_depth=max(depth, 4), governor=gov)
    kw = {"node_budget": gov.node_budget, "deadline_s": gov.global_deadline_s}
    if "cegis" in flags and ep.get("hidden") is not None:
        def witness(c):
            for ex in task.validation_pool:
                try:
                    if dsl.evaluate(c.program, ex.x) != ex.y:
                        return ex
                except Exception:
                    return ex
            return None
        res, _ = cegis_refine(task, s, witness, rounds=1, **kw)
    else:
        res = s.search(task, **kw)
    return res.candidates, res.expansions, bool(res.timed_out)


def _evaluate(ep, flags, cfg):
    task = _noise(ep["task"], "noisy_evidence" if "noisy_evidence" in flags else
                  "contradictory_evidence" if "contradictory_evidence" in flags else "none")
    cands, expansions, timed_out = _candidates({**ep, "task": task}, flags, cfg)
    if "type_checking" in flags:
        cands = [c for c in cands if not dsl.validate(c.program,
                                                       max_depth=cfg.max_cost * 4,
                                                       max_nodes=64)]
    task.meta["candidates"] = cands
    if "evidence" in flags:
        _r, f = run_validation(task, cfg)
        accepted, correct, wrong = f["accepted"], f["correct"], f["wrong"]
        conf = f["confidence"]
        coverage = f["coverage"]
    else:
        accepted = bool(cands)
        prog = cands[0].program if cands else None
        correct = bool(accepted and ep.get("hidden") is not None and
                       all(dsl.evaluate(prog, e.x) == dsl.evaluate(ep["hidden"], e.x)
                           for e in task.test))
        wrong = bool(accepted and not correct)
        conf = 0.9 if accepted else 0.0
        coverage = 1.0 if accepted else 0.0
    if "calibration" in flags:
        conf = K4._sigmoid(K4._logit(conf) / 0.75)
    if "abstention" in flags and conf < 0.55:
        accepted, correct, wrong = False, False, False
    if "ood_abstention" in flags and ep.get("hidden") is None:
        # For OOD episodes, abstain unless support-consistent and confident.
        if conf < 0.8:
            accepted, correct, wrong = False, False, False
    return {"accepted": accepted, "correct": correct, "wrong": wrong,
            "abstained": not accepted, "confidence": conf,
            "coverage": coverage, "expansions": expansions,
            "timed_out": timed_out}


def _cell(eps, flags, cfg):
    rows = [_evaluate(ep, flags, cfg) for ep in eps]
    return {
        "n": len(rows),
        "correct": M.rate(r["correct"] for r in rows),
        "wrong": M.rate(r["wrong"] for r in rows),
        "coverage": M.rate(r["accepted"] for r in rows),
        "abstention": M.rate(r["abstained"] for r in rows),
        "expansions": M.stats(r["expansions"] for r in rows),
        "timed_out": sum(1 for r in rows if r["timed_out"]),
    }


def run_k5(cfg: ExpKConfig, *, seeds=None,
           combinations=COMBINATIONS) -> dict[str, Any]:
    seeds = tuple(seeds if seeds is not None else cfg.seeds)
    mon = Monitor()
    results: dict[str, Any] = {}
    for name, a, b, source in combinations:
        eps = _episodes(source, seeds, cfg)
        cells = {
            "baseline": _cell(eps, set(), cfg),
            f"A_only[{a}]": _cell(eps, {a}, cfg),
            f"B_only[{b}]": _cell(eps, {b}, cfg),
            f"A_and_B": _cell(eps, {a, b}, cfg),
        }
        metrics = ("correct", "wrong", "coverage", "abstention", "expansions")
        interactions = {}
        for metric in metrics:
            values = [cells[k][metric]["mean"] for k in cells]
            interactions[metric] = round(values[3] - values[1] - values[2] + values[0], 6)
        regressions = []
        for metric, better in (("correct", "higher"), ("wrong", "lower"),
                               ("coverage", "higher"), ("expansions", "lower")):
            both = cells["A_and_B"][metric]["mean"]
            best_single = min(cells[f"A_only[{a}]"][metric]["mean"],
                              cells[f"B_only[{b}]"][metric]["mean"]) if better == "lower" \
                else max(cells[f"A_only[{a}]"][metric]["mean"],
                         cells[f"B_only[{b}]"][metric]["mean"])
            if both is not None and best_single is not None:
                worse = both < best_single if better == "higher" else both > best_single
                if worse:
                    regressions.append({"metric": metric, "combined": both,
                                        "best_single": best_single})
        results[name] = {"mechanism_a": a, "mechanism_b": b, "source": source,
                         "n_episodes": len(eps), "cells": cells,
                         "interaction_effects": interactions,
                         "regressions": regressions}
    out = {
        "stage": "K5", "experiment_id": "K5",
        "experiment_name": "Cross-Mechanism Interactions",
        "status": "completed", "seeds": list(seeds),
        "n_combinations": len(combinations),
        "combinations": results,
        "total_regressions": sum(len(v["regressions"]) for v in results.values()),
        "limitations": [
            "Factorial interaction is estimated additively; no significance test is applied to the interaction term.",
            "Some mechanisms are only meaningful for the episode source where they are applied.",
        ],
    }
    out.update(mon.finish())
    return out
