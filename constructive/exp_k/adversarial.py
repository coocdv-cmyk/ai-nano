"""K6 — Adversarial reliability campaign.

Twelve attacks probe safety and useful coverage. An "always abstain" policy is
included explicitly: refusing every difficult task scores zero useful coverage
and is never counted as success.
"""

from __future__ import annotations

import random
from typing import Any

from constructive.exp_i import dsl
from constructive.exp_i.resources import ResourceGovernor
from constructive.exp_i.search import Searcher
from constructive.exp_i.types import Candidate, Example, Task, ValidationState

from . import constructors as C
from . import generators as G
from . import metrics as M
from .config import ExpKConfig
from .engine import run_validation
from .resources import Monitor

ATTACKS = (
    "misleading_positive", "correlated_support", "missing_counterexamples",
    "near_identical_rules", "incorrect_intermediate", "noisy_observations",
    "contradictory_observations", "unseen_families",
    "misleading_retrieval_neighbors", "abrupt_distribution_change",
    "deep_deceptive", "resource_exhaustion",
)


def _misleading(task):
    wrong = task.meta["candidates"][1].program
    train = []
    for e in task.train:
        try:
            train.append(Example(e.x, dsl.evaluate(wrong, e.x)))
        except Exception:
            train.append(e)
    return _repack(task, train, task.validation_pool)


def _magnitude(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return abs(v)
    if isinstance(v, (list, tuple)):
        return sum(_magnitude(x) for x in v)
    return 0


def _correlated(task):
    xs = sorted([e.x for e in task.train], key=_magnitude)[:2]
    train = [e for e in task.train if e.x in xs]
    return _repack(task, train, task.validation_pool)


def _missing_cex(task):
    wrong = task.meta["candidates"][1].program
    val = []
    for e in task.validation_pool:
        try:
            if dsl.evaluate(wrong, e.x) == e.y:
                val.append(e)
        except Exception:
            pass
    return _repack(task, task.train, val or task.validation_pool)


def _noisy(task, seed):
    rng = random.Random(seed)
    train = [Example(e.x, e.y + 7 if isinstance(e.y, int)
                     and not isinstance(e.y, bool) and rng.random() < 0.3 else e.y)
             for e in task.train]
    return _repack(task, train, task.validation_pool)


def _contradictory(task):
    e0 = task.train[0]
    bad = e0.y + 999 if isinstance(e0.y, int) and not isinstance(e0.y, bool) else not e0.y
    return _repack(task, list(task.train) + [Example(e0.x, bad)], task.validation_pool)


def _repack(task, train, val):
    return Task(name=task.name + "-adv", family=task.family, train=train,
                validation_pool=val, test=list(task.test),
                input_type=task.input_type, output_type=task.output_type,
                meta=dict(task.meta))


def _build(attack: str, seed: int, cfg) -> tuple[Task, Any, dict[str, Any]]:
    rng = random.Random(seed * 313)
    if attack in ("misleading_positive", "correlated_support",
                  "missing_counterexamples", "noisy_observations",
                  "contradictory_observations"):
        rule = G.ALL_RULES[seed % len(G.ALL_RULES)]
        base = G.make_task(rule, seed, n_val=cfg.n_val, n_test=cfg.n_test)
        builders = {"misleading_positive": lambda t: _misleading(t),
                    "correlated_support": lambda t: _correlated(t),
                    "missing_counterexamples": lambda t: _missing_cex(t),
                    "noisy_observations": lambda t: _noisy(t, seed),
                    "contradictory_observations": lambda t: _contradictory(t)}
        return builders[attack](base), None, {"rule": rule}
    if attack == "near_identical_rules":
        rule = G.ALL_RULES[6 + (seed % 6)]  # held-out family
        t = G.make_task(rule, seed, n_val=cfg.n_val)
        return t, None, {"rule": rule, "ood": True}
    if attack == "incorrect_intermediate":
        depth = 8
        t, hidden = C.deep_chain_task(seed, depth)
        return t, hidden, {"depth": depth}
    if attack == "unseen_families":
        rule = G.ALL_RULES[12 + (seed % 4)]
        t = G.make_task(rule, seed, shift="magnitude", n_val=cfg.n_val)
        return t, None, {"rule": rule, "ood": True}
    if attack == "misleading_retrieval_neighbors":
        store = C.concept_pool(G.ALL_RULES[:6])
        rule = G.ALL_RULES[12 + (seed % 4)]
        t = G.make_task(rule, seed, n_val=cfg.n_val)
        obs = [(e.x, e.y) for e in t.train + t.validation_pool]
        res = store.retrieve(input_type=t.input_type, output_type=t.output_type,
                             observed_outputs=obs, top_k=1)
        return t, None, {"retrieval_top": res.top, "ood": True}
    if attack == "abrupt_distribution_change":
        rule = G.ALL_RULES[seed % 4]
        tasks, d, _r = C.drift_stream(rule, seed, "abrupt", n_val=cfg.n_val)
        return tasks[d + 1], None, {"rule": rule, "drifted": True}
    if attack == "deep_deceptive":
        t, hidden = C.deep_chain_task(seed, 12)
        return t, hidden, {"depth": 12}
    if attack == "resource_exhaustion":
        t, hidden = C.deep_chain_task(seed, 12)
        return t, hidden, {"depth": 12, "tiny_budget": True}
    raise ValueError(attack)


def _outcome(fields, meta) -> str:
    state = fields["state"]
    if state == ValidationState.NON_IDENTIFIABLE.value:
        return "non_identifiable"
    if state == ValidationState.CONTRADICTORY.value:
        return "correct_rejection" if meta.get("expected_contradiction") else "incorrect_rejection"
    if fields["wrong"]:
        return "incorrect_acceptance"
    if fields["correct"]:
        return "correct_acceptance"
    if fields["abstained"]:
        return ("incorrect_abstention" if fields["false_rejection"]
                else "correct_abstention")
    return "other"


def run_k6(cfg: ExpKConfig, *, seeds=None) -> dict[str, Any]:
    seeds = tuple(seeds if seeds is not None else cfg.seeds)
    mon = Monitor()
    rows: list[dict[str, Any]] = []
    for attack in ATTACKS:
        for seed in seeds:
            task, hidden, meta = _build(attack, seed, cfg)
            if attack in ("incorrect_intermediate", "deep_deceptive",
                          "resource_exhaustion"):
                budget = (300 if attack == "resource_exhaustion"
                          else cfg.search_node_budget)
                gov = ResourceGovernor(global_deadline_s=cfg.fit_deadline_s,
                                       node_budget=budget)
                depth = task.meta["depth"]
                s = Searcher(beam=cfg.search_beam,
                             max_cost=min(cfg.max_cost, depth + 1.0),
                             max_depth=depth, governor=gov)
                res = s.search(task, node_budget=budget,
                               deadline_s=cfg.fit_deadline_s)
                task.meta["candidates"] = res.candidates
                meta["expansions"] = res.expansions
                meta["timed_out"] = bool(res.timed_out)
            meta["expected_contradiction"] = attack == "contradictory_observations"
            for policy in ("system", "always_abstain"):
                if policy == "system":
                    _r, fields = run_validation(task, cfg)
                else:
                    fields = {"state": "ALWAYS_ABSTAIN", "accepted": False,
                              "correct": False, "wrong": False, "abstained": True,
                              "false_rejection": False, "confidence": 0.0,
                              "test_accuracy": None, "coverage": 0.0}
                rows.append({
                    "attack": attack, "seed": seed, "policy": policy,
                    "outcome": _outcome(fields, meta), **fields,
                    "meta": meta,
                })
                mon.charge(validation_calls=len(task.validation_pool),
                           candidates=len(task.meta.get("candidates", [])))
    by_attack: dict[str, Any] = {}
    for attack in ATTACKS:
        sel = [r for r in rows if r["attack"] == attack and r["policy"] == "system"]
        abst = [r for r in rows if r["attack"] == attack and r["policy"] == "always_abstain"]
        by_attack[attack] = {
            "n": len(sel), "confusion": M.confusion(sel),
            "incorrect_acceptance": M.rate(r["wrong"] for r in sel),
            "correct_acceptance": M.rate(r["correct"] for r in sel),
            "correct_abstention": M.rate(
                r["outcome"] == "correct_abstention" for r in sel),
            "incorrect_abstention": M.rate(
                r["outcome"] == "incorrect_abstention" for r in sel),
            "always_abstain_useful_coverage": M.rate(r["correct"] for r in abst),
        }
    out = {
        "stage": "K6", "experiment_id": "K6",
        "experiment_name": "Adversarial Reliability Campaign",
        "status": "completed", "seeds": list(seeds),
        "attacks": list(ATTACKS), "n_runs": len(rows), "episodes": rows,
        "by_attack": by_attack,
        "useful_coverage": M.rate(r["correct"] for r in rows
                                  if r["policy"] == "system"),
        "always_abstain_useful_coverage": M.rate(r["correct"] for r in rows
                                                 if r["policy"] == "always_abstain"),
        "no_label_leakage": ("Test labels are used only after decisions; generators "
                            "and retrieval stores never see the evaluation split."),
    }
    out.update(mon.finish())
    return out
