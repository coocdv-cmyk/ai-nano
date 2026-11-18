"""H7 — Adversarial validation, identifiability and robustness."""

from __future__ import annotations

import time
from typing import Any

from .episode import baseline_episode, ccm_episode
from .generators import build_list_twin_task, build_twin_task
from ..model import ConstructiveModel
from .config import probe_hardware
from ..worlds import (
    Task,
    contradictory_task,
    math_task,
    noisy_task,
    relational_task,
)
from .metrics import mean_ci
from ..core import same_value


def _prime_blind_task(seed: int) -> Task:
    """Support contains ONLY odd composites: no evidence+ve about primes."""
    import random
    rng = random.Random(seed)
    odd_composites = [n for n in range(9, 400) if n % 2 and _not_prime(n)]
    rng.shuffle(odd_composites)
    train_vals = odd_composites[:4]
    val_val = odd_composites[4]
    train = [(x, False) for x in train_vals]
    val = [(val_val, False)]
    test = [(x, x in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29))
            for x in (2, 3, 4, 6, 7, 8, 9, 15, 21)]
    test += [(x, _is_prime(x)) for x in range(30, 60)]
    return Task("exp_h", f"h7-prime-blind-s{seed}", "is_prime", train, val,
                test, "int", "bool", [9, 15], {"adversarial": "blind-positives"})


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def _not_prime(n: int) -> bool:
    return not _is_prime(n)


def run_h7(cfg, *, seeds=None, time_budget_s: float = 4.0,
           max_nodes: int = 150_000) -> dict[str, Any]:
    seeds = tuple(seeds or cfg.seeds[:6])
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []

    def add(name, task, **kw):
        row = ccm_episode(task, seed=kw.pop("seed", 0),
                          max_nodes=max_nodes, time_budget_s=time_budget_s)
        row["sub_experiment"] = name
        rows.append(row)

    # (a) one corrupted label (10 shown, executed here too)
    for s in seeds:
        base = math_task("square", seed=8000 + s, k=4)
        task = noisy_task(base, seed=8500 + s, flip=1)
        add("corrupted-label", task, seed=s)

    # (b) distribution shift after validation (numeric + relation)
    for s in seeds:
        from .generators import build_ood_numeric_task, build_ood_scene_task
        add("shift-numeric",
            build_ood_numeric_task("is_prime", s + 900, k=3, lo=0, hi=9,
                                   test_hi=60, n_test=20),
            seed=s + 900)
        add("shift-structure",
            build_ood_scene_task("same_shape", s + 900, k=3,
                                 train_size=(4, 6), test_size=(9, 12),
                                 n_test=20),
            seed=s + 900)

    # (c) data-order sensitivity: accepted concept string across permutations
    order_rows = []
    for s in seeds:
        task = relational_task("same_shape", seed=100 + s, k=4)
        import random as _random
        for shuffle in range(4):
            rng = _random.Random(s * 100 + shuffle)
            shuffled = list(task.train)
            rng.shuffle(shuffled)
            m = ConstructiveModel(max_cost=3, beam=45, max_nodes=max_nodes,
                                time_budget_s=time_budget_s)
            rep = m.fit(shuffled, task.input_type, task.output_type,
                        validation=task.val, probes=task.probes)
            order_rows.append({
                "seed": s, "shuffle": shuffle, "accepted": rep.accepted,
                "accepted_program": rep.candidates[0] if rep.candidates else None,
                "abstained": rep.abstained,
            })
    # (d) near-duplicate examples: duplicate one example twice
    for s in seeds:
        task = math_task("square", seed=1100 + s, k=3)
        dup = list(task.train) + [task.train[0]]
        dup_task = Task("exp_h", f"h7-near-dup-s{s}", task.rule, dup, task.val,
                        task.test, task.input_type, task.output_type,
                        task.probes, task.meta)
        add("near-duplicate-examples", dup_task, seed=s)

    # (e) contradiction: half examples follow a different rule
    for s in seeds:
        task = contradictory_task("square", "double", seed=2100 + s, k=4)
        add("contradiction", task, seed=s)

    # (f) adversarial counterexample: exhaustively check accepted numeric
    # concepts against a bounded oracle (0..60)
    exhaustive_rows = []
    from .generators import build_ood_numeric_task
    for rule in ("square", "digital_root", "is_prime"):
        for s in seeds:
            try:
                task = build_ood_numeric_task(rule, s + 300, k=3, lo=0, hi=9,
                                              test_hi=60, n_test=12)
            except ValueError:
                continue
            m = ConstructiveModel(max_cost=3, beam=45, max_nodes=max_nodes,
                                  time_budget_s=time_budget_s)
            rep = m.fit(task.train, task.input_type, task.output_type,
                        validation=task.val, probes=task.probes)
            false_accept = 0
            if rep.accepted:
                for x in range(0, 61):
                    try:
                        y = _oracle(rule, x)
                    except Exception:
                        continue
                    yp = m.predict_episode(task.train, x, task.input_type)
                    if yp is None or not same_value(yp, y):
                        false_accept += 1
            exhaustive_rows.append({
                "rule": rule, "seed": s, "accepted": rep.accepted,
                "prediction_misses_in_bounded_domain": false_accept if rep.accepted else None,
                "domain_size": 61,
            })

    # (g) twin ambiguity at low support (k=1,2,3,5)
    twin_rows = []
    for k in (1, 2, 3, 5):
        for s in seeds:
            for builder in (build_twin_task, build_list_twin_task):
                try:
                    task = builder(s, k)
                except ValueError:
                    continue
                row = ccm_episode(task, seed=s, max_nodes=max_nodes,
                                  time_budget_s=time_budget_s)
                row["k_small"] = k
                twin_rows.append(row)

    # (h) is_prime with blind positives
    blind_rows = []
    for s in seeds:
        task = _prime_blind_task(s)
        row = ccm_episode(task, seed=s, max_nodes=max_nodes,
                          time_budget_s=time_budget_s)
        blind_rows.append(row)

    sub_summary: dict[str, Any] = {}
    for name in sorted({r["sub_experiment"] for r in rows}):
        sel = [r for r in rows if r["sub_experiment"] == name]
        sub_summary[name] = {
            "n": len(sel),
            "accepted_rate": _mean(r["accepted"] for r in sel),
            "accuracy_mean": mean_ci(r["test_accuracy"] for r in sel),
            "coverage_mean": _mean(r["coverage"] for r in sel),
            "false_accepts_estimate": sum(
                1 for r in sel if r["accepted"] and r["test_accuracy"] < 1.0),
            "failure_categories": _counts(r["failure_category"] for r in sel),
        }
    return {
        "experiment_id": "H7",
        "experiment_name": "Adversarial Validation, Identifiability and Robustness",
        "status": "completed",
        "episodes": rows,
        "summary": sub_summary,
        "order_sensitivity": order_rows,
        "twin_small_k": twin_rows,
        "bounded_domain_checks": exhaustive_rows,
        "blind_positive_is_prime": blind_rows,
        "sub_summary": sub_summary,
        "hardware": probe_hardware().as_dict(),
        "wall_s": time.perf_counter() - t0,
    }


def _oracle(rule: str, x: int) -> int:
    if rule == "square":
        return x * x
    if rule == "digital_root":
        return 0 if x == 0 else 1 + (abs(x) - 1) % 9
    if rule == "is_prime":
        return _is_prime(x)
    raise ValueError(rule)


def _mean(xs) -> float | None:
    xs = list(xs)
    return round(sum(xs) / len(xs), 6) if xs else None


def _counts(items) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out
