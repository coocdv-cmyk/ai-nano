"""Benchmark harness for the constructive-learning experiments (A-F).

Everything here measures; nothing here fabricates.  Each episode reports the
system's test accuracy, its coverage (how often it answers instead of
abstaining), fit wall-clock time, search nodes, and peak traced memory.  The
JSON written by ``run_all`` is the source of every number in the report.

Definitions used throughout
---------------------------
accuracy            correct / all test items (an abstention counts as wrong)
coverage            answered / all test items
selective accuracy  correct / answered
examples-to-competence  smallest training size in the grid whose *mean* test
                    accuracy over seeds reaches 0.8
speedup             only computed for a task where both systems reach 0.8, and
                    reported as an examples ratio and a CPU-time ratio
"""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .baselines import MLP, Memorizer, NearestNeighbor, NotSupported, make_baseline
from .core import same_value
from .model import ConstructiveModel
from . import worlds as W

RESULTS_DIR = Path(__file__).resolve().parent / "results"
THRESHOLD = 0.8
CCM_KW: dict[str, Any] = {"max_cost": 3, "beam": 45, "max_nodes": 600_000, "time_budget_s": 10.0}


# ---------------------------------------------------------------------------
# Measurement primitives
# ---------------------------------------------------------------------------


def evaluate(predict: Callable[[Any], Any], task: W.Task, support: list[tuple[Any, Any]] | None = None,
             model: Any = None) -> dict[str, Any]:
    correct = answered = 0
    tol_correct = 0
    errors: list[float] = []
    for x, y in task.test:
        if support is not None:
            v = model.predict_episode(support, x, task.input_type)
        else:
            v = predict(x)
        numeric = isinstance(y, int) and not isinstance(y, bool)
        if v is not None:
            answered += 1
            if same_value(v, y):
                correct += 1
            if numeric:
                errors.append(abs(float(v) - float(y)))
                if abs(float(v) - float(y)) <= max(1.0, 0.01 * abs(float(y))):
                    tol_correct += 1
            elif same_value(v, y):
                tol_correct += 1
    n = len(task.test)
    return {
        "n": n,
        "correct": correct,
        "answered": answered,
        "accuracy": correct / n if n else 0.0,
        "coverage": answered / n if n else 0.0,
        "selective_accuracy": correct / answered if answered else 0.0,
        # secondary metrics: a continuous baseline is not handicapped by
        # having to land on the exact integer, and numeric error is visible
        "accuracy_tol": tol_correct / n if n else 0.0,
        "mae": (sum(errors) / len(errors)) if errors else None,
    }


def timed(fn: Callable[[], Any]) -> tuple[Any, float, int]:
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return out, dt, int(peak)


def ccm_episode(task: W.Task, model: ConstructiveModel | None = None,
                seed: int | None = None, **kw: Any) -> dict[str, Any]:
    m = model or ConstructiveModel(**{**CCM_KW, **kw})
    before = m.n_stored_nodes()
    rep, dt, peak = timed(lambda: m.fit(task.train, task.input_type, task.output_type,
                                        validation=task.val, probes=task.probes))
    result = evaluate(None, task, support=task.train, model=m)
    return {
        "system": "ccm",
        "task": task.name,
        "seed": seed,
        "family": task.family,
        "rule": task.rule,
        "k": len(task.train),
        "accepted": rep.accepted,
        "abstained": rep.abstained,
        "reason": rep.reason,
        "quarantined": rep.quarantined,
        "fit_s": dt,
        "nodes": rep.nodes,
        "peak_bytes": peak,
        "concept_cost": rep.cost,
        "stored_nodes": m.n_stored_nodes() - before,
        "confidence": rep.confidence,
        "margin": rep.margin,
        "validation_accuracy": rep.validation_accuracy,
        "candidates": rep.candidates[:3],
        **{f"test_{k}": v for k, v in result.items()},
    }


def baseline_episode(name: str, task: W.Task, seed: int | None = None,
                    **kw: Any) -> dict[str, Any]:
    b = make_baseline(name, **kw)
    # Fair label budget: CCM sees `train` for induction and one held-out
    # example for acceptance, so baselines receive both as training data.
    data = list(task.train) + list(task.val)
    try:
        _, dt, peak = timed(lambda: b.fit(data, task.input_type, task.output_type,
                                          validation=task.val))
        result = evaluate(b.predict, task)
        return {"system": name, "task": task.name, "rule": task.rule,
                "k": len(task.train), "labels": len(data), "seed": seed, "fit_s": dt,
                "peak_bytes": peak, **{f"test_{k}": v for k, v in result.items()}}
    except NotSupported as exc:
        return {"system": name, "task": task.name, "rule": task.rule, "error": str(exc)}


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if "test_accuracy" in r]
    if not ok:
        return {"n": 0}
    accs = [r["test_accuracy"] for r in ok]
    nodes = [r["nodes"] for r in ok if "nodes" in r]
    mae = [r["test_mae"] for r in ok if r.get("test_mae") is not None]
    return {
        "n": len(ok),
        "accuracy_mean": statistics.fmean(accs),
        "accuracy_min": min(accs),
        "accuracy_max": max(accs),
        "accuracy_tol_mean": statistics.fmean(
            r.get("test_accuracy_tol", r["test_accuracy"]) for r in ok),
        "mae_median": statistics.median(mae) if mae else None,
        "coverage_mean": statistics.fmean(r["test_coverage"] for r in ok),
        "fit_s_median": statistics.median(r["fit_s"] for r in ok),
        "fit_s_total": sum(r["fit_s"] for r in ok),
        "nodes_median": statistics.median(nodes) if nodes else None,
        "peak_bytes_median": statistics.median(r["peak_bytes"] for r in ok),
    }


# ---------------------------------------------------------------------------
# Experiment A - relational concepts
# ---------------------------------------------------------------------------

REL_A = ("left_of", "right_of", "above", "below", "same_color", "same_shape",
         "nearest", "biggest", "smallest")


def experiment_a(seeds: list[int] = (0, 1, 2), k: int = 4) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    for rule in REL_A:
        for s in seeds:
            task = W.relational_task(rule, seed=1000 + s, k=k)
            episodes.append(ccm_episode(task, seed=s))
            for b in ("memorizer", "nearest_neighbor", "mlp"):
                episodes.append(baseline_episode(b, task, seed=s))
    return {"episodes": episodes,
            "summary": _by_system(episodes),
            "by_rule": _by_rule(episodes)}


def experiment_a_curve(seeds: list[int] = (0, 1), k_values: list[int] | None = None) -> dict[str, Any]:
    k_values = k_values or [1, 2, 3, 4, 6, 8, 12, 20, 40, 80]
    rows: list[dict[str, Any]] = []
    for rule in ("left_of", "same_shape"):
        for k in k_values:
            for s in seeds:
                task = W.relational_task(rule, seed=2000 + s, k=k)
                rows.append(ccm_episode(task))
                rows.append(baseline_episode("mlp", task))
                rows.append(baseline_episode("nearest_neighbor", task))
    return {"rows": rows, "curve": _curve(rows)}


# ---------------------------------------------------------------------------
# Experiment B - rule discovery
# ---------------------------------------------------------------------------


def experiment_b(seeds: list[int] = (0, 1, 2)) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    prims = ("sum_list", "max_list", "range_list", "count_even", "sum_squares", "head")
    derived = ("sum_of_evens", "distinct_count", "sum_top3")
    for rule in prims:
        for s in seeds:
            task = W.list_rule_task(rule, seed=3000 + s, k=3)
            episodes.append(ccm_episode(task, seed=s))
            for b in ("nearest_neighbor", "mlp"):
                episodes.append(baseline_episode(b, task, seed=s))
    for rule in derived:
        for s in seeds:
            task = W.list_rule_task(rule, seed=3500 + s, k=4)
            episodes.append(ccm_episode(task, seed=s))
            for b in ("nearest_neighbor", "mlp"):
                episodes.append(baseline_episode(b, task, seed=s))
    return {"episodes": episodes, "summary": _by_system(episodes), "by_rule": _by_rule(episodes)}


def experiment_b_curve(seeds: list[int] = (0, 1)) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for rule in ("sum_list", "sum_of_evens"):
        for k in (2, 3, 4, 6, 10, 16, 24):
            for s in seeds:
                task = W.list_rule_task(rule, seed=4000 + s, k=k, n_test=12)
                rows.append(ccm_episode(task))
                rows.append(baseline_episode("mlp", task))
                rows.append(baseline_episode("nearest_neighbor", task))
                rows.append(baseline_episode("memorizer", task))
    return {"rows": rows, "curve": _curve(rows)}


# ---------------------------------------------------------------------------
# Experiment C - compositional learning
# ---------------------------------------------------------------------------


def composition_episode(seed: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    m = ConstructiveModel(**CCM_KW)
    a = W.concept_a_task(seed)
    b = W.concept_b_task(seed)
    for tag, task in (("A", a), ("B", b)):
        rep, dt, peak = timed(lambda t=task: m.fit(t.train, t.input_type, t.output_type,
                                                   validation=t.val, probes=t.probes))
        ev = evaluate(None, task, support=task.train, model=m)
        out.append({"system": "ccm", "part": "learn-" + tag, "task": task.name,
                    "accepted": rep.accepted, "fit_s": dt, "peak_bytes": peak,
                    "nodes": rep.nodes, "stored_nodes": m.n_stored_nodes(),
                    "candidates": rep.candidates[:2],
                    **{f"test_{k}": v for k, v in ev.items()}})

    target = W.composition_task(seed)
    comp, dt, peak = timed(lambda: m.compose(target.train, target.input_type, target.output_type,
                                             validation=target.val, probes=target.probes,
                                             max_cost=2))
    ev = evaluate(None, target, support=target.train, model=m)
    out.append({"system": "ccm", "part": "compose", "task": target.name,
                "accepted": comp.accepted, "term": comp.term, "fit_s": dt,
                "peak_bytes": peak, "nodes": comp.nodes,
                "stored_nodes": m.n_stored_nodes(), "candidates": comp.candidates[:2],
                **{f"test_{k}": v for k, v in ev.items()}})

    # control 1: same small budget, no learned concepts
    fresh = ConstructiveModel(**CCM_KW)
    c1, dt1, _ = timed(lambda: fresh.compose(target.train, target.input_type, target.output_type,
                                             validation=target.val, probes=target.probes,
                                             max_cost=2))
    ev1 = evaluate(None, target, support=target.train, model=fresh)
    out.append({"system": "no-concepts", "part": "compose", "task": target.name,
                "accepted": c1.accepted, "fit_s": dt1, "nodes": c1.nodes,
                **{f"test_{k}": v for k, v in ev1.items()}})

    # control 2: flat search with one extra cost unit, no concepts
    fresh2 = ConstructiveModel(**{**CCM_KW, "max_cost": 4})
    r2, dt2, _ = timed(lambda: fresh2.fit(target.train, target.input_type, target.output_type,
                                          validation=target.val, probes=target.probes))
    ev2 = evaluate(None, target, support=target.train, model=fresh2)
    out.append({"system": "flat-search-cost4", "part": "fit", "task": target.name,
                "accepted": r2.accepted, "timed_out": r2.timed_out, "fit_s": dt2, "nodes": r2.nodes,
                "candidates": r2.candidates[:2], "stored_nodes": fresh2.n_stored_nodes(),
                **{f"test_{k}": v for k, v in ev2.items()}})
    return out


def experiment_c(seeds: list[int] = (0, 1, 2, 3)) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for s in seeds:
        rows.extend(composition_episode(s))
    return {"rows": rows,
            "summary": {f"{sys}/{part}": _summ([r for r in rows
                                                if r.get("part") == part and r["system"] == sys])
                        for sys in ("ccm", "no-concepts", "flat-search-cost4")
                        for part in ("learn-A", "learn-B", "compose", "fit")}}


# ---------------------------------------------------------------------------
# Experiment D - mathematical structure
# ---------------------------------------------------------------------------

# ``double`` is omitted: it is behaviourally identical to the primitive ``mul(2)``
# and no data can distinguish them, so it is not an identifiable rule.
MATH_D = ("square", "sum_digits", "digital_root", "fib", "is_prime",
          "even", "add(3)", "mul(3)")


def experiment_d(seeds: list[int] = (0, 1, 2)) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    for rule in MATH_D:
        for s in seeds:
            task = W.math_task(rule, seed=5000 + s, k=3)
            episodes.append(ccm_episode(task, seed=s))
            for b in ("memorizer", "nearest_neighbor", "mlp"):
                episodes.append(baseline_episode(b, task, seed=s))
    return {"episodes": episodes, "summary": _by_system(episodes), "by_rule": _by_rule(episodes)}


def experiment_d_pairs(seeds: list[int] = (0, 1)) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    for rule in ("sum_pair", "diff_pair", "gcd_pair", "max_pair"):
        for s in seeds:
            task = W.math_task(rule, seed=5500 + s, k=3, pair=True)
            episodes.append(ccm_episode(task, seed=s))
            episodes.append(baseline_episode("mlp", task, seed=s))
            episodes.append(baseline_episode("nearest_neighbor", task, seed=s))
    return {"episodes": episodes, "summary": _by_system(episodes)}


# ---------------------------------------------------------------------------
# Experiment E - continual learning
# ---------------------------------------------------------------------------

CONTINUAL_RULES = ("left_of", "same_shape", "above", "nearest", "biggest")


def experiment_e(seeds: list[int] = (0, 1, 2)) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for s in seeds:
        tasks = [W.relational_task(r, seed=6000 + s * 10 + i, k=4) for i, r in enumerate(CONTINUAL_RULES)]
        # --- constructive model: one model, concepts accumulate ---
        m = ConstructiveModel(**CCM_KW)
        ccm_steps: list[dict[str, Any]] = []
        ccm_fit_times: list[float] = []
        for i, task in enumerate(tasks):
            before_params = m.n_parameters()
            before_nodes = m.n_stored_nodes()
            rep, dt, peak = timed(lambda t=task: m.fit(t.train, t.input_type, t.output_type,
                                                       validation=t.val, probes=t.probes))
            ccm_fit_times.append(dt)
            accs = [evaluate(None, t, support=t.train, model=m)["accuracy"] for t in tasks[: i + 1]]
            ccm_steps.append({
                "step": i, "task": task.name, "accepted": rep.accepted,
                "fit_s": dt, "peak_bytes": peak, "nodes": rep.nodes,
                "new_params": m.n_parameters() - before_params,
                "new_nodes": m.n_stored_nodes() - before_nodes,
                "accuracies": accs,
            })
        # --- MLP: sequential fine-tuning on the same task order ---
        mlp = MLP()
        mlp_steps: list[dict[str, Any]] = []
        for i, task in enumerate(tasks):
            data = list(task.train) + list(task.val)
            if i == 0:
                _, dt, peak = timed(lambda d=data, t=task: mlp.fit(d, t.input_type, t.output_type,
                                                                   validation=t.val))
            else:
                _, dt, peak = timed(lambda d=data, t=task: mlp.partial_fit(d, t.input_type, t.output_type))
            accs = [evaluate(mlp.predict, t)["accuracy"] for t in tasks[: i + 1]]
            mlp_steps.append({"step": i, "task": task.name, "fit_s": dt, "peak_bytes": peak,
                              "new_params": mlp.n_parameters(), "accuracies": accs})
        runs.append({"seed": s, "ccm": ccm_steps, "mlp": mlp_steps})
    return {"runs": runs,
            "summary": {
                "ccm_final_mean": statistics.fmean(
                    statistics.fmean(r["ccm"][-1]["accuracies"]) for r in runs),
                "ccm_forgetting_mean": statistics.fmean(_forgetting(r["ccm"]) for r in runs),
                "mlp_final_mean": statistics.fmean(
                    statistics.fmean(r["mlp"][-1]["accuracies"]) for r in runs),
                "mlp_forgetting_mean": statistics.fmean(_forgetting(r["mlp"]) for r in runs),
                "mlp_first_task_final": statistics.fmean(r["mlp"][-1]["accuracies"][0] for r in runs),
                "ccm_first_task_final": statistics.fmean(r["ccm"][-1]["accuracies"][0] for r in runs),
            }}


# ---------------------------------------------------------------------------
# Experiment F - error-driven correction
# ---------------------------------------------------------------------------


def experiment_f(noise_seeds: list[int] = tuple(range(10)),
                 contradiction_seeds: list[int] = (0, 1, 2, 3),
                 mirror_seeds: list[int] = (0, 1, 2, 3)) -> dict[str, Any]:
    noise: list[dict[str, Any]] = []
    for s in noise_seeds:
        base = W.math_task("square", seed=7000 + s, k=4)
        task = W.noisy_task(base, seed=7500 + s, flip=1)
        m = ConstructiveModel(**CCM_KW)
        rep, dt, peak = timed(lambda: m.fit(task.train, task.input_type, task.output_type,
                                            validation=task.val, probes=task.probes))
        ev = evaluate(None, task, support=task.train, model=m)
        noise.append({
            "seed": s, "injected": task.meta["flipped_indices"], "quarantined": rep.quarantined,
            "localised": sorted(rep.quarantined) == sorted(task.meta["flipped_indices"]),
            "accepted": rep.accepted, "abstained": rep.abstained, "reason": rep.reason,
            "fit_s": dt, "peak_bytes": peak, "candidates": rep.candidates[:2],
            **{f"test_{k}": v for k, v in ev.items()},
        })

    contra: list[dict[str, Any]] = []
    for s in contradiction_seeds:
        task = W.contradictory_task("square", "double", seed=8000 + s, k=4)
        m = ConstructiveModel(**CCM_KW)
        rep, dt, _ = timed(lambda: m.fit(task.train, task.input_type, task.output_type,
                                         validation=task.val, probes=task.probes))
        ev = evaluate(None, task, support=task.train, model=m)
        contra.append({"seed": s, "accepted": rep.accepted, "abstained": rep.abstained,
                       "reason": rep.reason, "fit_s": dt,
                       **{f"test_{k}": v for k, v in ev.items()}})

    mirror: list[dict[str, Any]] = []
    for s in mirror_seeds:
        base = W.mirrored_scene_task(seed=9000 + s, k=4)
        shift = W.mirrored_shift_task(seed=9000 + s, k=4)
        m = ConstructiveModel(**CCM_KW)
        rep = m.fit(base.train, base.input_type, base.output_type,
                    validation=base.val, probes=base.probes)
        old_before = evaluate(None, base, support=base.train, model=m)
        rev, dt, peak = timed(lambda: m.revise(rep.concept, shift.train, "scene",
                                               validation=shift.val, probes=shift.probes))
        old_after = evaluate(None, base, support=base.train, model=m)
        new_after = evaluate(None, shift, support=shift.train, model=m)
        mirror.append({
            "seed": s, "mode": rev.get("mode"), "guard": rev.get("guard"),
            "parent_guard": rev.get("parent_guard"), "fit_s": dt, "peak_bytes": peak,
            "nodes": rev.get("nodes"), "reason": rev.get("reason"),
            "old_before": old_before["accuracy"], "old_after": old_after["accuracy"],
            "new_after": new_after["accuracy"],
            "retention": rev.get("retention"),
        })
    return {
        "noise": {"rows": noise,
                  "localisation_rate": statistics.fmean(1.0 if r["localised"] else 0.0 for r in noise),
                  "clean_accuracy": statistics.fmean(r["test_accuracy"] for r in noise),
                  "false_accept_rate": statistics.fmean(
                      1.0 if (r["accepted"] and not r["localised"]) else 0.0 for r in noise)},
        "contradiction": {"rows": contra,
                          "rejection_rate": statistics.fmean(1.0 if not r["accepted"] else 0.0 for r in contra),
                          "coverage": statistics.fmean(r["test_coverage"] for r in contra)},
        "mirror": {"rows": mirror,
                   "scoped_rate": statistics.fmean(1.0 if r["mode"] == "scoped" else 0.0 for r in mirror),
                   "old_retention": statistics.fmean(r["old_after"] for r in mirror),
                   "new_accuracy": statistics.fmean(r["new_after"] for r in mirror)},
    }


# ---------------------------------------------------------------------------
# Experiment G - does the learned retrieval prior do any work?
# ---------------------------------------------------------------------------


def experiment_g(seeds: list[int] = (0, 1, 2)) -> dict[str, Any]:
    """Mixed-type continual suite plus a prior ablation.

    The only learned parameters in CCM are the 17 weights of the retrieval
    prior.  This measures whether they change any prediction, and reports the
    retention of a mixed scene/list/integer suite.
    """
    rows: list[dict[str, Any]] = []
    for s in seeds:
        tasks = W.continual_suite(s)
        m = ConstructiveModel(**CCM_KW)
        fit_times: list[float] = []
        for t in tasks:
            _, dt, _ = timed(lambda t=t: m.fit(t.train, t.input_type, t.output_type,
                                               validation=t.val, probes=t.probes))
            fit_times.append(dt)
        for t in tasks:
            with_prior = evaluate(None, t, support=t.train, model=m)
            saved = list(m.w)
            m.w = [0.0] * len(saved)
            without = evaluate(None, t, support=t.train, model=m)
            m.w = saved
            rows.append({
                "seed": s, "task": t.name, "rule": t.rule,
                "accuracy_with": with_prior["accuracy"],
                "accuracy_without": without["accuracy"],
                "same": abs(with_prior["accuracy"] - without["accuracy"]) < 1e-12,
                "retention": m.retention(),
                "n_concepts": len(m.concepts),
                "n_parameters": m.n_parameters(),
                "stored_nodes": m.n_stored_nodes(),
                "fit_times": fit_times,
            })
    return {"rows": rows,
            "summary": {
                "accuracy_with_prior": statistics.fmean(r["accuracy_with"] for r in rows),
                "accuracy_without_prior": statistics.fmean(r["accuracy_without"] for r in rows),
                "identical_predictions": statistics.fmean(1.0 if r["same"] else 0.0 for r in rows),
                "n_concepts": statistics.fmean(r["n_concepts"] for r in rows),
                "n_parameters": statistics.fmean(r["n_parameters"] for r in rows),
                "stored_nodes": statistics.fmean(r["stored_nodes"] for r in rows),
            }}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _forgetting(steps: list[dict[str, Any]]) -> float:
    """Largest accuracy drop on a task between learning it and the final step."""
    final = steps[-1]["accuracies"]
    drops = [max(0.0, steps[i]["accuracies"][i] - final[i]) for i in range(len(steps))]
    return max(drops) if drops else 0.0


def _by_system(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for system in sorted({r["system"] for r in records}):
        out[system] = summarise([r for r in records if r["system"] == system])
    return out


def _by_rule(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for rule in sorted({r["rule"] for r in records}):
        sel = [r for r in records if r["rule"] == rule]
        out[rule] = {sys: summarise([r for r in sel if r["system"] == sys])
                     for sys in sorted({r["system"] for r in sel})}
    return out


def _summ(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if "test_accuracy" in r]
    if not records:
        return {"n": 0}
    return {"n": len(ok),
            "accuracy_mean": statistics.fmean(r["test_accuracy"] for r in ok) if ok else None,
            "accepted_rate": statistics.fmean(1.0 if r.get("accepted") else 0.0 for r in records),
            "fit_s_median": statistics.median(r["fit_s"] for r in records) if records else None}


def _curve(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Group by (rule, k, system) -> mean accuracy; then examples-to-competence."""
    grouped: dict[tuple[str, int, str], list[float]] = {}
    for r in rows:
        if "test_accuracy" not in r:
            continue
        grouped.setdefault((r["rule"], r.get("k", -1), r["system"]), []).append(r["test_accuracy"])
    table: dict[str, Any] = {}
    for (rule, k, system), accs in grouped.items():
        table.setdefault(rule, {}).setdefault(system, {})[str(k)] = statistics.fmean(accs)
    e2c: dict[str, Any] = {}
    for rule, systems in table.items():
        e2c[rule] = {}
        for system, by_k in systems.items():
            ks = sorted(int(x) for x in by_k)
            hit = [k for k in ks if by_k[str(k)] >= THRESHOLD]
            e2c[rule][system] = min(hit) if hit else None
    return {"table": table, "examples_to_competence": e2c}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _hardware() -> dict[str, Any]:
    ram_gb = None
    try:
        import ctypes

        class _Mem(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        m = _Mem()
        m.dwLength = ctypes.sizeof(_Mem)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        ram_gb = round(m.ullTotalPhys / 1e9, 2)
    except Exception:
        pass
    info: dict[str, Any] = {
        "cpu": platform.processor(),
        "logical_cores": os.cpu_count(),
        "ram_gb": ram_gb,
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "cuda": False,
        "c_compiler": "none on PATH",
    }
    try:
        import numpy
        info["numpy"] = numpy.__version__
    except Exception:
        pass
    return info


def run_all(quick: bool = False, out: Path | None = None) -> dict[str, Any]:
    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "hardware": _hardware(),
        "ccm": CCM_KW,
        "threshold": THRESHOLD,
        "quick": quick,
    }
    seeds = (0, 1) if quick else (0, 1, 2)
    result: dict[str, Any] = {"meta": meta}
    t0 = time.perf_counter()
    result["A"] = experiment_a(seeds=seeds)
    result["A_curve"] = experiment_a_curve(seeds=(0, 1) if quick else (0, 1, 2))
    result["B"] = experiment_b(seeds=seeds)
    result["B_curve"] = experiment_b_curve(seeds=(0, 1) if quick else (0, 1, 2))
    result["C"] = experiment_c(seeds=(0, 1) if quick else (0, 1, 2, 3))
    result["D"] = experiment_d(seeds=seeds)
    result["D_pairs"] = experiment_d_pairs(seeds=(0, 1) if quick else (0, 1, 2))
    result["E"] = experiment_e(seeds=seeds)
    result["G"] = experiment_g(seeds=(0, 1) if quick else (0, 1, 2))
    result["F"] = experiment_f(noise_seeds=tuple(range(6 if quick else 10)),
                               contradiction_seeds=(0, 1) if quick else (0, 1, 2, 3),
                               mirror_seeds=(0, 1) if quick else (0, 1, 2, 3))
    meta["total_s"] = time.perf_counter() - t0
    meta["speedups"] = {"matched_quality": _speedups(result),
                        "curves": _curve_speedups(result)}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    target = out or RESULTS_DIR / f"results-{time.strftime('%Y%m%d-%H%M%S')}.json"
    meta["output"] = str(target)
    target.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    (RESULTS_DIR / "latest.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def _speedups(result: dict[str, Any]) -> dict[str, Any]:
    """CPU/example ratios at matched quality (>=0.8 accuracy for both)."""
    out: list[dict[str, Any]] = []
    for exp in ("A", "B", "D"):
        for rec in result.get(exp, {}).get("episodes", []):
            if rec["system"] != "ccm" or rec.get("test_accuracy", 0) < THRESHOLD:
                continue
            peers = [r for r in result[exp]["episodes"]
                     if r["task"] == rec["task"] and r.get("seed") == rec.get("seed")
                     and r["system"] in ("mlp", "nearest_neighbor", "memorizer")
                     and r.get("test_accuracy", 0) >= THRESHOLD]
            if not peers:
                continue
            best = max(peers, key=lambda r: r["test_accuracy"])
            out.append({"experiment": exp, "rule": rec["rule"], "seed": rec.get("seed"),
                        "ccm_s": rec["fit_s"], "peer": best["system"],
                        "peer_s": best["fit_s"],
                        "time_ratio": best["fit_s"] / max(rec["fit_s"], 1e-9)})
    return out


def _curve_speedups(result: dict[str, Any]) -> dict[str, Any]:
    """Examples-to-competence and time-to-competence ratios from the curves.

    A ratio is only reported for a rule where both systems reach the quality
    threshold; ``None`` means the baseline did not reach it in the measured
    range and no ratio is claimed.
    """
    out: dict[str, Any] = {}
    for block_name in ("A_curve", "B_curve"):
        block = result.get(block_name, {})
        rows = block.get("rows", [])
        table = block.get("curve", {}).get("examples_to_competence", {})
        for rule, systems in table.items():
            ccm_k = systems.get("ccm")
            if ccm_k is None:
                continue
            ccm_t = _median_fit(rows, rule, ccm_k, "ccm")
            for system, peer_k in systems.items():
                if system == "ccm" or peer_k is None:
                    continue
                peer_t = _median_fit(rows, rule, peer_k, system)
                out[f"{block_name}:{rule}:{system}"] = {
                    "ccm_k": ccm_k,
                    "peer_k": peer_k,
                    "example_ratio": peer_k / ccm_k,
                    "ccm_fit_s": ccm_t,
                    "peer_fit_s": peer_t,
                    "time_ratio": (peer_t / ccm_t) if ccm_t and peer_t else None,
                }
    return out


def _median_fit(rows: list[dict[str, Any]], rule: str, k: int, system: str) -> float | None:
    hits = [r["fit_s"] for r in rows
            if r.get("rule") == rule and r.get("k") == k and r.get("system") == system]
    return statistics.median(hits) if hits else None


if __name__ == "__main__":
    res = run_all(quick="--quick" in sys.argv)
    print(json.dumps({k: v for k, v in res.items() if k != "meta"}, indent=2, default=str)[:8000])
    print("meta:", json.dumps(res["meta"], indent=2))
