"""H6 — Autonomous representation / abstraction discovery.

An isolated *experimental* mechanism: after a validated fit, frequent,
well-typed subtrees of the winning program are stored as low-confidence
macro concepts.  They are search units only; they are not retrieved as
solutions and they carry no validation of their own beyond the behavioural
evidence implied by the parent program.

Honesty note: this is a *heuristic proposal*, implemented so we can measure
whether it helps.  It is not part of CCM and never reaches
``omega_nano``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..core import Term, V, eval_term, same_value, term_cost
from ..model import Concept, ConstructiveModel
from ..worlds import Task
from .config import probe_hardware
from .generators import build_abstraction_tasks
from .h1_discovery import _counts, _mean, _median  # noqa: F401  (report helper)


def _walk(t: Term):
    yield t
    for a in t.args:
        yield from _walk(a)


def harvest(model: ConstructiveModel, term: Term, ret: str, max_macros: int = 6) -> list[str]:
    from ..core import infer_type
    from ..model import embed_term
    made: list[str] = []
    seen = set()
    for sub in _walk(term):
        if sub is term:
            continue
        if sub.op in ("var", "lit", "concept"):
            continue
        key = str(sub)
        if key in seen:
            continue
        seen.add(key)
        try:
            sub_ret = infer_type(sub, {"x": "list"}, model.prims, model.concepts)
        except Exception:
            continue
        if not sub_ret or sub_ret == "any":
            continue
        name = f"m{len(model.concepts) + len(model.hypotheses):03d}"
        concept = Concept(
            name=name,
            params=("x",),
            body=sub,
            ret=sub_ret,
            input_type="list",
            arg_types=("list",),
            confidence=0.35,
            support=0,
            created=len(model.concepts),
            embedding=embed_term(sub, model.dim),
            validation=[],
            note="harvested macro",
        )
        model.concepts[name] = concept
        model.active.append(name)
        made.append(name)
        if len(made) >= max_macros:
            break
    return made


def run_h6(cfg, *, seeds=None, k: int = 4) -> dict[str, Any]:
    seeds = tuple(seeds or cfg.seeds[:3])
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for arm in ("baseline", "harvest"):
            model = ConstructiveModel(max_cost=3, beam=45, max_nodes=250_000,
                                      time_budget_s=4.0)
            tasks = build_abstraction_tasks(seed, k=k)
            for idx, task in enumerate(tasks):
                before_nodes = model.n_stored_nodes()
                t_fit = time.perf_counter()
                rep = model.fit(task.train, task.input_type, task.output_type,
                                validation=task.val, probes=task.probes)
                fit_s = time.perf_counter() - t_fit
                winner_nodes = model.concepts[rep.concept].body.size() if rep.accepted else None
                macros: list[str] = []
                if arm == "harvest" and rep.accepted:
                    macros = harvest(model, model.concepts[rep.concept].body,
                                     task.output_type, max_macros=4)
                correct = 0
                for x, y in task.test:
                    v = model.predict_episode(task.train, x, task.input_type)
                    if v is not None and same_value(v, y):
                        correct += 1
                rows.append({
                    "seed": seed,
                    "arm": arm,
                    "task": task.meta["hidden_program"],
                    "stream_index": idx,
                    "accepted": rep.accepted,
                    "fit_s": fit_s,
                    "nodes": rep.nodes,
                    "test_accuracy": correct / len(task.test),
                    "winner_nodes": winner_nodes,
                    "new_nodes": model.n_stored_nodes() - before_nodes,
                    "macros_harvested": macros,
                    "n_concepts": len(model.concepts),
                    "stored_nodes": model.n_stored_nodes(),
                })
    # reuse metric: did a macro appear in a later winner?
    for row in rows:
        row["n_macros"] = len(row["macros_harvested"])
    by_arm: dict[str, Any] = {}
    for arm in ("baseline", "harvest"):
        sel = [r for r in rows if r["arm"] == arm]
        by_arm[arm] = {
            "n": len(sel),
            "accuracy_mean": _mean(r["test_accuracy"] for r in sel),
            "fit_s_median": _median(r["fit_s"] for r in sel),
            "nodes_sum": sum(r["nodes"] for r in sel),
            "accepted_rate": _mean(r["accepted"] for r in sel),
            "winner_nodes_mean": _mean(r["winner_nodes"] for r in sel if r["winner_nodes"] is not None),
            "final_stored_nodes_median": _median(
                [r["stored_nodes"] for r in sel]),
        }
    return {
        "experiment_id": "H6",
        "experiment_name": "Autonomous Representation and Abstraction Discovery",
        "status": "completed",
        "mechanism": "harvest-large-shared-subtrees (isolated, not in CCM)",
        "episodes": rows,
        "summary": by_arm,
        "hardware": probe_hardware().as_dict(),
        "wall_s": time.perf_counter() - t0,
    }
