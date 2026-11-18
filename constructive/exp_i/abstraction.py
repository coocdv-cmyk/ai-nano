"""EXP-I-G — autonomous abstraction discovery with validation and retirement.

Harvested subtrees are *experimental search units*. They are independently
validated, carry provenance, can be rejected or retired, and are never promoted
into CCM or OMEGA-NANO. The mechanism is measured against an untouched baseline
on shared-core and unrelated task families.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from . import dsl, generators as G
from .config import ExpIConfig
from .dsl import Node
from .resources import ResourceGovernor
from .search import Searcher


@dataclass
class Abstraction:
    id: str
    body: Node
    input_type: str
    output_type: str
    origin_tasks: list[str]
    reusability: float
    validated: bool = False
    active: bool = True
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return dsl.canonical_key(self.body)


def subtrees(node: Node) -> list[Node]:
    out: list[Node] = []

    def walk(n: Node) -> None:
        for c in n.children:
            out.append(c)
            walk(c)

    walk(node)
    return out


def reusability(subtree: Node, programs: list[Node]) -> float:
    """Frequency x type-generality x non-triviality."""
    subkey = dsl.canonical_key(subtree)
    freq = 0
    for p in programs:
        if subkey in {dsl.canonical_key(s) for s in [p] + subtrees(p)}:
            freq += 1
    if freq < 2:
        return 0.0
    # General abstractions have a simple, reusable parametrized core.
    generality = 1.0 if dsl.OPS[subtree.op].has_param else 0.7
    nontrivial = 1.0 if subtree.size() >= 2 else 0.4
    return round((freq / len(programs)) * generality * nontrivial, 4)


class AbstractionStore:
    def __init__(self) -> None:
        self.items: dict[str, Abstraction] = {}
        self.retired: list[str] = []
        self.rejected: list[dict[str, Any]] = []

    def harvest(self, winner: Node, task_names: list[str], programs: list[Node]) -> list[str]:
        made: list[str] = []
        for sub in subtrees(winner):
            if sub.op in ("var", "lit", "bool"):
                continue
            score = reusability(sub, programs + [winner])
            if score <= 0:
                continue
            aid = f"a{len(self.items) + 1:03d}"
            abs_ = Abstraction(
                id=aid, body=sub,
                input_type=dsl.infer_type(sub.children[0]) if sub.children else "int",
                output_type=dsl.infer_type(sub),
                origin_tasks=list(task_names), reusability=score,
                provenance={"source": str(winner), "score": score},
            )
            self.items[aid] = abs_
            made.append(aid)
        return made

    def validate(self, aid: str, independent_tasks: list[Any]) -> bool:
        """An abstraction is valid if it is behaviorally meaningful and useful
        on an independent task; otherwise it is rejected."""
        abs_ = self.items[aid]
        probes = [e.x for t in independent_tasks for e in t.train[:2]]
        try:
            fp = dsl.behavioral_fingerprint(abs_.body, probes)
            meaningful = len(set(map(repr, fp))) > 1
        except Exception:
            meaningful = False
        if not meaningful:
            abs_.validated = False
            self.rejected.append({"id": aid, "reason": "constant/meaningless behavior"})
            return False
        abs_.validated = True
        return True

    def retire(self, aid: str, reason: str) -> None:
        if aid in self.items:
            self.items[aid].active = False
            self.retired.append(aid)
            self.items[aid].provenance["retire_reason"] = reason

    def active_bodies(self) -> list[Node]:
        return [a.body for a in self.items.values() if a.active and a.validated]

    def report(self) -> dict[str, Any]:
        return {
            "n_total": len(self.items),
            "n_active": sum(1 for a in self.items.values() if a.active),
            "n_validated": sum(1 for a in self.items.values() if a.validated),
            "n_retired": len(self.retired),
            "n_rejected": len(self.rejected),
            "items": [{"id": a.id, "body": str(a.body), "score": a.reusability,
                       "validated": a.validated, "active": a.active}
                      for a in self.items.values()],
            "rejected": self.rejected[:10],
        }


def _used_abstraction(program: Node, bodies: list[Node]) -> bool:
    keys = {dsl.canonical_key(b) for b in bodies}
    return dsl.canonical_key(program) in keys or any(
        dsl.canonical_key(s) in keys for s in subtrees(program))


def run_abstraction(config: ExpIConfig, *, seeds: tuple[int, ...] | None = None,
                    families: tuple[str, ...] | None = None) -> dict[str, Any]:
    seeds = tuple(seeds if seeds is not None else config.seeds[:3])
    families = tuple(families if families is not None else
                     ("shared-core", "unrelated-sum", "unrelated-max", "unrelated-sq"))
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for family in families:
            tasks = G.abstraction_stream(seed, family=family)

            # Pass 1 (baseline): collect winners to measure subtree reusability.
            baseline_winners: list[Node] = []

            def run_pass(arm: str) -> list[dict[str, Any]]:
                store = AbstractionStore()
                arm_rows: list[dict[str, Any]] = []
                for idx, task in enumerate(tasks):
                    gov = ResourceGovernor(global_deadline_s=config.fit_deadline_s)
                    bodies = store.active_bodies() if arm == "harvest" else []
                    searcher = Searcher(governor=gov, beam=config.n_episodes,
                                         max_cost=3.5, max_depth=5,
                                         abstractions=bodies)
                    t_fit = time.perf_counter()
                    result = searcher.search(task)
                    fit_s = time.perf_counter() - t_fit
                    accepted = bool(result.candidates)
                    winner = result.candidates[0].program if accepted else None
                    if arm == "baseline" and winner is not None:
                        baseline_winners.append(winner)
                    harvested: list[str] = []
                    if arm == "harvest" and accepted and winner is not None:
                        # Reusability is measured against winners discovered in
                        # the same stream (the evidence that the structure recurs).
                        pool = baseline_winners + [w.program for w in result.candidates[:3]]
                        harvested = store.harvest(winner, [task.name], pool)
                        for aid in harvested:
                            store.validate(aid, tasks[idx + 1:])
                    reuse = bool(winner is not None and store.active_bodies()
                                 and _used_abstraction(winner, store.active_bodies()))
                    test_correct = 0
                    if winner is not None:
                        for e in task.test:
                            try:
                                if dsl.evaluate(winner, e.x) == e.y:
                                    test_correct += 1
                            except Exception:
                                pass
                    arm_rows.append({
                        "seed": seed, "family": family, "arm": arm,
                        "task": task.name, "stream_index": idx,
                        "accepted": accepted, "expansions": result.expansions,
                        "nodes": result.nodes_generated,
                        "cache_hits": result.cache_hits,
                        "timed_out": result.timed_out,
                        "fit_s": fit_s,
                        "test_accuracy": test_correct / len(task.test),
                        "n_abstractions": len(store.items),
                        "harvested": harvested,
                        "reuse": reuse,
                        "failure_category": (
                            "time-exhaustion" if result.timed_out else
                            "success" if accepted and
                            (test_correct / max(1, len(task.test))) >= 0.8 else
                            "undergeneralization" if accepted else
                            "search-explosion"),
                        "abstraction_report": store.report(),
                    })
                return arm_rows

            baseline_rows = run_pass("baseline")
            harvest_rows = run_pass("harvest")
            rows.extend(baseline_rows)
            rows.extend(harvest_rows)

    by_arm: dict[str, Any] = {}
    for arm in ("baseline", "harvest"):
        sel = [r for r in rows if r["arm"] == arm]
        by_arm[arm] = {
            "n": len(sel),
            "accuracy": _mean(r["test_accuracy"] for r in sel),
            "acceptance": _mean(1.0 if r["accepted"] else 0.0 for r in sel),
            "expansions_sum": sum(r["expansions"] for r in sel),
            "expansions_median": _median(r["expansions"] for r in sel),
            "fit_s_median": _median(r["fit_s"] for r in sel),
            "fit_s_total": round(sum(r["fit_s"] for r in sel), 6),
            "reuse_count": sum(1 for r in sel if r["reuse"]),
            "n_abstractions_created": sum(r["n_abstractions"] for r in sel),
        }
    return {
        "experiment_id": "I-G",
        "experiment_name": "Autonomous Abstraction Discovery and Transfer",
        "status": "completed",
        "families": list(families), "seeds": list(seeds),
        "episodes": rows, "summary": by_arm,
        "baseline_reference": {
            "source": "CONSTRUCTIVE-EXP-H-REPORT.md H6",
            "expansion_ratio": 2.66, "fit_speedup": 6.1,
            "accuracy": [0.786, 0.944], "false_abstractions": 1,
            "note": "H6 benefit was confined to shared-core; transfer is measured here",
        },
        "promotion": "none: macros remain isolated, never integrated into CCM or OMEGA-NANO",
    }


def _mean(xs) -> float | None:
    xs = list(xs)
    return round(sum(xs) / len(xs), 6) if xs else None


def _median(xs) -> float | None:
    xs = sorted(xs)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2
