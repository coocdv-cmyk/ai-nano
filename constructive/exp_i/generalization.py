"""EXP-I-E — numeric and structural generalization experiments.

Tests semantic generalization versus accidental fitting with independent
generator families. Domain boundaries are tracked explicitly: acceptance is
only credited when the validated domain supports the extrapolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import generators as G
from .config import ExpIConfig
from .dsl import evaluate
from .metrics import mean_ci, rates
from .types import Expressiveness, ValidationState
from .validation import ValidationEngine


@dataclass
class DomainBoundary:
    train_domain: list[int] | None
    test_domain: list[int] | None
    structure: str
    extrapolating: bool
    claim: str


def domain_boundary(rule: str, meta: dict[str, Any], state: ValidationState) -> DomainBoundary:
    train = meta.get("train_domain")
    test = meta.get("test_domain")
    structure = "list-length" if rule in G.LIST_RULES else (
        "scene-density" if rule in G.SCENE_RULES else "numeric-magnitude")
    extrapolating = bool(train and test and max(test) > max(train))
    if state is ValidationState.ACCEPTED and extrapolating:
        claim = ("accepted extrapolation: the rule was discriminated within the "
                 "validated domain and applied beyond it; OOD accuracy measured")
    elif extrapolating:
        claim = ("abstained: available evidence does not identify the rule across "
                 "the domain boundary; extrapolation is NOT credited")
    else:
        claim = "in-domain decision; no extrapolation claimed"
    return DomainBoundary(train, test, structure, extrapolating, claim)


def run_ood(config: ExpIConfig, *, seeds: tuple[int, ...] | None = None,
            rules: tuple[str, ...] | None = None,
            n_episodes: int | None = None) -> dict[str, Any]:
    seeds = tuple(seeds if seeds is not None else config.seeds)
    rules = tuple(rules if rules is not None else G.OOD_RULES)
    eng = ValidationEngine(budget=config.n_val,
                           abstain_threshold=config.abstain_threshold,
                           eval_fn=evaluate)
    rows: list[dict[str, Any]] = []
    for rule in rules:
        for seed in seeds:
            task = G.numeric_ood_task(rule, seed, k=4, n_test=12)
            cands = G.ood_candidates(rule)
            result = eng.validate(cands, task)
            # Evaluation-side scoring with independent reference labels; never
            # shown to the generator or the candidate selector.
            if rule in G.INT_RULES + G.LIST_RULES:
                fn = G.REFERENCE_LABELS[rule]
                test_correct = sum(1 for e in task.test if _same(fn(e.x), e.y))
                # What the *accepted candidate* would actually predict on test.
                accepted_pred = None
                if result.accepted and result.candidate_id:
                    prog = next((c.program for c in cands if c.id == result.candidate_id), None)
                    if prog is not None:
                        accepted_correct = 0
                        for e in task.test:
                            try:
                                if _same(evaluate(prog, e.x), e.y):
                                    accepted_correct += 1
                            except Exception:
                                pass
                        accepted_pred = accepted_correct
            else:
                test_correct = len(task.test)  # reference-defined, no DSL candidate
                accepted_pred = None
            boundary = domain_boundary(rule, task.meta, result.state)
            express = (Expressiveness.NOT_EXPRESSIBLE.value if not cands
                       else (Expressiveness.NOT_IDENTIFIABLE.value
                             if result.state in (ValidationState.NON_IDENTIFIABLE,
                                                 ValidationState.INSUFFICIENT_EVIDENCE)
                             else Expressiveness.EXPRESSIBLE_DISCOVERED.value))
            wrong_accept = bool(result.accepted and accepted_pred is not None
                                and accepted_pred < 0.8 * max(1, len(task.test)))
            rows.append({
                "rule": rule, "seed": seed, "state": result.state.value,
                "accepted": result.accepted,
                "expressiveness": express,
                "in_domain_reference_accuracy": test_correct / len(task.test),
                "accepted_candidate_test_accuracy": (
                    accepted_pred / len(task.test) if accepted_pred is not None else None),
                "domain_boundary": boundary.__dict__,
                "false_acceptance": wrong_accept,
                "failure_category": (
                    "false-acceptance" if wrong_accept else
                    "non-identifiable" if result.state is ValidationState.NON_IDENTIFIABLE else
                    "insufficient-evidence" if result.state is ValidationState.INSUFFICIENT_EVIDENCE else
                    "representation-limitation" if not cands else
                    "success" if result.accepted else "false-rejection"),
                "n_test": len(task.test),
            })

    summary: dict[str, Any] = {}
    for rule in rules:
        sel = [r for r in rows if r["rule"] == rule]
        summary[rule] = {
            "n": len(sel),
            "acceptance_rate": rates(r["accepted"] for r in sel),
            "false_acceptance_rate": rates(r["false_acceptance"] for r in sel),
            "ood_accuracy_given_acceptance": mean_ci(
                r["accepted_candidate_test_accuracy"] for r in sel
                if r["accepted_candidate_test_accuracy"] is not None),
            "states": {s: sum(1 for r in sel if r["state"] == s)
                       for s in {r["state"] for r in sel}},
        }
    return {
        "experiment_id": "I-E",
        "experiment_name": "Numeric and Structural Generalization",
        "status": "completed",
        "rules": list(rules), "seeds": list(seeds),
        "n_episodes": len(rows), "episodes": rows, "summary": summary,
        "baseline_reference": {
            "source": "CONSTRUCTIVE-EXP-H-REPORT.md H3",
            "numeric_ood_accuracy": 0.728,
            "sum_digits_ood_accuracy": 0.185,
            "note": "EXP-H accepted sum_digits spuriously; EXP-I must abstain or generalize",
        },
    }


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    return a == b
