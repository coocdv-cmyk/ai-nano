"""Adapter: drive EXP-I mechanisms with EXP-J-generated data (read-only).

This module never modifies EXP-I. It only constructs EXP-I ``Task``/``Candidate``
objects from EXP-J generators and interprets the resulting validation states.
Test labels are used exclusively by the *evaluation side* here, after a decision
has been made — never by candidate selection, support fitting or calibration.
"""

from __future__ import annotations

import random
from typing import Any

from constructive.exp_i import dsl
from constructive.exp_i.types import Candidate, Example, Task, ValidationState
from constructive.exp_i.validation import ValidationEngine

from .config import ExpJConfig
from .types import EpisodeRecord


def probe_pool_for(task: Task, n: int = 24) -> list[Any]:
    """Unlabeled probes drawn outside all labelled splits."""
    used = {repr(e.x) for e in (task.train + task.validation_pool + task.test)}
    rng = random.Random(9001 + sum(ord(c) for c in task.name))
    td = task.meta.get("train_domain", {"lo": 0, "hi": 9})
    out: list[Any] = []
    guard = 0
    while len(out) < n and guard < 10000:
        guard += 1
        if task.input_type == "int":
            v = rng.randint(int(td.get("hi", 9)) + 1, int(td.get("hi", 9)) + 60)
        elif task.input_type == "pair":
            v = (rng.randint(int(td.get("hi", 9)) + 1, int(td.get("hi", 9)) + 40),
                 rng.randint(0, 20))
        else:
            length = rng.randint(int(td.get("len_hi", 5)) + 1,
                                 int(td.get("len_hi", 5)) + 4)
            v = [rng.randint(30, 90) for _ in range(length)]
        if repr(v) not in used and v not in out:
            out.append(v)
    return out


def _program_for(cid: str | None, task: Task) -> Any | None:
    if cid is None:
        return None
    for c in task.meta.get("candidates", []):
        if c.id == cid:
            return c.program
    return None


def _test_accuracy(program: Any, task: Task) -> float | None:
    if program is None:
        return None
    correct = 0
    for ex in task.test:
        try:
            if dsl.evaluate(program, ex.x) == ex.y:
                correct += 1
        except Exception:
            pass
    return round(correct / len(task.test), 6) if task.test else None


def _support_consistent(program: Any, task: Task) -> bool:
    if program is None:
        return False
    for ex in task.train + task.validation_pool:
        try:
            if dsl.evaluate(program, ex.x) != ex.y:
                return False
        except Exception:
            return False
    return True


def run_validation(task: Task, cfg: ExpJConfig, *,
                   calibrator: Any = None,
                   probe_pool: list[Any] | None = None,
                   governor: Any = None) -> tuple[Any, dict[str, Any]]:
    """Validate with EXP-I engine; return (result, derived episode fields)."""
    eng = ValidationEngine(budget=cfg.n_val,
                           abstain_threshold=cfg.abstain_threshold,
                           calibrator=calibrator, eval_fn=dsl.evaluate)
    probes = probe_pool if probe_pool is not None else probe_pool_for(task)
    result = eng.validate(task.meta.get("candidates", []), task,
                          probe_pool=probes, governor=governor)
    prog = _program_for(result.candidate_id, task)
    acc = _test_accuracy(prog, task)
    correct_prog = _program_for("true", task)
    avail = _support_consistent(correct_prog, task)
    st = result.state
    fields = {
        "state": st.value,
        "accepted": result.accepted,
        "correct_acceptance": bool(result.accepted and result.candidate_id == "true"),
        "wrong_acceptance": bool(result.accepted and result.candidate_id != "true"),
        "abstained": not result.accepted,
        "false_rejection": bool((not result.accepted) and avail),
        "non_identifiable": st is ValidationState.NON_IDENTIFIABLE,
        "contradictory": st is ValidationState.CONTRADICTORY,
        "insufficient": st is ValidationState.INSUFFICIENT_EVIDENCE,
        "confidence": round(float(result.confidence), 6),
        "calibrated_confidence": (round(float(result.calibrated_confidence), 6)
                                  if result.calibrated_confidence is not None else None),
        "test_accuracy": acc,
        "coverage": round(float(result.coverage), 6),
        "evidence_informative": sum(1 for e in result.evidence if e.informative),
        "evidence_redundant": sum(1 for e in result.evidence if not e.informative),
        "provenance": result.provenance,
    }
    return result, fields


def to_episode(task: Task, seed: int, fields: dict[str, Any], *,
               family: str | None = None) -> EpisodeRecord:
    cat = (
        "wrong-acceptance" if fields["wrong_acceptance"] else
        "success" if fields["correct_acceptance"] else
        "non-identifiable" if fields["non_identifiable"] else
        "contradictory" if fields["contradictory"] else
        "insufficient-evidence" if fields["insufficient"] else
        "false-rejection" if fields["false_rejection"] else
        "abstention",
    )[0]
    return EpisodeRecord(
        seed=seed, family=family or task.family, task=task.name,
        state=fields["state"], accepted=fields["accepted"],
        correct_acceptance=fields["correct_acceptance"],
        wrong_acceptance=fields["wrong_acceptance"],
        abstained=fields["abstained"], false_rejection=fields["false_rejection"],
        non_identifiable=fields["non_identifiable"],
        contradictory=fields["contradictory"], insufficient=fields["insufficient"],
        confidence=fields["confidence"], test_accuracy=fields["test_accuracy"],
        coverage=fields["coverage"],
        calibrated_confidence=fields["calibrated_confidence"],
        evidence_informative=fields["evidence_informative"],
        evidence_redundant=fields["evidence_redundant"],
        failure_category=cat,
        meta={"noise_kind": task.meta.get("noise_kind"),
              "noise_level": task.meta.get("noise_level"),
              "shift": task.meta.get("shift")},
    )
