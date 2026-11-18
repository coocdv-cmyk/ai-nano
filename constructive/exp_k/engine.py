"""Adapter: drive EXP-I mechanisms with EXP-K data (read-only).

Test labels are touched only by the evaluation side, after a decision is made.
"""

from __future__ import annotations

import random
from typing import Any

from constructive.exp_i import dsl
from constructive.exp_i.types import Task, ValidationState
from constructive.exp_i.validation import ValidationEngine

from .config import ExpKConfig


def probe_pool_for(task: Task, n: int = 24) -> list[Any]:
    used = {repr(e.x) for e in (task.train + task.validation_pool + task.test)}
    rng = random.Random(4242 + sum(ord(c) for c in task.name))
    td = task.meta.get("train_domain", {"lo": 0, "hi": 9})
    out: list[Any] = []
    guard = 0
    while len(out) < n and guard < 10000:
        guard += 1
        if task.input_type == "int":
            v = rng.randint(int(td.get("hi", 9)) + 1, int(td.get("hi", 9)) + 70)
        elif task.input_type == "pair":
            v = (rng.randint(int(td.get("hi", 9)) + 1, int(td.get("hi", 9)) + 40),
                 rng.randint(0, 20))
        else:
            length = rng.randint(int(td.get("len_hi", 5)) + 1, int(td.get("len_hi", 5)) + 4)
            v = [rng.randint(30, 95) for _ in range(length)]
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


def test_accuracy(program: Any, task: Task) -> float | None:
    if program is None:
        return None
    ok = 0
    for ex in task.test:
        try:
            if dsl.evaluate(program, ex.x) == ex.y:
                ok += 1
        except Exception:
            pass
    return round(ok / len(task.test), 6) if task.test else None


def support_consistent(program: Any, task: Task) -> bool:
    if program is None:
        return False
    for ex in task.train + task.validation_pool:
        try:
            if dsl.evaluate(program, ex.x) != ex.y:
                return False
        except Exception:
            return False
    return True


def run_validation(task: Task, cfg: ExpKConfig, *, calibrator: Any = None,
                   probe_pool: list[Any] | None = None,
                   governor: Any = None) -> tuple[Any, dict[str, Any]]:
    eng = ValidationEngine(budget=cfg.n_val,
                           abstain_threshold=cfg.abstain_threshold,
                           calibrator=calibrator, eval_fn=dsl.evaluate)
    result = eng.validate(task.meta.get("candidates", []), task,
                          probe_pool=probe_pool if probe_pool is not None
                          else probe_pool_for(task), governor=governor)
    prog = _program_for(result.candidate_id, task)
    fields = {
        "state": result.state.value,
        "accepted": result.accepted,
        "correct": bool(result.accepted and result.candidate_id == "true"),
        "wrong": bool(result.accepted and result.candidate_id != "true"),
        "abstained": not result.accepted,
        "false_rejection": bool((not result.accepted) and
                                support_consistent(_program_for("true", task), task)),
        "non_identifiable": result.state is ValidationState.NON_IDENTIFIABLE,
        "contradictory": result.state is ValidationState.CONTRADICTORY,
        "insufficient": result.state is ValidationState.INSUFFICIENT_EVIDENCE,
        "confidence": round(float(result.confidence), 6),
        "calibrated_confidence": (round(float(result.calibrated_confidence), 6)
                                  if result.calibrated_confidence is not None else None),
        "test_accuracy": test_accuracy(prog, task),
        "val_accuracy": round(float(result.accuracy), 6),
        "coverage": round(float(result.coverage), 6),
        "evidence_informative": sum(1 for e in result.evidence if e.informative),
        "evidence_redundant": sum(1 for e in result.evidence if not e.informative),
    }
    return result, fields
