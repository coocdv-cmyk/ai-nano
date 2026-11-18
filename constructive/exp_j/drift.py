"""EXP-J distribution-drift generators.

The support domain stays fixed; the validation and test domains move. This
isolates the engine's ability to notice that incoming evidence no longer comes
from the training distribution.
"""

from __future__ import annotations

from typing import Any

from constructive.exp_i.types import Task

from . import generators as G
from .types import DriftKind


def _shift_domain(base: dict[str, Any], step: int, *, gradual: bool) -> dict[str, Any]:
    d = dict(base)
    if "lo" in d and "hi" in d:
        move = step * (4 if gradual else 40)
        d["lo"] = d["lo"] + move
        d["hi"] = d["hi"] + move
    if "len" in d:
        d["len"] = d["len"] + step * (1 if gradual else 4)
        d["len_hi"] = d["len_hi"] + step * (1 if gradual else 4)
    return d


def make_drift_stream(rule: str, seed: int, kind: DriftKind | str, *,
                      n_base: int = 3, n_drift: int = 6, n_recover: int = 3,
                      k: int = 4, n_val: int = 4, n_test: int = 10) -> tuple[list[Task], int, int]:
    """Return (tasks, drift_index, recovery_index).

    Phases: ``n_base`` in-domain episodes, then ``n_drift`` drifted episodes,
    then ``n_recover`` episodes back in-domain. Gradual drift increments a step
    counter; abrupt drift jumps directly to a far domain.
    """
    kind = DriftKind(kind)
    train_d, test_d = G.default_domains(rule, "none")
    tasks: list[Task] = []
    idx = 0
    for i in range(n_base):
        tasks.append(_mk(rule, seed, idx, 0, train_d, train_d, test_d,
                         k, n_val, n_test, "base", kind))
        idx += 1
    drift_index = idx
    for i in range(n_drift):
        step = (i + 1) if kind is DriftKind.GRADUAL else 2
        drifted = _shift_domain(test_d, step, gradual=(kind is DriftKind.GRADUAL))
        tasks.append(_mk(rule, seed, idx, step, train_d, drifted, drifted,
                         k, n_val, n_test, "drift", kind))
        idx += 1
    recovery_index = idx
    for i in range(n_recover):
        tasks.append(_mk(rule, seed, idx, 0, train_d, train_d, test_d,
                         k, n_val, n_test, "recover", kind))
        idx += 1
    return tasks, drift_index, recovery_index


def _mk(rule, seed, idx, step, train_d, val_d, test_d, k, n_val, n_test,
        phase, kind) -> Task:
    t = G.build_task(rule, seed * 101 + idx, train_d, val_d, test_d,
                     k=k, n_val=n_val, n_test=n_test, tag=f"{kind.value}-{phase}")
    t.meta.update({"phase": phase, "drift_step": step, "drift_kind": kind.value})
    return t
