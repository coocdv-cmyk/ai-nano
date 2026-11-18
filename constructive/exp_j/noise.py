"""EXP-J noise and corruption generators.

All noise is applied to support / validation observations only. Test labels are
never modified: they are the untouched evaluation side of every experiment.
"""

from __future__ import annotations

import random
from typing import Any

from constructive.exp_i import dsl
from constructive.exp_i.types import Example, Task

from .types import NoiseKind

_LEVELS = (0.0, 0.01, 0.05, 0.10, 0.20, 0.30)


def noise_levels() -> tuple[float, ...]:
    return _LEVELS


def _perturb(y: Any, salt: int = 7) -> Any:
    if isinstance(y, bool):
        return not y
    if isinstance(y, int):
        return y + salt
    if isinstance(y, list):
        return list(y) + [salt]
    return y


def _structural_pred(x: Any) -> bool:
    """A deterministic structural condition independent of the label."""
    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        return x % 2 == 0
    if isinstance(x, (list, tuple)):
        return len(x) >= 4
    return False


def apply_noise(task: Task, kind: NoiseKind | str, level: float, seed: int) -> Task:
    """Return a new Task with noise injected into support / validation only."""
    kind = NoiseKind(kind)
    rng = random.Random(seed * 5557 + int(level * 1000))
    train = [Example(e.x, e.y) for e in task.train]
    val = [Example(e.x, e.y) for e in task.validation_pool]
    n_flipped = 0
    n_added = 0
    n_removed = 0

    if kind is NoiseKind.CLEAN:
        pass

    elif kind is NoiseKind.RANDOM_LABEL:
        for group in (train, val):
            for i in range(len(group)):
                if rng.random() < level:
                    group[i] = Example(group[i].x, _perturb(group[i].y))
                    n_flipped += 1

    elif kind is NoiseKind.STRUCTURED_CORRUPTION:
        # Corrupt only examples meeting a structural predicate; the systematic
        # bias can fool a learner that fits the majority regime.
        for group in (train, val):
            for i in range(len(group)):
                if _structural_pred(group[i].x) and rng.random() < level:
                    group[i] = Example(group[i].x, _perturb(group[i].y, 13))
                    n_flipped += 1

    elif kind is NoiseKind.CONTRADICTION:
        # Directly detectable: identical input with conflicting label.
        n = max(1, round(level * len(train))) if level > 0 else 0
        for idx in rng.sample(range(len(train)), min(n, len(train))):
            ex = train[idx]
            train.append(Example(ex.x, _perturb(ex.y)))
            n_added += 1

    elif kind is NoiseKind.MISSING:
        if level > 0:
            n = min(len(train) - 2, max(1, round(level * len(train))))
            for idx in sorted(rng.sample(range(len(train)), max(0, n)), reverse=True):
                train.pop(idx)
                n_removed += 1

    elif kind is NoiseKind.DUPLICATE:
        n = max(1, round(level * len(train))) if level > 0 else 0
        for idx in rng.sample(range(len(train)), min(n, len(train))):
            ex = train[idx]
            train.append(Example(ex.x, ex.y))
            n_added += 1

    elif kind is NoiseKind.MISLEADING_SUPPORT:
        # Adversarially misleading support: relabel support points to a
        # distractor program's outputs, so a learner that trusts support alone
        # adopts the wrong hypothesis. Test labels remain untouched.
        cands = task.meta.get("candidates", [])
        distractors = [c for c in cands if c.id != "true"]
        if distractors and level > 0:
            wrong = distractors[0].program
            for i in range(len(train)):
                if rng.random() < level:
                    try:
                        y = dsl.evaluate(wrong, train[i].x)
                        train[i] = Example(train[i].x, y)
                        n_flipped += 1
                    except Exception:
                        pass

    else:  # pragma: no cover - guarded by enum
        raise ValueError(f"unknown noise kind {kind}")

    meta = dict(task.meta)
    meta.update({"noise_kind": kind.value, "noise_level": level,
                 "noise_seed": seed, "n_flipped": n_flipped,
                 "n_added": n_added, "n_removed": n_removed})
    return Task(
        name=f"{task.name}-{kind.value}-{level:.2f}", family=task.family,
        train=train, validation_pool=val, test=list(task.test),
        input_type=task.input_type, output_type=task.output_type, meta=meta,
    )
