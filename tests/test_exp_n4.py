"""EXP-014 (N4) instrument invariants — the gate itself is in results.json."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "EXPERIMENTS" / "EXP-014-n4-search"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from tasks import make_tasks  # noqa: E402


def test_composed_tasks_are_three_function_and_train_holdout_disjoint():
    tasks = make_tasks()
    assert len(tasks) == 6
    train = [t for t in tasks if t["n4_split"] == "train"]
    hold = [t for t in tasks if t["n4_split"] == "holdout"]
    assert (len(train), len(hold)) == (3, 3)
    train_components = {c for t in train for c in t["components"]}
    hold_components = {c for t in hold for c in t["components"]}
    assert train_components.isdisjoint(hold_components)
    for task in tasks:
        modules = [p for p in task["files"] if not p.startswith("tests/")]
        assert len(modules) == 3, task["id"]
        tests = task["files"]["tests/test_service.py"]
        assert tests.count("def test_") >= 3, task["id"]
        assert len(task["components"]) == 3
