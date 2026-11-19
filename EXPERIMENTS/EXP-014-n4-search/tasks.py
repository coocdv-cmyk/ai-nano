"""Fresh N4 task inventory: new themes fed through EXP-012's verified templates.

No N0, EXP-011 or N2 task/theme is reused.  The templates come from the already
verified N2 generator, but every module, function, value and phrasing here is
new to N4.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
N2 = ROOT / "EXPERIMENTS" / "EXP-012-n2-mtp"
for p in (str(ROOT), str(N2)):
    if p not in sys.path:
        sys.path.insert(0, p)

import corpus as n2  # noqa: E402

BLOCKS: dict[str, tuple[tuple[Any, ...], ...]] = {
    "mutable_default": (
        ("vault", "open_vault", "drawer", 1, 2, "token"),
        ("crate", "fill_crate", "slot", 3, 4, "item"),
        ("queue", "enqueue_job", "pending", 5, 6, "job"),
        ("tray", "drop_tray", "stack", 7, 8, "card"),
        ("bin", "store_bin", "items", 9, 10, "part"),
        ("folder", "add_folder", "files", 11, 12, "doc"),
    ),
    "boundary_guard": (
        ("racks", "peek_rack", "racks", "pos", (14, 15, 16)),
        ("stacks", "top_stack", "stacks", "depth", (24, 25, 26)),
        ("cells", "read_cell", "cells", "at", (34, 35, 36)),
        ("rows", "show_row", "rows", "row", (44, 45, 46)),
        ("pages", "get_page", "pages", "no", (54, 55, 56)),
        ("slots", "take_slot", "slots", "ix", (64, 65, 66)),
    ),
    "off_by_one": (
        ("slices", "take_head", "slices", 6),
        ("windows", "first_window", "windows", 7),
        ("blocks", "lead_block", "blocks", 8),
        ("groups", "front_group", "groups", 9),
        ("spans", "open_span", "spans", 10),
        ("chains", "head_chain", "chains", 11),
    ),
    "comparison_flip": (
        ("access2", "can_enter", "level", "floor", 14),
        ("gates2", "may_pass", "rank", "minrank", 15),
        ("locks2", "opens", "key", "need", 16),
        ("permits2", "allowed", "grade", "bar", 17),
        ("tickets2", "admits", "tier", "cut", 18),
        ("passes2", "qualifies", "score", "mark", 19),
    ),
}

FAMILIES = tuple(BLOCKS)


def _single_tasks() -> list[dict[str, Any]]:
    n2._theme_blocks = lambda: BLOCKS  # type: ignore[assignment]
    out: list[dict[str, Any]] = []
    for family in FAMILIES:
        for variant in range(len(BLOCKS[family])):
            out.append(n2.make_task(family, variant))
    return out


def compose(singles: list[dict[str, Any]], group: int) -> dict[str, Any]:
    """Merge three single-function workspaces into one multi-function task.

    Train: groups 0..2; held-out: groups 3..5.
    """
    files: dict[str, str] = {}
    tests: list[str] = []
    specs: list[str] = []
    for i, task in enumerate(singles):
        specs.append(task["spec"])
        for rel, text in task["files"].items():
            if rel.startswith("tests/"):
                tests.append(text.replace("def test_", f"def test_{i}_"))
            else:
                if rel in files:
                    raise ValueError(f"module name collision: {rel}")
                files[rel] = text
    files["tests/test_service.py"] = "\n\n".join(tests)
    return {
        "id": f"n4-multi-{group:02d}",
        "family": "multi",
        "variant": group,
        "split": "train" if group < 3 else "holdout",
        "spec": " ; ".join(specs),
        "files": files,
        "n4_split": "train" if group < 3 else "holdout",
        "components": [t["id"] for t in singles],
    }


def make_tasks() -> list[dict[str, Any]]:
    """Six three-function tasks: three train, three held-out.

    `off_by_one` is excluded from composition: its fresh N4 variants are not
    repaired by the current operator (verified separately), so including them
    would make every composed task unsolvable and the gate vacuous. Recorded as
    a negative result rather than silently dropped.
    """
    usable = {"mutable_default", "boundary_guard", "comparison_flip"}
    singles = [t for t in _single_tasks() if t["family"] in usable]
    groups: list[list[dict[str, Any]]] = [[] for _ in range(6)]
    for i, task in enumerate(singles):
        groups[i % 6].append(task)
    return [compose(group, gi) for gi, group in enumerate(groups)]
