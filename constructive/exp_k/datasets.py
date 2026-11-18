"""EXP-K dataset construction, manifests and split-integrity checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from constructive.exp_i.types import Task

from . import generators as G
from .config import RESULTS_DIR, fingerprint_of
from .distributions import SHIFTS


def split_integrity(task: Task) -> dict[str, Any]:
    tr = [repr(e.x) for e in task.train]
    va = [repr(e.x) for e in task.validation_pool]
    te = [repr(e.x) for e in task.test]
    return {
        "task": task.name,
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "train_val_overlap": bool(set(tr) & set(va)),
        "train_test_overlap": bool(set(tr) & set(te)),
        "val_test_overlap": bool(set(va) & set(te)),
        "clean": not ((set(tr) & set(va)) or (set(tr) & set(te)) or (set(va) & set(te))),
    }


def dataset_manifest(config: dict[str, Any], seeds: tuple[int, ...],
                     *, rules: tuple[str, ...] | None = None,
                     shifts: tuple[str, ...] = SHIFTS) -> dict[str, Any]:
    rules = rules or G.ALL_RULES
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for rule in rules:
            for shift in shifts:
                t = G.make_task(rule, seed * 17 + sum(ord(c) for c in shift),
                                shift=shift, n_val=config["n_val"],
                                n_test=config["n_test"])
                integ = split_integrity(t)
                rows.append({
                    "seed": seed, "rule": rule, "shift": shift,
                    "task": t.name, "params": t.meta["params"],
                    "n_train": integ["n_train"], "n_val": integ["n_val"],
                    "n_test": integ["n_test"], "clean": integ["clean"],
                    "train_domain": t.meta["train_domain"],
                    "test_domain": t.meta["test_domain"],
                    "train_inputs": [e.x for e in t.train],
                    "test_inputs": [e.x for e in t.test],
                })
    dirty = [r["task"] for r in rows if not r["clean"]]
    return {
        "config": config, "n_rules": len(rules), "n_seeds": len(seeds),
        "n_shifts": len(shifts), "n_datasets": len(rows),
        "fingerprint": fingerprint_of({"seeds": list(seeds), "rules": list(rules),
                                       "shifts": list(shifts), "config": config}),
        "all_splits_disjoint": not dirty, "dirty": dirty,
        "rows": rows,
    }


def save_manifest(name: str, manifest: dict[str, Any]) -> Path:
    d = RESULTS_DIR / "manifests"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path
