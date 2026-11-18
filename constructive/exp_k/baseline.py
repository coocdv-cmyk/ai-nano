"""K0 — reproducible baseline reproduction of the four EXP-J failures.

Baseline and all later treatment arms use the same constructors, seeds and
configurations, so comparisons are matched rather than cross-dataset.
"""

from __future__ import annotations

import json
from typing import Any

from constructive.exp_i import dsl
from constructive.exp_i.resources import ResourceGovernor
from constructive.exp_i.search import Searcher
from constructive.exp_i.types import Example, Task, ValidationState

from . import constructors as C
from . import generators as G
from . import metrics as M
from .config import (EXPJ_BASELINE, EXPI_BASELINE, RESULTS_DIR, fingerprint_of,
                     git_branch, git_commit, reproducibility, sha256_file)
from .datasets import dataset_manifest, save_manifest
from .engine import run_validation
from .resources import Monitor
from .types import DriftKind

CAL_RULES = G.ALL_RULES[:6]
DRIFT_RULES = G.ALL_RULES[:4]


def _outcome(fields: dict[str, Any]) -> str:
    if fields["wrong"]:
        return "incorrect_acceptance"
    if fields["correct"]:
        return "correct_acceptance"
    if fields["abstained"]:
        return "correct_abstention" if not fields["false_rejection"] else "incorrect_abstention"
    return "other"


def baseline_deep(cfg, *, seeds=None) -> dict[str, Any]:
    seeds = tuple(seeds if seeds is not None else cfg.seeds)
    rows: list[dict[str, Any]] = []
    for depth in cfg.depths:
        for seed in seeds:
            task, hidden = C.deep_chain_task(seed, depth)
            gov = ResourceGovernor(global_deadline_s=cfg.fit_deadline_s,
                                   node_budget=cfg.search_node_budget)
            s = Searcher(beam=cfg.search_beam, max_cost=min(cfg.max_cost, depth + 1.0),
                         max_depth=depth, max_nodes=64, governor=gov)
            res = s.search(task, node_budget=cfg.search_node_budget,
                           deadline_s=cfg.fit_deadline_s)
            task.meta["candidates"] = res.candidates
            _r, fields = run_validation(task, cfg)
            rows.append({"depth": depth, "seed": seed,
                         "outcome": _outcome(fields), **fields,
                         "expansions": res.expansions, "elapsed_s": round(res.elapsed_s, 6),
                         "timed_out": bool(res.timed_out),
                         "n_candidates": len(res.candidates)})
    by_depth = {}
    for d in cfg.depths:
        sel = [r for r in rows if r["depth"] == d]
        by_depth[str(d)] = {
            "n": len(sel), "confusion": M.confusion(sel),
            "incorrect_acceptance": M.rate(r["wrong"] for r in sel),
            "correct_acceptance": M.rate(r["correct"] for r in sel),
            "coverage": M.rate(r["accepted"] for r in sel),
            "expansions": M.stats(r["expansions"] for r in sel),
        }
    return {"arm": "baseline", "n": len(rows), "episodes": rows,
            "by_depth": by_depth,
            "incorrect_acceptance_total": sum(1 for r in rows if r["wrong"])}


def baseline_retrieval(cfg, *, seeds=None) -> dict[str, Any]:
    seeds = tuple(seeds if seeds is not None else cfg.seeds)
    # Matched condition: held-out families share input/output types with the
    # learned families, as in EXP-J's silent-substitution probe.
    learned = G.ALL_RULES[:6]
    heldout = G.ALL_RULES[12:]
    store = C.concept_pool(learned)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for rule in heldout:
            task = G.make_task(rule, seed * 19 + 3, n_val=cfg.n_val)
            observed = [(e.x, e.y) for e in task.train + task.validation_pool]
            res = store.retrieve(input_type=task.input_type,
                                 output_type=task.output_type,
                                 observed_outputs=observed, top_k=3)
            silent = bool(res.top) and not res.ambiguous
            rows.append({"seed": seed, "rule": rule, "top": res.top,
                         "ambiguous": bool(res.ambiguous),
                         "request_evidence": bool(res.request_evidence),
                         "silent_substitution": silent,
                         "top_score": round(float(res.confidence), 6),
                         "ranked": [cid for cid, _ in res.ranked]})
    return {"arm": "baseline", "learned_rules": list(learned),
            "heldout_rules": list(heldout), "n": len(rows), "episodes": rows,
            "silent_substitution": M.rate(r["silent_substitution"] for r in rows),
            "ambiguity_rate": M.rate(r["ambiguous"] for r in rows)}


def baseline_drift(cfg, *, seeds=None) -> dict[str, Any]:
    seeds = tuple(seeds if seeds is not None else cfg.seeds)
    rows: list[dict[str, Any]] = []
    by_kind: dict[str, Any] = {}
    for kind in (DriftKind.GRADUAL, DriftKind.ABRUPT):
        missed = 0
        total = 0
        k_rows = []
        for seed in seeds:
            for rule in DRIFT_RULES:
                tasks, d_idx, r_idx = C.drift_stream(rule, seed * 41, kind,
                                                     n_val=cfg.n_val,
                                                     n_test=cfg.n_test)
                flags = []
                for t in tasks:
                    _r, fields = run_validation(t, cfg)
                    flags.append(not fields["accepted"] or fields["wrong"])
                    k_rows.append({"seed": seed, "rule": rule,
                                   "phase": t.meta["phase"],
                                   "accepted": fields["accepted"],
                                   "wrong": fields["wrong"],
                                   "test_accuracy": fields["test_accuracy"]})
                detected = any(flags[d_idx:])
                missed += int(not detected)
                total += 1
        rows.extend(k_rows)
        by_kind[kind.value] = {"n_streams": total, "missed": missed,
                               "miss_rate": round(missed / total, 6) if total else None,
                               "n_episodes": len(k_rows)}
    return {"arm": "baseline", "n": len(rows), "episodes": rows,
            "by_kind": by_kind,
            "miss_rate": round(sum(v["missed"] for v in by_kind.values()) /
                               sum(v["n_streams"] for v in by_kind.values()), 6)}


def _misleading_task(seed: int, rule: str, cfg) -> Task:
    t = G.make_task(rule, seed, n_val=cfg.n_val, n_test=cfg.n_test)
    distractor = t.meta["candidates"][1].program
    train = []
    for ex in t.train:
        try:
            train.append(Example(ex.x, dsl.evaluate(distractor, ex.x)))
        except Exception:
            train.append(Example(ex.x, ex.y))
    return Task(name=t.name + "-misleading", family=t.family, train=train,
                validation_pool=list(t.validation_pool), test=list(t.test),
                input_type=t.input_type, output_type=t.output_type,
                meta=dict(t.meta))


def baseline_calibration(cfg, *, seeds=None) -> dict[str, Any]:
    seeds = tuple(seeds if seeds is not None else cfg.seeds)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for rule in CAL_RULES:
            for shift in ("none", "magnitude"):
                t = G.make_task(rule, seed * 7 + len(shift), shift=shift,
                                n_val=cfg.n_val, n_test=cfg.n_test)
                _r, fields = run_validation(t, cfg)
                rows.append({"seed": seed, "rule": rule, "shift": shift,
                             "confidence": fields["confidence"],
                             "correct": int(fields["correct"]),
                             "accepted": fields["accepted"],
                             "test_accuracy": fields["test_accuracy"]})
    split = max(1, len(rows) // 2)
    fit, ev = rows[:split], rows[split:]
    fit_c = [r["confidence"] for r in fit]
    fit_y = [r["correct"] for r in fit]
    ev_c = [r["confidence"] for r in ev]
    ev_y = [r["correct"] for r in ev]
    return {"arm": "baseline", "n": len(rows), "episodes": rows,
            "ece_independent": M.ece(ev_c, ev_y),
            "ece_pooled": M.ece([r["confidence"] for r in rows],
                                [r["correct"] for r in rows]),
            "brier": M.brier(ev_c, ev_y), "nll": M.nll(ev_c, ev_y),
            "n_fit": len(fit), "n_eval": len(ev),
            "confidence": M.stats(ev_c)}


def baseline_misleading(cfg, *, seeds=None) -> dict[str, Any]:
    seeds = tuple(seeds if seeds is not None else cfg.seeds)
    rows = []
    for seed in seeds:
        for rule in G.ALL_RULES:
            t = _misleading_task(seed, rule, cfg)
            _r, fields = run_validation(t, cfg)
            rows.append({"seed": seed, "rule": rule, **fields})
    return {"arm": "baseline", "n": len(rows), "episodes": rows,
            "wrong_acceptance": M.rate(r["wrong"] for r in rows),
            "correct_acceptance": M.rate(r["correct"] for r in rows),
            "abstention": M.rate(r["abstained"] for r in rows)}


def integrity_snapshot() -> dict[str, Any]:
    out: dict[str, Any] = {"git_head": git_commit(), "git_branch": git_branch(),
                           "manifests": {}}
    for name, path in (("exp_j", EXPJ_BASELINE), ("exp_i", EXPI_BASELINE)):
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            root = path.parents[3]  # repository root
            changed = [f for f, h in data["files"].items()
                       if not (root / f).exists() or sha256_file(root / f) != h]
            out["manifests"][name] = {
                "path": str(path.relative_to(path.parents[3])),
                "n_files": len(data["files"]), "changed": changed,
                "all_unchanged": not changed,
            }
    return out


def run_k0(cfg) -> dict[str, Any]:
    mon = Monitor()
    config = cfg.as_dict()
    manifest = dataset_manifest(config, cfg.seeds)
    save_manifest("k0_dataset_manifest", manifest)
    integ = integrity_snapshot()
    baselines = {
        "deep": baseline_deep(cfg),
        "retrieval": baseline_retrieval(cfg),
        "drift": baseline_drift(cfg),
        "calibration": baseline_calibration(cfg),
        "misleading": baseline_misleading(cfg),
    }
    out = {
        "stage": "K0", "experiment_id": "K0",
        "experiment_name": "Baseline and Reproducibility",
        "status": "completed",
        "config": config,
        "config_fingerprint": fingerprint_of(config),
        "git_commit": git_commit(), "git_branch": git_branch(),
        "integrity": integ,
        "dataset_manifest": {
            "fingerprint": manifest["fingerprint"],
            "n_datasets": manifest["n_datasets"],
            "all_splits_disjoint": manifest["all_splits_disjoint"],
        },
        "seed_registry": {"seeds": list(cfg.seeds), "n_rules": len(G.ALL_RULES),
                          "rules": list(G.ALL_RULES),
                          "fingerprint": manifest["fingerprint"]},
        "baselines": baselines,
        "reproducibility": reproducibility(cfg),
        "targets_from_exp_j": {
            "deep_incorrect_acceptance": "105/280",
            "retrieval_silent_substitution": 1.0,
            "drift_miss_rate": 0.60,
            "independent_ece": 0.159648,
            "misleading_wrong_acceptance": 0.08,
        },
        "limitations": [
            "Independent EXP-K generators are used; results reproduce failure conditions, not identical EXP-J numbers.",
            "Systems under test (EXP-I/EXP-J) are unchanged and imported read-only.",
        ],
    }
    out.update(mon.finish())
    return out
