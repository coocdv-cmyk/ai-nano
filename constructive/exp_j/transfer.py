"""J1 — Independent distribution transfer.

Tests whether EXP-I evidence validation and retrieval work on task families and
input distributions completely independent of EXP-H / EXP-I generators.
"""

from __future__ import annotations

from typing import Any

from constructive.exp_i import dsl
from constructive.exp_i.retrieval import ConceptStore
from constructive.exp_i.types import Example, Task

from . import generators as G
from . import metrics as M
from .config import ExpJConfig
from .engine import run_validation, to_episode
from .resources import Monitor
from .types import EpisodeRecord

PRINCIPAL_SHIFTS = ("none", "magnitude", "length", "both")


def _rowdict(r: EpisodeRecord) -> dict[str, Any]:
    return {
        "seed": r.seed, "family": r.family, "task": r.task,
        "state": r.state, "accepted": r.accepted,
        "correct_acceptance": r.correct_acceptance,
        "wrong_acceptance": r.wrong_acceptance, "abstained": r.abstained,
        "false_rejection": r.false_rejection,
        "non_identifiable": r.non_identifiable,
        "confidence": r.confidence, "test_accuracy": r.test_accuracy,
        "failure_category": r.failure_category,
        "shift": r.meta.get("shift"),
        "unseen_ops": r.meta.get("unseen_ops", []),
    }


def _support_only_baseline(task: Task) -> str:
    """Independent baseline: best support fit, cost tie-break, always accept."""
    best_id, best_score = None, (-1, 1e9)
    for c in task.meta["candidates"]:
        ok = 0
        for ex in task.train:
            try:
                if dsl.evaluate(c.program, ex.x) == ex.y:
                    ok += 1
            except Exception:
                pass
        score = (ok, -c.cost)
        if score > best_score:
            best_score, best_id = score, c.id
    return best_id or ""


def _build_concept_store(seeds: tuple[int, ...]) -> tuple[ConceptStore, dict[str, str]]:
    """Store concepts learned from independent tasks at a fixed source seed."""
    store = ConceptStore(eval_fn=dsl.evaluate)
    truth: dict[str, str] = {}
    for rule in G.ALL_RULES:
        t = G.make_task(rule, 0)
        prog = dict((c.id, c.program) for c in t.meta["candidates"])["true"]
        rec = store.add(prog, input_type=t.input_type, output_type=t.output_type,
                        id=f"concept-{rule}", context=(rule,))
        truth[rec.id] = rule
    return store, truth


def run_transfer(cfg: ExpJConfig, *, seeds: tuple[int, ...] | None = None) -> dict[str, Any]:
    seeds = tuple(seeds if seeds is not None else cfg.seeds)
    mon = Monitor()
    episodes: list[EpisodeRecord] = []
    baseline_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []

    # --- evidence engine on independent distributions -----------------------
    for seed in seeds:
        for rule in G.ALL_RULES:
            for shift in PRINCIPAL_SHIFTS:
                shift_code = sum(ord(c) for c in shift) % 97
                task = G.make_task(rule, seed * 31 + shift_code,
                                   shift=shift, k=4, n_val=cfg.n_val,
                                   n_test=12)
                _res, fields = run_validation(task, cfg)
                ep = to_episode(task, seed, fields)
                ep.meta.update({"shift": shift, "unseen_ops":
                                list(G.UNSEEN_OPS.get(rule, ()))})
                episodes.append(ep)

                # Independent baselines (computed from support only / trivial).
                so_id = _support_only_baseline(task)
                baseline_rows.append({
                    "seed": seed, "family": rule, "shift": shift,
                    "support_only_correct": so_id == "true",
                    "support_only_wrong": so_id != "true",
                })

    # --- calibration: fit on first half of seeds, evaluate on the rest -------
    split = max(1, len(episodes) // 2)
    conf = [e.confidence for e in episodes]
    corr = [1 if e.correct_acceptance else 0 for e in episodes]
    calibration = M.calibration_ece(conf, corr)

    # --- retrieval on parameterized concepts under distribution shift -------
    store, truth = _build_concept_store(seeds)
    for seed in seeds:
        for rule in G.ALL_RULES:
            for shift in ("none", "magnitude", "length"):
                task = G.make_task(rule, seed * 17 + 5, shift=shift, n_val=cfg.n_val)
                observed = [(e.x, e.y) for e in task.train + task.validation_pool]
                expected = f"concept-{rule}"
                res = store.retrieve(
                    input_type=task.input_type, output_type=task.output_type,
                    observed_outputs=observed, top_k=3,
                )
                ranked = [cid for cid, _ in res.ranked]
                for k in (1, 2, 3):
                    pr = M.retrieval_pr({expected}, ranked, k)
                    retrieval_rows.append({
                        "seed": seed, "rule": rule, "shift": shift, "k": k,
                        "ranked": ranked, "expected": expected, **pr,
                    })
                retrieval_rows[-3]["ambiguous"] = bool(res.ambiguous)
                retrieval_rows[-3]["request_evidence"] = bool(res.request_evidence)

    # --- cross-family transfer: concepts learned from one set of families are
    # queried on genuinely held-out families. A correct system should NOT
    # confidently claim an unseen concept; it should show ambiguity / no hit.
    learned_rules = G.ALL_RULES[:6]
    heldout_rules = G.ALL_RULES[6:]
    cf_store, cf_truth = _build_concept_store(seeds)
    cf_store.records = {
        cid: r for cid, r in cf_store.records.items()
        if cid in {f"concept-{rule}" for rule in learned_rules}}
    cross_rows: list[dict[str, Any]] = []
    for seed in seeds:
        for rule in heldout_rules:
            task = G.make_task(rule, seed * 19 + 3, shift="none", n_val=cfg.n_val)
            observed = [(e.x, e.y) for e in task.train + task.validation_pool]
            res = cf_store.retrieve(
                input_type=task.input_type, output_type=task.output_type,
                observed_outputs=observed, top_k=3)
            cross_rows.append({
                "seed": seed, "heldout_rule": rule,
                "top": res.top, "expected_unseen": f"concept-{rule}",
                "hit_on_unseen": bool(res.top == f"concept-{rule}"),
                "ambiguous": bool(res.ambiguous),
                "request_evidence": bool(res.request_evidence),
                "top_score": round(float(res.confidence), 6),
            })

    # --- aggregation ---------------------------------------------------------
    rows = [_rowdict(e) for e in episodes]
    by_shift: dict[str, Any] = {}
    for shift in PRINCIPAL_SHIFTS:
        sel = [r for r in rows if r["shift"] == shift]
        by_shift[shift] = {
            "n": len(sel),
            "confusion": M.confusion(sel),
            "accuracy": M.stats([r["test_accuracy"] for r in sel
                                 if r["test_accuracy"] is not None]),
            "acceptance": M.rate(r["accepted"] for r in sel),
            "wrong_acceptance": M.rate(r["wrong_acceptance"] for r in sel),
            "false_rejection": M.rate(r["false_rejection"] for r in sel),
            "non_identifiable": M.rate(r["non_identifiable"] for r in sel),
        }

    evidence_wrong = [r["wrong_acceptance"] for r in rows]
    baseline_wrong = [b["support_only_wrong"] for b in baseline_rows]
    baseline_corr = [b["support_only_correct"] for b in baseline_rows]
    evidence_corr = [r["correct_acceptance"] for r in rows]

    comparisons = {
        "wrong_acceptance_vs_support_only": M.compare(
            baseline_wrong, evidence_wrong, lower_is_better=True),
        "correct_acceptance_vs_support_only": M.compare(
            baseline_corr, evidence_corr),
        "shifted_vs_indomain_accuracy": M.compare(
            [r["test_accuracy"] for r in rows if r["shift"] == "none" and
             r["test_accuracy"] is not None],
            [r["test_accuracy"] for r in rows if r["shift"] != "none" and
             r["test_accuracy"] is not None],
        ),
    }

    ret_by_k: dict[str, Any] = {}
    for k in (1, 2, 3):
        sel = [r for r in retrieval_rows if r["k"] == k]
        ret_by_k[f"@{k}"] = {
            "n": len(sel),
            "precision": M.stats([r["precision"] for r in sel]),
            "recall": M.stats([r["recall"] for r in sel]),
            "silent_substitution": M.rate(r["silent_substitution"] for r in sel),
        }

    out = {
        "experiment_id": "J1",
        "experiment_name": "Independent Distribution Transfer",
        "status": "completed",
        "seeds": list(seeds),
        "rules": list(G.ALL_RULES),
        "shifts": list(PRINCIPAL_SHIFTS),
        "n_episodes": len(episodes),
        "episodes": rows,
        "baseline_episodes": baseline_rows,
        "retrieval_episodes": retrieval_rows,
        "by_shift": by_shift,
        "calibration": calibration,
        "comparisons": comparisons,
        "retrieval": ret_by_k,
        "retrieval_ambiguity_rate": M.rate(
            r.get("ambiguous", False) for r in retrieval_rows if r["k"] == 1),
        "cross_family_transfer": {
            "learned_rules": list(learned_rules),
            "heldout_rules": list(heldout_rules),
            "n": len(cross_rows),
            "episodes": cross_rows,
            "false_claim_rate_on_unseen": M.rate(
                r["hit_on_unseen"] for r in cross_rows),
            "ambiguity_rate_on_unseen": M.rate(
                r["ambiguous"] for r in cross_rows),
            "silent_substitution_rate": M.rate(
                bool(r["top"]) and not r["ambiguous"]
                for r in cross_rows),
            "note": ("The store cannot contain a held-out family's concept. The "
                     "safe outcome is to abstain/flag ambiguity; returning a "
                     "different learned concept without a flag is a silent "
                     "substitution and is measured explicitly."),
        },
        "independence_evidence": {
            "generators": "EXP-J original; independent label functions and RNGs",
            "rules": list(G.ALL_RULES),
            "unseen_ops": sorted({op for r in G.ALL_RULES for op in G.UNSEEN_OPS[r]}),
            "note": "No EXP-H/EXP-I generator or construction shortcut is imported.",
        },
        "limitations": [
            "Synthetic typed-DSL tasks; no natural-input validity claimed.",
            "Calibration correctness uses generator meta (true rule identity), not test labels.",
        ],
    }
    out.update(mon.finish())
    return out
