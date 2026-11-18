"""K2 — OOD retrieval and silent-substitution investigation.

Matched arms test whether explicit compatibility, OOD rejection, ambiguity
abstention or independent verification stop a retrieval system from treating a
top-ranked candidate as automatically correct. Familiar, held-out-family,
near-neighbor, incomplete and contradictory queries are evaluated together.
"""

from __future__ import annotations

import time
from typing import Any

from constructive.exp_i import dsl

from . import constructors as C
from . import generators as G
from . import metrics as M
from .config import ExpKConfig
from .resources import Monitor
from .types import Completion, RetrievalOutcome

FAMILIAR = G.ALL_RULES[:6]
HELDOUT = G.ALL_RULES[12:]

ARMS = ("baseline", "type_aware", "ood_rejection", "ambiguity_abstention",
        "provenance_compatible", "independent_verification")


def _queries(seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seed in seeds:
        for rule in FAMILIAR:
            t = G.make_task(rule, seed * 13 + 5, n_val=5, n_test=8)
            out.append({
                "seed": seed, "rule": rule, "kind": "familiar",
                "expected": f"c-{rule}", "input_type": t.input_type,
                "output_type": t.output_type,
                "observed": [(e.x, e.y) for e in t.train + t.validation_pool],
            })
        for rule in HELDOUT:
            t = G.make_task(rule, seed * 17 + 7, n_val=5, n_test=8)
            out.append({
                "seed": seed, "rule": rule, "kind": "heldout",
                "expected": None, "input_type": t.input_type,
                "output_type": t.output_type,
                "observed": [(e.x, e.y) for e in t.train + t.validation_pool],
            })
        # Incomplete specification: a single observation for a known family.
        rule = FAMILIAR[seed % len(FAMILIAR)]
        t = G.make_task(rule, seed * 23 + 3, n_val=5, n_test=8)
        out.append({
            "seed": seed, "rule": rule, "kind": "incomplete",
            "expected": f"c-{rule}", "input_type": t.input_type,
            "output_type": t.output_type,
            "observed": [(e.x, e.y) for e in t.train[:1]],
        })
        # Contradictory description: labels from two different families.
        a = G.make_task(FAMILIAR[0], seed * 3 + 1, n_val=5)
        b = G.make_task(FAMILIAR[1], seed * 3 + 2, n_val=5)
        out.append({
            "seed": seed, "rule": f"{FAMILIAR[0]}+{FAMILIAR[1]}",
            "kind": "contradictory", "expected": None,
            "input_type": a.input_type, "output_type": a.output_type,
            "observed": ([(e.x, e.y) for e in a.train[:2]] +
                         [(e.x, e.y) for e in b.train[:2]]),
        })
    return out


def _agreement(body, observed) -> float:
    ok = 0
    for x, y in observed:
        try:
            ok += int(dsl.evaluate(body, x) == y)
        except Exception:
            pass
    return ok / max(1, len(observed))


def _decide(arm, result, queries, store):
    """Return (accepted, top_id, flags) under the given safeguard arm."""
    top = result.top
    ranked = [cid for cid, _ in result.ranked]
    if not ranked:
        return False, None, {"no_candidate": True}
    top_rec = store.records[top]
    compat = (top_rec.input_type == queries["input_type"] and
              top_rec.output_type == queries["output_type"])
    agree = _agreement(top_rec.body, queries["observed"])
    if arm == "baseline":
        return True, top, {"compat": compat, "agree": agree}
    if arm == "type_aware":
        return (compat, top if compat else None, {"compat": compat, "agree": agree})
    if arm == "ood_rejection":
        return (compat and agree >= 0.8, top if compat and agree >= 0.8 else None,
                {"compat": compat, "agree": agree})
    if arm == "ambiguity_abstention":
        ok = compat and not result.ambiguous and agree >= 0.8
        return ok, top if ok else None, {"compat": compat, "agree": agree}
    if arm == "provenance_compatible":
        ok = (compat and agree >= 0.8 and not result.ambiguous and
              bool(set(top_rec.context) & {queries["rule"]}))
        return ok, top if ok else None, {"compat": compat, "agree": agree}
    if arm == "independent_verification":
        verified = []
        for cid in ranked:
            rec = store.records[cid]
            if (rec.input_type == queries["input_type"] and
                    rec.output_type == queries["output_type"] and
                    _agreement(rec.body, queries["observed"]) >= 0.99):
                verified.append(cid)
        ok = len(verified) == 1
        return ok, verified[0] if ok else None, {
            "compat": compat, "agree": agree, "n_verified": len(verified)}
    raise ValueError(arm)


def run_k2(cfg: ExpKConfig, *, seeds=None, arms=ARMS) -> dict[str, Any]:
    seeds = tuple(seeds if seeds is not None else cfg.seeds)
    mon = Monitor()
    queries = _queries(seeds)
    rows: list[dict[str, Any]] = []
    for arm in arms:
        store = C.concept_pool(FAMILIAR)
        for q in queries:
            t0 = time.perf_counter()
            res = store.retrieve(input_type=q["input_type"],
                                 output_type=q["output_type"],
                                 observed_outputs=q["observed"], top_k=3)
            latency = time.perf_counter() - t0
            accepted, top, flags = _decide(arm, res, q, store)
            expected = q["expected"]
            if q["kind"] == "familiar":
                outcome = (RetrievalOutcome.CORRECT.value if accepted and top == expected
                           else RetrievalOutcome.WRONG.value if accepted
                           else RetrievalOutcome.WRONG_ABSTENTION.value)
            else:  # heldout / incomplete / contradictory: correct action is abstain
                outcome = (RetrievalOutcome.CORRECT_ABSTENTION.value if not accepted
                           else RetrievalOutcome.WRONG.value)
            ranked = [cid for cid, _ in res.ranked]
            rows.append({
                "arm": arm, "seed": q["seed"], "rule": q["rule"],
                "kind": q["kind"], "expected": expected, "outcome": outcome,
                "accepted": accepted, "top": top, "ranked": ranked,
                "ambiguous": bool(res.ambiguous),
                "latency_ms": round(latency * 1000, 4),
                "diagnostic": flags,
            })

    by_arm: dict[str, Any] = {}
    for arm in arms:
        sel = [r for r in rows if r["arm"] == arm]
        fam = [r for r in sel if r["kind"] == "familiar"]
        ood = [r for r in sel if r["kind"] in ("heldout", "contradictory")]
        inc = [r for r in sel if r["kind"] == "incomplete"]
        ranked_ok1 = [r["ranked"][0] == r["expected"] for r in fam if r["ranked"]]
        ranked_ok2 = [r["expected"] in r["ranked"][:2] for r in fam]
        by_arm[arm] = {
            "n": len(sel),
            "confusion": M.confusion(sel),
            "recall_at_1": M.rate(ranked_ok1),
            "recall_at_2": M.rate(ranked_ok2),
            "precision_at_1": M.rate(ranked_ok1),
            "precision_at_2": M.rate(ranked_ok2),
            "correct_top_familiar": M.rate(
                r["outcome"] == RetrievalOutcome.CORRECT.value for r in fam),
            "silent_substitution_ood": M.rate(
                r["accepted"] for r in ood),
            "correct_abstention_ood": M.rate(
                r["outcome"] == RetrievalOutcome.CORRECT_ABSTENTION.value for r in ood),
            "correct_abstention_incomplete": M.rate(
                r["outcome"] == RetrievalOutcome.CORRECT_ABSTENTION.value for r in inc),
            "wrong_abstention_familiar": M.rate(
                r["outcome"] == RetrievalOutcome.WRONG_ABSTENTION.value for r in fam),
            "ambiguous_query_detection": M.rate(r["ambiguous"] for r in ood),
            "latency_ms": M.stats(r["latency_ms"] for r in sel),
        }
    # Paired OOD comparison by (seed, rule).
    base_key = {(r["seed"], r["rule"], r["kind"]): r["accepted"]
                for r in rows if r["arm"] == "baseline"}
    comparisons = {}
    for arm in arms:
        if arm == "baseline":
            continue
        arm_key = {(r["seed"], r["rule"], r["kind"]): r["accepted"]
                   for r in rows if r["arm"] == arm}
        shared = sorted(set(base_key) & set(arm_key))
        comparisons[arm] = M.compare([base_key[k] for k in shared],
                                     [arm_key[k] for k in shared],
                                     lower_is_better=True)
        comparisons[arm]["n_paired"] = len(shared)

    out = {
        "stage": "K2", "experiment_id": "K2",
        "experiment_name": "OOD Retrieval and Silent Substitution",
        "status": "completed", "seeds": list(seeds),
        "familiar_rules": list(FAMILIAR), "heldout_rules": list(HELDOUT),
        "arms": list(arms), "n_queries": len(queries), "n_runs": len(rows),
        "episodes": rows, "by_arm": by_arm,
        "comparison_to_baseline": comparisons,
        "critical_rule": ("The top-ranked candidate is never automatically correct; "
                          "when no candidate can be justified the system must abstain."),
        "limitations": ["Held-out families are generated independently of the concept pool."],
    }
    out.update(mon.finish())
    return out
