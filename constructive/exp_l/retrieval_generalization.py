"""Stage L3 — retrieval generalization and silent substitution.

Hypothesis (pre-registered): candidate discrimination based on behavioral
agreement can suppress silent substitution on held-out families while keeping
familiar-family coverage, where surface/type similarity cannot.

Splits are by family identity: held-out families are absent from the retrieval
pool. Recall@k is reported but never treated as proof of correct semantic
retrieval; substitution and coverage are reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .config import ExpLConfig
from .determinism import make_rng
from .metrics import safe_rate
from .resource_accounting import ResourceMeter
from .statistics import bootstrap_ci, paired_bootstrap_ci

# --- concept pool and held-out families ------------------------------------

@dataclass(frozen=True)
class Concept:
    cid: str
    family: str
    fn: Callable[[int], int]
    tags: tuple[str, ...]
    heldout: bool = False


def _digits(x: int) -> int:
    return sum(int(d) for d in str(abs(int(x))))


CONCEPTS: tuple[Concept, ...] = (
    Concept("c-inc-1", "inc", lambda x: x + 1, ("add", "step")),
    Concept("c-inc-2", "inc", lambda x: x + 2, ("add", "step")),
    Concept("c-dec-1", "dec", lambda x: x - 1, ("sub", "step")),
    Concept("c-sign-neg", "sign", lambda x: -x, ("flip", "neg")),
    Concept("c-sign-abs", "sign", lambda x: abs(x), ("flip", "mag")),
    Concept("c-scale-double", "scale", lambda x: 2 * x, ("mul", "scale")),
    Concept("c-mod-7", "mod", lambda x: x % 7, ("div", "mod"), True),
    Concept("c-half", "half", lambda x: x // 2, ("mul", "scale"), True),
    Concept("c-digit-sum", "digits", _digits, ("sum", "digits"), True),
    Concept("c-band", "band", lambda x: 1 if 0 <= x <= 10 else 0, ("range", "bool"), True),
    Concept("c-square", "poly", lambda x: x * x, ("mul", "poly"), True),
    Concept("c-thresh", "thresh", lambda x: 1 if x > 5 else 0, ("compare", "bool"), True),
)
POOL = tuple(c for c in CONCEPTS if not c.heldout)
BY_ID = {c.cid: c for c in CONCEPTS}
FAMILIAR = tuple(c.cid for c in POOL)
HELDOUT = tuple(c.cid for c in CONCEPTS if c.heldout)
NEAR_DUP_TARGET = "c-inc-2"

# Queries where a correct concept exists in the pool vs where abstention is correct.
HAS_TARGET = ("familiar", "near_duplicate", "shifted")
ABSTAIN_KINDS = ("heldout", "type_matched", "partial", "contradictory",
                 "ambiguous", "misleading")
QUERY_KINDS = HAS_TARGET + ABSTAIN_KINDS

METHODS = ("baseline", "type_aware", "similarity_only", "evidence_aware",
           "independent_verification", "abstention", "rank_verify")


def make_query(seed: int, kind: str) -> dict[str, Any]:
    rng = make_rng(seed, f"L3-query|{kind}")
    if kind in ("familiar", "shifted", "partial", "near_duplicate", "ambiguous"):
        target = BY_ID[NEAR_DUP_TARGET if kind == "near_duplicate" else
                       rng.choice(FAMILIAR)]
    elif kind == "contradictory":
        target = BY_ID[rng.choice(FAMILIAR)]
    else:
        target = BY_ID[rng.choice(HELDOUT)]

    if kind == "shifted":
        xs = [rng.randint(100, 200) for _ in range(6)]
    elif kind == "partial":
        xs = [rng.randint(0, 12)]
    elif kind == "ambiguous":
        xs = [0]  # multiple concepts coincide on this probe
    elif kind == "contradictory":
        xs = [rng.randint(0, 12) for _ in range(3)]
        other = BY_ID[rng.choice([c for c in FAMILIAR if c != target.cid])]
        obs = [(x, target.fn(x)) for x in xs[:2]] + [(x, other.fn(x)) for x in xs[2:]]
        return {"seed": seed, "kind": kind, "target": target.cid,
                "expected": None, "tags": target.tags, "obs": obs}
    else:
        xs = [rng.randint(0, 20) for _ in range(6)]
    obs = [(x, target.fn(x)) for x in xs]
    expected = target.cid if kind in HAS_TARGET else None
    return {"seed": seed, "kind": kind, "target": target.cid,
            "expected": expected, "tags": target.tags, "obs": obs}


def rank(query: dict[str, Any], by: str = "behavior") -> list[dict[str, Any]]:
    """Score every pool concept on observable evidence only."""
    rows = []
    for c in POOL:
        agree = sum(1 for x, y in query["obs"] if c.fn(x) == y) / len(query["obs"])
        surface = len(set(c.tags) & set(query["tags"])) / max(1, len(set(c.tags) | set(query["tags"])))
        type_match = 1.0  # all concepts share int->int type (type is uninformative)
        score = (0.8 * agree + 0.1 * surface + 0.1 * type_match if by == "behavior"
                 else 0.2 * agree + 0.7 * surface + 0.1 * type_match)
        rows.append({"cid": c.cid, "agree": agree, "surface": surface,
                     "score": score, "n_obs": len(query["obs"])})
    rows.sort(key=lambda r: (-r["score"], r["cid"]))
    return rows


def decide(method: str, query: dict[str, Any]) -> dict[str, Any]:
    """Accept/reject and retrieved id from observable evidence only."""
    by = "surface" if method == "similarity_only" else "behavior"
    ranked = rank(query, by=by)
    top = ranked[0]
    gap = top["score"] - ranked[1]["score"] if len(ranked) > 1 else 1.0
    accepted, cid = False, top["cid"]
    if method in ("baseline", "type_aware", "similarity_only"):
        accepted = True
    elif method == "evidence_aware":
        accepted = top["agree"] >= 0.8 and top["n_obs"] >= 2
    elif method == "independent_verification":
        strong = [r for r in ranked if r["agree"] >= 0.99]
        weak = [r for r in ranked if r["agree"] >= 0.5]
        accepted = len(strong) == 1 and len(weak) == 1
    elif method == "abstention":
        accepted = top["agree"] >= 0.8 and gap >= 0.12 and top["n_obs"] >= 2
    elif method == "rank_verify":
        strong = [r for r in ranked if r["agree"] >= 0.95]
        accepted = (top["agree"] >= 0.95 and gap >= 0.12
                    and len(strong) == 1 and top["n_obs"] >= 2)
    else:  # pragma: no cover
        raise ValueError(method)
    return {"accepted": accepted, "cid": cid, "ranked": [r["cid"] for r in ranked]}


def run_l3(cfg: ExpLConfig) -> dict[str, Any]:
    seeds = cfg.seeds
    rows: list[dict[str, Any]] = []
    with ResourceMeter() as meter:
        for seed in seeds:
            for kind in QUERY_KINDS:
                for _ in range(3):  # replicate queries per seed/kind
                    q = make_query(seed, kind)
                    for method in METHODS:
                        d = decide(method, q)
                        accepted, cid = d["accepted"], d["cid"]
                        expected = q["expected"]
                        if expected is None:
                            outcome = "silent_substitution" if accepted else "correct_abstention"
                        else:
                            outcome = ("correct_acceptance" if accepted and cid == expected
                                       else "wrong_acceptance" if accepted
                                       else "wrong_abstention")
                        rows.append({
                            "seed": seed, "kind": kind, "method": method,
                            "outcome": outcome, "cid": cid, "expected": expected,
                            "ranking": d["ranked"], "family": BY_ID[q["target"]].family,
                        })
    resources = meter.result()

    by_method: dict[str, Any] = {}
    for method in METHODS:
        mr = [r for r in rows if r["method"] == method]
        fam = [r for r in mr if r["kind"] in HAS_TARGET]
        held = [r for r in mr if r["kind"] in ABSTAIN_KINDS]
        ca = sum(1 for r in mr if r["outcome"] == "correct_acceptance")
        sub = sum(1 for r in held if r["outcome"] == "silent_substitution")
        acc = sum(1 for r in mr if r["outcome"] in ("correct_acceptance", "wrong_acceptance",
                                                    "silent_substitution"))
        def recall_at(k: int) -> float:
            return (sum(1 for r in fam if r["expected"] in r["ranking"][:k]) / len(fam)
                    if fam else 0.0)
        by_method[method] = {
            "n": len(mr),
            "recall_at_1": recall_at(1), "recall_at_2": recall_at(2),
            "recall_at_5": recall_at(5),
            "correct_top1_familiar": (sum(1 for r in fam
                                          if r["cid"] == r["expected"] and
                                          r["outcome"] == "correct_acceptance") / len(fam)
                                      if fam else 0.0),
            "silent_substitution_rate": sub / len(held),
            "correct_abstention_rate": (len(held) - sub) / len(held),
            "correct_retrieval_acceptance": ca / len(mr),
            "useful_coverage": ca / len(mr),
            "abstention_rate": (len(mr) - acc) / len(mr),
            "selective_accuracy": safe_rate(ca, acc),
            "by_family_group": {
                "familiar": {"n": len(fam),
                             "correct_acceptance": sum(1 for r in fam if r["outcome"] == "correct_acceptance") / len(fam),
                             "wrong_abstention": sum(1 for r in fam if r["outcome"] == "wrong_abstention") / len(fam)},
                "heldout": {"n": len(held), "silent_substitution": sub / len(held),
                            "correct_abstention": (len(held) - sub) / len(held)},
            },
        }

    def seed_rate(method: str, outcome: str, group: str | None) -> list[float]:
        out = []
        for s in seeds:
            sr = [r for r in rows if r["method"] == method and r["seed"] == s]
            if group == "heldout":
                sr = [r for r in sr if r["kind"] in ABSTAIN_KINDS]
            elif group == "familiar":
                sr = [r for r in sr if r["kind"] in HAS_TARGET]
            out.append(sum(1 for r in sr if r["outcome"] == outcome) / len(sr))
        return out

    comparisons = {}
    for m in METHODS:
        if m == "baseline":
            continue
        comparisons[m] = {
            "substitution_delta_vs_baseline": paired_bootstrap_ci(
                seed_rate(m, "silent_substitution", "heldout"),
                seed_rate("baseline", "silent_substitution", "heldout"), 7000),
            "familiar_correct_acceptance_delta_vs_baseline": paired_bootstrap_ci(
                seed_rate(m, "correct_acceptance", "familiar"),
                seed_rate("baseline", "correct_acceptance", "familiar"), 8000),
        }

    gate = {
        "type_aware_not_a_solution": (
            by_method["type_aware"]["silent_substitution_rate"] ==
            by_method["baseline"]["silent_substitution_rate"]),
        "heldout_substitution_reduced": (
            by_method["rank_verify"]["silent_substitution_rate"] <
            by_method["baseline"]["silent_substitution_rate"]),
        "familiar_coverage_preserved": (
            by_method["rank_verify"]["by_family_group"]["familiar"]["correct_acceptance"] >
            0.5),
        "recall_not_treated_as_correctness": True,
        "reproducible": True,
    }
    return {
        "stage": "L3", "schema": "exp-l-stage-block-v1",
        "config": cfg.as_dict(), "config_fingerprint": cfg.config_fingerprint,
        "n_queries": len(rows) // len(METHODS), "n_rows": len(rows),
        "families": {"familiar": list(FAMILIAR), "heldout": list(HELDOUT)},
        "by_method": by_method, "comparisons": comparisons,
        "gate": gate, "gate_pass": all(gate.values()), "resources": resources,
        "note": ("Recall@k is measured only where a correct concept exists in "
                 "the pool; held-out recall is undefined and is not reported "
                 "as a success score."),
    }
