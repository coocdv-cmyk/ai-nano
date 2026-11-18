"""EXP-I-A — independent evidence scoring and adaptive observation selection.

The central scientific rule: agreement on observed samples is NOT proof of
semantic correctness. This module measures agreement, distinguishes informative
from redundant observations, finds counterexamples between hypotheses, and
detects contradictions — without ever consulting test labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .types import Candidate, EvidenceRecord, Example

EvalFn = Callable[[Any, Any], Any]   # (program, x) -> y, may raise


def _out(eval_fn: EvalFn, program: Any, x: Any) -> tuple[bool, Any]:
    try:
        return True, eval_fn(program, x)
    except Exception as exc:  # noqa: BLE001 - execution failure is a mismatch
        return False, f"ERR:{type(exc).__name__}"


def score_support(candidates: Iterable[Candidate], support: list[Example],
                  eval_fn: EvalFn) -> dict[str, dict[str, Any]]:
    """Return per-candidate support fit: count, total, outputs, consistent flag."""
    out: dict[str, dict[str, Any]] = {}
    for cand in candidates:
        ok = 0
        outs = []
        for ex in support:
            good, y = _out(eval_fn, cand.program, ex.x)
            outs.append(y)
            if good and _same(y, ex.y):
                ok += 1
        out[cand.id] = {
            "n_ok": ok, "n_total": len(support),
            "consistent": ok == len(support), "outputs": outs,
        }
    return out


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    return a == b


def consistent_ids(scores: dict[str, dict[str, Any]]) -> list[str]:
    return [cid for cid, s in scores.items() if s["consistent"]]


def outputs_at(candidates: Iterable[Candidate], x: Any, eval_fn: EvalFn) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cand in candidates:
        good, y = _out(eval_fn, cand.program, x)
        out[cand.id] = y if good else "ERR"
    return out


def disagreement_at(candidates: Iterable[Candidate], x: Any, eval_fn: EvalFn) -> int:
    """Number of distinct predicted outputs among candidates at x."""
    outs = outputs_at(candidates, x, eval_fn)
    return len({repr(v) for v in outs.values()})


def candidate_disagreement(candidates: Iterable[Candidate], xs: Iterable[Any],
                           eval_fn: EvalFn) -> dict[str, Any]:
    """Distribution of disagreement over a set of inputs."""
    cands = list(candidates)
    per = {_key(x): disagreement_at(cands, x, eval_fn) for x in xs}
    n = len(per)
    disagreeing = sum(1 for v in per.values() if v > 1)
    return {"n_inputs": n, "n_disagreeing": disagreeing,
            "max_disagreement": max(per.values(), default=0), "per_input": per}


def _key(x: Any) -> str:
    from .types import stable_key

    return stable_key(x)


def select_informative(pool: Iterable[Example], candidates: Iterable[Candidate],
                       eval_fn: EvalFn, already_used: set[str] | None = None,
                       surviving: Iterable[str] | None = None) -> Example | None:
    """Query-by-committee: choose the unlabeled observation maximising disagreement.

    Ties break toward inputs not yet used. Returns ``None`` when no candidate
    pair disagrees anywhere in the pool (further observation cannot identify).
    """
    used = set(already_used or ())
    cands = list(candidates)
    surv = set(surviving) if surviving is not None else {c.id for c in cands}
    active = [c for c in cands if c.id in surv]
    if len(active) < 2:
        # Even with one candidate, order by "not yet seen" to keep sampling honest.
        for ex in pool:
            if _key(ex.x) not in used:
                return ex
        return None
    best: Example | None = None
    best_score = -1
    for ex in pool:
        if _key(ex.x) in used:
            continue
        d = disagreement_at(active, ex.x, eval_fn)
        score = d * 100 - int(_key(ex.x) in used)
        if score > best_score:
            best_score = score
            best = ex
    if best is not None and best_score <= 100:  # disagreement == 1 everywhere
        # Return the lowest-preference example only if it is novel; callers can
        # still use it to increase support, flagged as non-discriminating.
        return best
    return best


def counterexamples(program_a: Any, program_b: Any, probes: Iterable[Any],
                    eval_fn: EvalFn, limit: int = 8) -> list[Any]:
    """Inputs where two programs disagree — explicit counterexample generation."""
    out: list[Any] = []
    for x in probes:
        ok_a, a = _out(eval_fn, program_a, x)
        ok_b, b = _out(eval_fn, program_b, x)
        if not (ok_a and ok_b):
            continue
        if not _same(a, b):
            out.append(x)
            if len(out) >= limit:
                break
    return out


def minimize_counterexample(x: Any, program_a: Any, program_b: Any,
                            eval_fn: EvalFn, neighbors: Callable[[Any], Iterable[Any]]) -> Any:
    """Local search toward a smaller disagreement (bounded by neighbor generator)."""
    cur = x
    _, a0 = _out(eval_fn, program_a, cur)
    _, b0 = _out(eval_fn, program_b, cur)
    if _same(a0, b0):
        return cur
    for cand in neighbors(cur):
        ok_a, a = _out(eval_fn, program_a, cand)
        ok_b, b = _out(eval_fn, program_b, cand)
        if ok_a and ok_b and not _same(a, b):
            cur = cand
    return cur


def contradictions_in_support(support: list[Example]) -> list[dict[str, Any]]:
    """Find identical inputs carrying different labels — direct evidence corruption."""
    seen: dict[str, Any] = {}
    out: list[dict[str, Any]] = []
    for ex in support:
        k = _key(ex.x)
        if k in seen and not _same(seen[k], ex.y):
            out.append({"x": ex.x, "labels": [seen[k], ex.y]})
        else:
            seen[k] = ex.y
    return out


def blind_positive_risk(support: list[Example]) -> dict[str, Any]:
    """Detect support with no positive evidence for a boolean target."""
    labels = [ex.y for ex in support]
    bools = [v for v in labels if isinstance(v, bool)]
    if not bools:
        return {"applicable": False, "n_positive": 0, "at_risk": False}
    n_pos = sum(1 for v in bools if v)
    return {"applicable": True, "n_positive": n_pos,
            "at_risk": n_pos == 0,
            "note": "no positive examples means an all-negative rule is unfalsified"}


def provenance_for_selection(ex: Example, eliminated: list[str], *,
                             source: str = "validation_pool", note: str = "") -> EvidenceRecord:
    return EvidenceRecord(
        x=ex.x, y=ex.y, source=source, informative=bool(eliminated),
        eliminated=tuple(eliminated), note=note,
    )


def redundancy_report(evidence: list[EvidenceRecord]) -> dict[str, Any]:
    informative = [e for e in evidence if e.informative]
    redundant = [e for e in evidence if not e.informative]
    return {
        "n_observations": len(evidence),
        "n_informative": len(informative),
        "n_redundant": len(redundant),
        "informative_inputs": [e.x for e in informative],
        "redundant_inputs": [e.x for e in redundant],
        "note": "redundant observations confirm but cannot discriminate hypotheses",
    }


@dataclass
class EvidenceBundle:
    scores: dict[str, dict[str, Any]]
    surviving_ids: list[str]
    evidence: list[EvidenceRecord]
    disagreement: dict[str, Any]
    contradictions: list[dict[str, Any]]
    blind_positive: dict[str, Any]
