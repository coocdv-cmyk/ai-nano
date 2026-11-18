"""EXP-I-C — independent, collision-aware concept retrieval.

Stored program bodies are immutable. Every change is a new *version* with an
immutable history. Retrieval never silently substitutes a concept: if two
candidates are close in signature but distinct in behavior, both are returned
with an ambiguity flag and the caller is asked for more evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from . import dsl
from .types import stable_key


@dataclass(frozen=True)
class ConceptRecord:
    id: str
    version: int
    body: Any                            # immutable DSL Node
    input_type: str
    output_type: str
    op_categories: tuple[str, ...]
    params: tuple[Any, ...]
    structural_fp: str
    behavioral_fp: tuple
    context: tuple[str, ...] = ()
    active: bool = True
    created: int = 0
    note: str = ""
    history: tuple[dict[str, Any], ...] = ()

    def signature(self) -> tuple:
        return (self.input_type, self.output_type, self.op_categories,
                tuple(type(p).__name__ for p in self.params))


@dataclass
class RetrievalResult:
    ranked: list[tuple[str, float]] = field(default_factory=list)
    ambiguous: bool = False
    ambiguity_ids: tuple[str, ...] = ()
    confidence: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    request_evidence: bool = False

    @property
    def top(self) -> str | None:
        return self.ranked[0][0] if self.ranked else None


def diagnostic_probes(input_type: str) -> list[Any]:
    if input_type == "int":
        return [-7, -3, -1, 0, 1, 2, 3, 5, 9, 10, 15, 19, 25, 40]
    if input_type == "list":
        return [[], [1], [1, 2, 3], [2, 4, 6], [3, 5, 7], [0, 0, 5],
                [1, 1, 1, 1], [10, 20], [-1, 2, 3]]
    if input_type == "pair":
        return [(0, 0), (2, 3), (7, 4), (-2, 5)]
    return [0, 1, 2]


def _categories(node: Any) -> tuple[str, ...]:
    cats: set[str] = set()

    def walk(n: Any) -> None:
        cats.add(dsl.OPS[n.op].category)
        for c in n.children:
            walk(c)

    walk(node)
    return tuple(sorted(cats))


def _params(node: Any) -> tuple[Any, ...]:
    out: list[Any] = []

    def walk(n: Any) -> None:
        if dsl.OPS[n.op].has_param and n.params:
            out.append(n.params[0])
        for c in n.children:
            walk(c)

    walk(node)
    return tuple(out)


class ConceptStore:
    """Versioned concept memory with parameter-aware, ambiguity-aware retrieval."""

    def __init__(self, *, eval_fn: Callable[[Any, Any], Any] | None = None):
        self.eval_fn = eval_fn or dsl.evaluate
        self.records: dict[str, ConceptRecord] = {}
        self.counter = 0
        self.retired: list[str] = []
        self.aliases: list[dict[str, Any]] = []
        self.collision_log: list[dict[str, Any]] = []

    # ---- writes ------------------------------------------------------------
    def add(self, body: Any, *, input_type: str, output_type: str,
            id: str | None = None, context: Iterable[str] = (),
            note: str = "") -> ConceptRecord:
        self.counter += 1
        cid = id or f"c{self.counter:03d}"
        probes = diagnostic_probes(input_type)
        rec = ConceptRecord(
            id=cid, version=1, body=body, input_type=input_type,
            output_type=output_type, op_categories=_categories(body),
            params=_params(body), structural_fp=dsl.canonical_key(body),
            behavioral_fp=dsl.behavioral_fingerprint(body, probes),
            context=tuple(context), created=self.counter, note=note,
        )
        self.records[cid] = rec
        self._detect_alias(rec)
        return rec

    def refine(self, cid: str, new_body: Any, *, evidence: str = "",
               note: str = "") -> ConceptRecord:
        """Immutable refinement: original body is preserved in ``history``."""
        old = self.records[cid]
        probes = diagnostic_probes(old.input_type)
        snapshot = {
            "version": old.version, "body": old.body,
            "behavioral_fp": old.behavioral_fp, "note": old.note,
            "evidence": evidence,
        }
        new = ConceptRecord(
            id=old.id, version=old.version + 1, body=new_body,
            input_type=old.input_type, output_type=old.output_type,
            op_categories=_categories(new_body), params=_params(new_body),
            structural_fp=dsl.canonical_key(new_body),
            behavioral_fp=dsl.behavioral_fingerprint(new_body, probes),
            context=old.context, created=self.counter, note=note or "refinement",
            history=old.history + (snapshot,),
        )
        self.records[cid] = new
        self._detect_alias(new)
        return new

    def retire(self, cid: str, reason: str = "") -> None:
        if cid in self.records:
            old = self.records[cid]
            self.records[cid] = ConceptRecord(**{**old.__dict__, "active": False,
                                                 "note": f"{old.note}; retired: {reason}"})
            if cid not in self.retired:
                self.retired.append(cid)

    def _detect_alias(self, rec: ConceptRecord) -> None:
        for other in self.records.values():
            if other.id == rec.id or not other.active:
                continue
            if (other.behavioral_fp == rec.behavioral_fp and
                    other.structural_fp != rec.structural_fp):
                self.aliases.append({"a": other.id, "b": rec.id,
                                     "note": "behaviorally identical, structurally distinct"})

    # ---- retrieval ---------------------------------------------------------
    def retrieve(self, *, input_type: str, output_type: str,
                 query_x: Any | None = None, context: Iterable[str] = (),
                 observed_outputs: list[tuple[Any, Any]] | None = None,
                 top_k: int = 3) -> RetrievalResult:
        ctx = set(context)
        active = [r for r in self.records.values() if r.active]
        scored: list[tuple[str, float]] = []
        for r in active:
            type_score = 1.0 if (r.input_type == input_type and
                                 r.output_type == output_type) else 0.0
            cat_overlap = len(set(r.op_categories) & ctx) / max(1, len(ctx or r.op_categories))
            context_score = 1.0 if (ctx & set(r.context)) else 0.0
            behavior_score = 0.0
            if observed_outputs:
                agree = 0
                for x, y in observed_outputs:
                    try:
                        pred = self.eval_fn(r.body, x)
                        if _same(pred, y):
                            agree += 1
                    except Exception:
                        pass
                behavior_score = agree / len(observed_outputs)
            score = 0.5 * behavior_score + 0.2 * type_score + \
                0.15 * context_score + 0.15 * cat_overlap
            scored.append((r.id, score))
        scored.sort(key=lambda t: -t[1])
        scored = scored[:top_k]

        result = RetrievalResult(ranked=scored)
        if len(active) < 2:
            result.confidence = scored[0][1] if scored else 0.0
            return result

        # Ambiguity: top two close in score but behaviorally distinct.
        if len(scored) >= 2:
            a, b = scored[0], scored[1]
            score_gap = a[1] - b[1]
            rec_a, rec_b = self.records[a[0]], self.records[b[0]]
            behavior_distinct = rec_a.behavioral_fp != rec_b.behavioral_fp
            signature_similar = (rec_a.input_type == rec_b.input_type and
                                 rec_a.output_type == rec_b.output_type and
                                 set(rec_a.op_categories) & set(rec_b.op_categories))
            if score_gap <= 0.12 and behavior_distinct and signature_similar:
                result.ambiguous = True
                result.ambiguity_ids = (a[0], b[0])
                result.request_evidence = True
                result.confidence = min(a[1], b[1])
                self.collision_log.append({
                    "a": a[0], "b": b[0], "gap": score_gap,
                    "params_a": rec_a.params, "params_b": rec_b.params,
                    "behavior_distinct": behavior_distinct,
                })
                result.diagnostics = {
                    "collision": self.collision_log[-1],
                    "note": "two plausible but behaviorally distinct concepts; "
                            "requesting further evidence instead of substituting",
                }
                return result
        result.confidence = scored[0][1] if scored else 0.0
        result.diagnostics = {
            "n_active": len(active),
            "alias_count": len(self.aliases),
            "collision_count": len(self.collision_log),
        }
        return result

    # ---- diagnostics -------------------------------------------------------
    def memory_bytes(self) -> int:
        import sys

        total = 0
        for r in self.records.values():
            total += sys.getsizeof(r)
            total += sys.getsizeof(str(r.behavioral_fp))
            total += sys.getsizeof(str(r.history))
        return total

    def report(self) -> dict[str, Any]:
        return {
            "n_records": len(self.records),
            "n_active": sum(1 for r in self.records.values() if r.active),
            "n_retired": len(self.retired),
            "n_aliases": len(self.aliases),
            "n_collisions": len(self.collision_log),
            "memory_bytes": self.memory_bytes(),
            "aliases": self.aliases[:10],
            "collisions": self.collision_log[:10],
        }


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    return a == b
