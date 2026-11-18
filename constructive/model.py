"""The constructive learner: search, concept formation, uncertainty, repair.

The model is a concept graph plus a very small learned retrieval prior. It is
not an agent and it calls no external service. Learning is local: a new concept
is appended with its own body, validation set and confidence; existing concepts
are never retrained.

Epistemics
----------
Each induced program is a hypothesis, not a fact.  Three independent filters
decide whether a hypothesis becomes a concept:

1.  **fit** - it must reproduce every training output;
2.  **probe separation** - hypotheses that agree on the training inputs but
    disagree on unlabelled probe inputs are kept as *distinct* hypotheses, so
    disagreement is visible instead of being hidden by deduplication;
3.  **held-out validation** - only hypotheses that also reproduce a held-out
    example are accepted.  Everything else is stored as an unvalidated
    hypothesis and never used for prediction.

That makes "it fits the examples" insufficient by construction.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from .core import (
    EvalError,
    Prim,
    Scene,
    Term,
    V,
    C,
    L,
    build_prims,
    eval_term,
    infer_type,
    same_value,
    term_cost,
    tree_edit_distance,
    value_key,
    well_typed,
)

UNDEF = object()


# ---------------------------------------------------------------------------
# Concept
# ---------------------------------------------------------------------------


@dataclass
class Concept:
    name: str
    params: tuple[str, ...]
    body: Term
    ret: str
    input_type: str = "any"
    arg_types: tuple[str, ...] = ()
    guard: Term | None = None
    confidence: float = 0.5
    support: int = 0
    uses: int = 0
    created: int = 0
    embedding: tuple[float, ...] = ()
    validation: list[tuple[Any, Any]] = field(default_factory=list)
    parent: str | None = None
    note: str = ""

    def apply(self, args: list[Any], prims: dict[str, Prim], concepts: dict[str, "Concept"]) -> Any:
        if len(args) != len(self.params):
            raise EvalError(f"{self.name}: expected {len(self.params)} args, got {len(args)}")
        env = {p: a for p, a in zip(self.params, args)}
        return eval_term(self.body, env, prims, concepts)

    def accepts(self, x: Any, prims: dict[str, Prim], concepts: dict[str, "Concept"]) -> bool:
        if self.guard is None:
            return True
        try:
            return bool(eval_term(self.guard, {"x": x}, prims, concepts))
        except EvalError:
            return False

    def stored_nodes(self) -> int:
        return self.body.size() + (self.guard.size() if self.guard else 0)


def embed_term(t: Term, dim: int = 16) -> tuple[float, ...]:
    """Fixed-width hashed bag-of-operators embedding of a program."""
    vec = [0.0] * dim
    for s in _walk(t):
        if s.op in ("var", "lit"):
            continue
        h = int(hashlib.sha256(s.op.encode("utf-8")).hexdigest()[:8], 16) % dim
        vec[h] += 1.0
    n = math.sqrt(sum(v * v for v in vec)) or 1.0
    return tuple(v / n for v in vec)


def _walk(t: Term) -> Iterable[Term]:
    yield t
    for a in t.args:
        yield from _walk(a)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    term: Term
    outputs: tuple[Any, ...]
    cost: int
    n_ok: int
    variation: int
    probes: tuple[Any, ...] = ()
    posterior: float = 0.0
    validation_accuracy: float | None = None
    alts: tuple[Term, ...] = ()
    key: tuple = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "term": str(self.term),
            "cost": self.cost,
            "n_ok": self.n_ok,
            "variation": self.variation,
            "posterior": round(self.posterior, 6),
            "validation_accuracy": self.validation_accuracy,
        }


class Searcher:
    """Type-directed bottom-up enumeration with observational deduplication."""

    def __init__(self, prims: dict[str, Prim], concepts: dict[str, Concept], beam: int = 45,
                 max_nodes: int = 90_000, time_budget_s: float = 2.0):
        self.prims = prims
        self.concepts = concepts
        self.beam = beam
        self.max_nodes = max_nodes
        self.time_budget_s = time_budget_s
        self.nodes = 0
        self.timed_out = False
        self.levels: dict[tuple[str, int], list[Candidate]] = {}

    # -- public ---------------------------------------------------------
    def search(self, examples: list[tuple[Any, Any]], input_type: str, target_type: str,
               max_cost: int = 4, eps: float = 0.0, beta: float = 1.0,
               top_k: int = 8, probes: list[Any] | None = None) -> tuple[list[Candidate], int]:
        """Bottom-up enumerative synthesis.

        Programs are built from the *values* of their subprograms instead of
        re-evaluating syntax trees, so the cost of extending a program is one
        shallow application per example, not a full recursive re-evaluation.
        """
        t0 = time.perf_counter()
        inputs = [x for x, _ in examples]
        targets = [y for _, y in examples]
        probes = list(probes or [])
        n = len(examples)
        n_in = len(inputs)
        total = n_in + len(probes)

        def vals_at(e: Candidate, i: int) -> Any:
            return e.outputs[i] if i < n_in else e.probes[i - n_in]

        def lift1(fn: Any, e: Candidate) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
            out: list[Any] = []
            for i in range(total):
                v = vals_at(e, i)
                if v is UNDEF:
                    out.append(UNDEF)
                    continue
                try:
                    out.append(fn(v))
                except Exception:
                    out.append(UNDEF)
            return tuple(out[:n_in]), tuple(out[n_in:])

        def lift2(fn: Any, a: Candidate, b: Candidate) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
            out: list[Any] = []
            for i in range(total):
                x = vals_at(a, i)
                y = vals_at(b, i)
                if x is UNDEF or y is UNDEF:
                    out.append(UNDEF)
                    continue
                try:
                    out.append(fn(x, y))
                except Exception:
                    out.append(UNDEF)
            return tuple(out[:n_in]), tuple(out[n_in:])

        def n_ok(vals: tuple[Any, ...]) -> int:
            return sum(1 for v, y in zip(vals, targets) if v is not UNDEF and same_value(v, y))

        def variation(vals: tuple[Any, ...]) -> int:
            return len({("U",) if v is UNDEF else value_key(v) for v in vals})

        levels: dict[tuple[str, int], list[Candidate]] = defaultdict(list)
        seen: dict[tuple, list[Term]] = {}
        solutions: list[Candidate] = []

        # Type relevance: a partial program is worth keeping only if its type
        # can still reach the target type within the remaining cost budget.
        useful: dict[int, set[str]] = {0: {target_type}}
        for d in range(1, max_cost + 1):
            acc = set(useful[d - 1])
            for p in self.prims.values():
                if p.ret in acc:
                    acc.update(p.args)
            for c in self.concepts.values():
                if c.ret in acc:
                    acc.update(c.arg_types or tuple("any" for _ in c.params))
            useful[d] = acc

        # Intermediate levels are kept broad (dedup already bounds them);
        # only the final level, where the answer must appear, uses the beam.
        mid_cap = max(self.beam, 4000)

        def add(term: Term, vals: tuple[Any, ...], pvals: tuple[Any, ...],
                typ: str, cost: int) -> None:
            if cost >= max_cost and typ != target_type:
                return
            if cost < max_cost and typ != "any" and typ not in useful[max_cost - cost]:
                return
            key = (
                typ,
                tuple(("U",) if v is UNDEF else value_key(v) for v in vals),
                tuple(("U",) if v is UNDEF else value_key(v) for v in pvals),
            )
            if key in seen:
                group = seen[key]
                if len(group) < 8:
                    group.append(term)
                return
            seen[key] = [term]
            c = Candidate(term, vals, cost, n_ok(vals), variation(vals), pvals, key=key)
            levels[(typ, cost)].append(c)

        # seeds: the variable, small integer constants, booleans
        add(V("x"), tuple(inputs), tuple(probes), input_type, 0)
        for lit in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1):
            typ = "bool" if isinstance(lit, bool) else "int"
            add(L(lit), (lit,) * n_in, (lit,) * len(probes), typ, 1)
        for lit in (True, False):
            add(L(lit), (lit,) * n_in, (lit,) * len(probes), "bool", 1)

        def budget_left() -> bool:
            if self.nodes >= self.max_nodes:
                return False
            if time.perf_counter() - t0 > self.time_budget_s:
                self.timed_out = True
                return False
            return True

        for cost in range(1, max_cost + 1):
            # --- learned concepts as single units ---------------------------
            for c in list(self.concepts.values()):
                if not budget_left():
                    break
                if c.ret not in useful[max_cost - cost]:
                    continue
                atypes = c.arg_types or tuple("any" for _ in c.params)
                fn = (lambda v, _c=c: _c.apply([v], self.prims, self.concepts))
                if len(c.params) == 1:
                    for e in list(levels.get((atypes[0], cost - 1), [])):
                        self.nodes += 1
                        vals, pvals = lift1(fn, e)
                        add(C(c.name, e.term), vals, pvals, c.ret, cost)
                elif len(c.params) == 2:
                    for c1 in range(0, cost - 1):
                        c2 = cost - 1 - c1
                        for a in levels.get((atypes[0], c1), [])[: self.beam]:
                            if not budget_left():
                                break
                            for b in levels.get((atypes[1], c2), [])[: self.beam]:
                                if not budget_left():
                                    break
                                self.nodes += 1
                                vals, pvals = lift2(
                                    lambda x, y, _c=c: _c.apply([x, y], self.prims, self.concepts),
                                    a, b)
                                add(C(c.name, a.term, b.term), vals, pvals, c.ret, cost)

            # --- primitives -------------------------------------------------
            for prim in self.prims.values():
                if not budget_left():
                    break
                if prim.ret not in useful[max_cost - cost]:
                    continue
                if len(prim.args) == 1:
                    for e in list(levels.get((prim.args[0], cost - prim.cost), [])):
                        self.nodes += 1
                        vals, pvals = lift1(prim.fn, e)
                        add(Term(prim.name, (e.term,)), vals, pvals, prim.ret, cost)
                elif len(prim.args) == 2:
                    for c1 in range(0, cost - prim.cost):
                        c2 = cost - prim.cost - c1
                        for a in levels.get((prim.args[0], c1), [])[: self.beam]:
                            if not budget_left():
                                break
                            for b in levels.get((prim.args[1], c2), [])[: self.beam]:
                                if not budget_left():
                                    break
                                self.nodes += 1
                                vals, pvals = lift2(prim.fn, a, b)
                                add(Term(prim.name, (a.term, b.term)), vals, pvals, prim.ret, cost)

            # attach deduplicated alternatives (same behaviour on train+probes,
            # possibly different behaviour on held-out selection data)
            for bucket in levels.values():
                for cand in bucket:
                    if not cand.alts and cand.key in seen:
                        cand.alts = tuple(seen[cand.key][1:8])

            # --- prune each (type, cost) bucket ----------------------------
            for key in list(levels.keys()):
                bucket = levels[key]
                cap = self.beam if key[1] >= max_cost else mid_cap
                if len(bucket) > cap:
                    bucket.sort(key=lambda e: (-int(e.n_ok == n), -e.variation, -e.n_ok, str(e.term)))
                    levels[key] = bucket[:cap]

            # --- stop at the first cost with an exact solution --------------
            exact = [e for e in levels.get((target_type, cost), []) if e.n_ok == n]
            if exact:
                solutions = exact
                break

        if not solutions:
            pool = []
            for (typ, _c), bucket in levels.items():
                if typ != target_type:
                    continue
                pool.extend(bucket)
            pool.sort(key=lambda e: (-e.n_ok, e.cost, -e.variation, str(e.term)))
            solutions = [e for e in pool if e.n_ok >= max(1, int(math.ceil(0.5 * n)))][:top_k]
        else:
            solutions.sort(key=lambda e: (e.cost, str(e.term)))
            solutions = solutions[:top_k]

        if solutions:
            k_out = max(2, len({value_key(y) for y in targets}))
            logs = []
            for cand in solutions:
                ll = 0.0
                for v, y in zip(cand.outputs, targets):
                    if v is UNDEF:
                        ll += math.log(max(eps, 1e-9))
                    elif same_value(v, y):
                        ll += math.log(max(1.0 - eps, 1e-9))
                    else:
                        ll += math.log(max(eps / max(1, k_out - 1), 1e-9))
                logs.append(ll - beta * cand.cost)
            m = max(logs)
            ws = [math.exp(v - m) for v in logs]
            ssum = sum(ws) or 1.0
            for cand, w in zip(solutions, ws):
                cand.posterior = w / ssum
        self.levels = levels
        return solutions, self.nodes


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


@dataclass
class FitReport:
    concept: str
    accepted: bool
    abstained: bool
    confidence: float
    margin: float
    cost: int
    candidates: list[str]
    nodes: int
    elapsed_s: float
    partial: bool
    validation_accuracy: float | None
    reason: str = ""
    timed_out: bool = False
    quarantined: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["elapsed_s"] = round(self.elapsed_s, 6)
        d["confidence"] = round(self.confidence, 6)
        d["margin"] = round(self.margin, 6)
        return d


@dataclass
class ComposeReport:
    term: str
    accepted: bool
    concept: str
    validation_accuracy: float | None
    nodes: int
    elapsed_s: float
    candidates: list[str]

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["elapsed_s"] = round(self.elapsed_s, 6)
        return d


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ConstructiveModel:
    """A compact constructive learner.

    ``fit`` induces one concept from a few examples. ``predict_episode``
    retrieves the stored concept that explains a support set and applies it to
    a query. ``compose`` builds a new program out of already validated concepts.
    ``revise`` performs the smallest repair it can justify.
    """

    def __init__(self, dim: int = 16, beta: float = 1.0, abstain_margin: float = 0.12,
                 max_cost: int = 4, beam: int = 45, max_nodes: int = 90_000,
                 time_budget_s: float = 2.0, seed: int = 0):
        self.dim = dim
        self.beta = beta
        self.abstain_margin = abstain_margin
        self.max_cost = max_cost
        self.beam = beam
        self.max_nodes = max_nodes
        self.time_budget_s = time_budget_s
        self.seed = seed
        self.concepts: dict[str, Concept] = {}
        self.hypotheses: dict[str, Concept] = {}
        self.rejected: list[dict[str, Any]] = []
        self.active: list[str] = []
        self.prims: dict[str, Prim] = build_prims()
        # the only learned parameters: a linear prior over concept embeddings
        self.w = [0.0] * (dim + 1)
        self.lr = 0.05
        self.history: list[dict[str, Any]] = []

    # -- learned prior ---------------------------------------------------
    def prior(self, c: Concept) -> float:
        z = self.w[-1] + sum(w * v for w, v in zip(self.w[:-1], c.embedding))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    def reward(self, c: Concept, reward: float) -> None:
        for i, v in enumerate(c.embedding):
            self.w[i] += self.lr * reward * v
        self.w[-1] += self.lr * reward

    def feedback(self, concept_name: str, correct: bool) -> None:
        """Online feedback on a retrieval/prediction outcome."""
        c = self.concepts.get(concept_name) or self.hypotheses.get(concept_name)
        if c is not None:
            self.reward(c, 1.0 if correct else -1.0)
            c.uses += 1

    def n_parameters(self) -> int:
        return len(self.w) + sum(c.stored_nodes() for c in self.concepts.values())

    def n_stored_nodes(self) -> int:
        return sum(c.stored_nodes() for c in self.concepts.values())

    # -- constants -------------------------------------------------------
    def _constants(self, examples: list[tuple[Any, Any]]) -> list[int]:
        vals = {0, 1, 2, 3}
        for x, y in examples:
            for v in _flatten(x) + _flatten(y):
                if isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 12:
                    vals.add(v)
        return sorted(vals)[:12]

    # -- fit -------------------------------------------------------------
    def fit(self, examples: list[tuple[Any, Any]], input_type: str, output_type: str,
            validation: list[tuple[Any, Any]] | None = None, probes: list[Any] | None = None,
            eps: float = 0.0, max_cost: int | None = None) -> FitReport:
        t0 = time.perf_counter()
        if not examples:
            raise ValueError("fit requires at least one example")
        val = list(validation or [])
        self.prims = build_prims(self._constants(examples + val))
        searcher = Searcher(self.prims, self.concepts, self.beam, self.max_nodes, self.time_budget_s)
        cands, nodes = searcher.search(examples, input_type, output_type,
                                       max_cost=max_cost or self.max_cost, eps=eps,
                                       beta=self.beta, probes=probes)

        if not cands:
            rep = FitReport("", False, True, 0.0, 0.0, 0, [], nodes,
                            time.perf_counter() - t0, True, None, "no candidate within budget",
                            searcher.timed_out)
            self.rejected.append(rep.to_dict())
            self.history.append(rep.to_dict())
            return rep

        for c in cands:
            if val:
                best_term = c.term
                best_acc = _accuracy_term(best_term, val, self.prims, self.concepts)
                for alt in c.alts:
                    acc = _accuracy_term(alt, val, self.prims, self.concepts)
                    if acc > best_acc or (acc == best_acc and alt.size() < best_term.size()):
                        best_term, best_acc = alt, acc
                if best_term is not c.term:
                    c.term = best_term
                c.validation_accuracy = best_acc

        exact = [c for c in cands if c.n_ok == len(examples)]
        if val:
            survivors = [c for c in exact if c.validation_accuracy == 1.0]
        else:
            survivors = exact

        if not survivors:
            # Error-driven salvage: a rule that explains every example but one,
            # and still generalises to held-out validation, is accepted with the
            # inconsistent example quarantined as the identified violation.
            near = [c for c in cands
                    if c.n_ok == len(examples) - 1 and len(examples) >= 3
                    and (not val or c.validation_accuracy == 1.0)]
            if near:
                best = min(near, key=lambda c: (c.cost, str(c.term)))
                bad = [i for i, (v, (_x, y)) in enumerate(zip(best.outputs, examples))
                       if v is UNDEF or not same_value(v, y)]
                name = f"c{len(self.concepts) + len(self.hypotheses):03d}"
                c = Concept(
                    name=name,
                    params=("x",),
                    body=best.term,
                    ret=output_type,
                    input_type=input_type,
                    arg_types=(input_type,),
                    confidence=0.65,
                    support=len(examples) - len(bad),
                    created=len(self.concepts),
                    embedding=embed_term(best.term, self.dim),
                    validation=[e for i, e in enumerate(examples) if i not in bad] + val,
                    note="rule with quarantined outlier",
                )
                self.concepts[name] = c
                self.active.append(name)
                self.reward(c, 1.0)
                rep = FitReport(name, True, False, 0.65, 1.0, best.cost,
                                [str(best.term)], nodes, time.perf_counter() - t0, True,
                                best.validation_accuracy,
                                "one inconsistent example quarantined", searcher.timed_out, bad)
                self.history.append(rep.to_dict())
                return rep

            best = max(cands, key=lambda c: (c.n_ok, c.validation_accuracy or 0.0, -c.cost))
            reason = ("fits all examples but fails held-out validation"
                      if exact else "no program reproduces all examples")
            rep = FitReport("", False, True, 0.0, 0.0, best.cost, [str(c.term) for c in cands[:5]],
                            nodes, time.perf_counter() - t0, best.n_ok < len(examples),
                            best.validation_accuracy, reason, searcher.timed_out)
            self.rejected.append(rep.to_dict())
            self.history.append(rep.to_dict())
            return rep

        # posterior among validated exact fits only
        _normalise(survivors, examples, eps, self.beta)
        best = max(survivors, key=lambda c: (c.posterior, -c.cost))
        second = max((c for c in survivors if c.term != best.term), default=None,
                     key=lambda c: c.posterior)
        margin = best.posterior - (second.posterior if second else 0.0)
        ambiguous = second is not None and margin < self.abstain_margin
        val_acc = best.validation_accuracy if val else None

        # reuse: the winner is exactly a call to an existing concept
        reused = None
        if best.term.op == "concept" and len(best.term.args) == 1 and best.term.args[0] == V("x"):
            reused = self.concepts.get(best.term.value)

        if reused is not None:
            reused.support += 1
            reused.uses += 1
            reused.confidence = min(0.99, 0.5 * reused.confidence + 0.5 * best.posterior)
            c = reused
            note = "reused"
        else:
            name = f"c{len(self.concepts) + len(self.hypotheses):03d}"
            conf = max(0.05, min(0.99, best.posterior * (1.0 - 0.5 ** max(1, len(examples)))))
            c = Concept(
                name=name,
                params=("x",),
                body=best.term,
                ret=output_type,
                input_type=input_type,
                arg_types=(input_type,),
                confidence=conf,
                support=len(examples),
                created=len(self.concepts),
                embedding=embed_term(best.term, self.dim),
                validation=list(examples) + val,
                note="validated concept",
            )
            note = "new"
            if ambiguous:
                # two observationally distinct rules fit; unsettled evidence
                self.hypotheses[name] = c
                note = "hypothesis"
            else:
                self.concepts[name] = c
                self.active.append(name)

        if note in ("new", "reused"):
            self.reward(c, 1.0)

        rep = FitReport(
            concept=c.name,
            accepted=note in ("new", "reused"),
            abstained=bool(ambiguous or note == "hypothesis"),
            confidence=float(best.posterior),
            margin=float(margin),
            cost=best.cost,
            candidates=[str(x.term) for x in survivors[:5]],
            nodes=nodes,
            elapsed_s=time.perf_counter() - t0,
            partial=False,
            validation_accuracy=val_acc,
            reason="" if note in ("new", "reused") else "ambiguous with low support",
            timed_out=searcher.timed_out,
        )
        self.history.append(rep.to_dict())
        return rep

    # -- retrieval and prediction ---------------------------------------
    def rank(self, support: list[tuple[Any, Any]], input_type: str | None = None) -> list[dict[str, Any]]:
        """Rank stored concepts by how well they explain a support set.

        The support set is the few-shot task specification available at test
        time.  Matching is behavioural, so it identifies *which* previously
        acquired concept applies; the learned prior only breaks ties.
        """
        out: list[dict[str, Any]] = []
        for c in self.concepts.values():
            if input_type is not None and c.input_type != input_type:
                continue
            ok = 0
            defined = 0
            for x, y in support:
                if not c.accepts(x, self.prims, self.concepts):
                    continue
                defined += 1
                try:
                    if same_value(c.apply([x], self.prims, self.concepts), y):
                        ok += 1
                except EvalError:
                    pass
            ratio = ok / len(support) if support else 0.0
            out.append({
                "concept": c.name,
                "match": ratio,
                "defined": defined,
                "prior": self.prior(c),
                "confidence": c.confidence,
                "score": ratio * (0.5 + 0.5 * self.prior(c)) * c.confidence,
            })
        out.sort(key=lambda r: (-r["match"], -r["score"], r["concept"]))
        return out

    def predict_episode(self, support: list[tuple[Any, Any]], x: Any, input_type: str,
                        return_meta: bool = False) -> Any:
        ranked = self.rank(support, input_type)
        meta: dict[str, Any] = {"abstained": True, "reason": "no concepts"}
        if not ranked:
            return (None, meta) if return_meta else None
        top = ranked[0]
        if top["match"] < 0.75:
            meta["reason"] = "support set not explained by a stored concept"
            meta["match"] = top["match"]
            return (None, meta) if return_meta else None
        c = self.concepts[top["concept"]]
        if not c.accepts(x, self.prims, self.concepts):
            meta["reason"] = f"guard rejects input for {c.name}"
            return (None, meta) if return_meta else None
        try:
            value = c.apply([x], self.prims, self.concepts)
        except EvalError as exc:
            meta["reason"] = f"undefined: {exc}"
            return (None, meta) if return_meta else None
        # ambiguity: an equally matching concept disagrees on this query
        rivals = [r for r in ranked[1:] if r["match"] == top["match"]]
        for r in rivals:
            rc = self.concepts[r["concept"]]
            if not rc.accepts(x, self.prims, self.concepts):
                continue
            try:
                rv = rc.apply([x], self.prims, self.concepts)
            except EvalError:
                continue
            if not same_value(rv, value):
                meta.update({"reason": f"ambiguous with {rc.name}",
                             "rivals": [r["concept"] for r in rivals]})
                return (None, meta) if return_meta else None
        meta = {"abstained": False, "concept": c.name, "match": top["match"],
                "prior": top["prior"], "confidence": c.confidence}
        return (value, meta) if return_meta else value

    def predict(self, x: Any, return_meta: bool = False) -> Any:
        """Predict with the active concept set (no support set available)."""
        outs: list[tuple[float, Concept, Any]] = []
        for name in self.active:
            c = self.concepts.get(name)
            if c is None or not c.accepts(x, self.prims, self.concepts):
                continue
            try:
                v = c.apply([x], self.prims, self.concepts)
            except EvalError:
                continue
            outs.append((c.confidence * max(0.05, self.prior(c)), c, v))
        if not outs:
            return (None, {"abstained": True, "reason": "no applicable concept"}) if return_meta else None
        outs.sort(key=lambda t: t[0], reverse=True)
        best_score, best_c, best_v = outs[0]
        margin = ((best_score - outs[1][0]) / max(best_score, 1e-9)) if len(outs) > 1 else 1.0
        abstained = margin < self.abstain_margin or best_c.confidence < 0.30
        meta = {"abstained": abstained, "concept": best_c.name, "margin": margin}
        return (None if abstained else best_v, meta) if return_meta else (None if abstained else best_v)

    # -- composition -----------------------------------------------------
    def compose(self, examples: list[tuple[Any, Any]], input_type: str, output_type: str,
                validation: list[tuple[Any, Any]] | None = None,
                probes: list[Any] | None = None, max_cost: int = 2) -> ComposeReport:
        """Induce a program that *reuses* stored concepts as units.

        The cost budget is deliberately smaller than the cost of the target's
        primitive expansion, so success requires composition rather than a
        lucky flat search.
        """
        t0 = time.perf_counter()
        val = list(validation or [])
        self.prims = build_prims(self._constants(examples + val))
        searcher = Searcher(self.prims, self.concepts, self.beam, self.max_nodes, self.time_budget_s)
        cands, nodes = searcher.search(examples, input_type, output_type, max_cost=max_cost,
                                       probes=probes)
        exact = [c for c in cands if c.n_ok == len(examples)]
        if val:
            exact = [c for c in exact if _accuracy_term(c.term, val, self.prims, self.concepts) == 1.0]
        exact = [c for c in exact if _contains_concept_call(c.term)]
        if not exact:
            return ComposeReport("", False, "", None, nodes, time.perf_counter() - t0,
                                 [str(c.term) for c in cands[:5]])
        best = min(exact, key=lambda c: (c.cost, str(c.term)))
        val_acc = _accuracy_term(best.term, val, self.prims, self.concepts) if val else None
        name = f"c{len(self.concepts) + len(self.hypotheses):03d}"
        c = Concept(
            name=name,
            params=("x",),
            body=best.term,
            ret=output_type,
            input_type=input_type,
            arg_types=(input_type,),
            confidence=0.9,
            support=len(examples),
            created=len(self.concepts),
            embedding=embed_term(best.term, self.dim),
            validation=list(examples) + val,
            note="composed",
        )
        self.concepts[name] = c
        self.active.append(name)
        self.reward(c, 1.0)
        return ComposeReport(str(best.term), True, name, val_acc, nodes,
                             time.perf_counter() - t0, [str(x.term) for x in exact[:5]])

    # -- guards and repair ---------------------------------------------
    def find_guard(self, positives: list[Any], negatives: list[Any], input_type: str,
                   max_cost: int = 2) -> Term | None:
        """Find a boolean term true on every positive and false on every negative."""
        examples = [(x, True) for x in positives] + [(x, False) for x in negatives]
        if not examples:
            return None
        searcher = Searcher(self.prims, self.concepts, beam=30, max_nodes=20_000, time_budget_s=0.5)
        cands, _ = searcher.search(examples, input_type, "bool", max_cost=max_cost, eps=0.0)
        for c in sorted(cands, key=lambda c: (c.cost, str(c.term))):
            if c.n_ok == len(examples):
                return c.term
        return None

    def revise(self, concept_name: str, new_examples: list[tuple[Any, Any]],
               input_type: str, validation: list[tuple[Any, Any]] | None = None,
               probes: list[Any] | None = None) -> dict[str, Any]:
        """Smallest justified repair.

        Order of preference:
          1. global rewrite - one body explains old *and* new evidence, edit
             distance <= 2, and every old example still passes;
          2. scoped concept - a child concept handles the new context, guarded
             by a predicate that separates new from old inputs, and the parent
             is locally restricted with the negated guard;
          3. rejection - no justified repair.
        """
        t0 = time.perf_counter()
        if concept_name not in self.concepts:
            return {"mode": "rejected", "reason": f"unknown concept {concept_name}"}
        c = self.concepts[concept_name]
        old = list(c.validation)
        new = list(new_examples)
        if not new:
            return {"mode": "rejected", "reason": "no new evidence"}
        val = list(validation or [])
        combined = old + new
        self.prims = build_prims(self._constants(combined + val))

        # --- 1. global rewrite -----------------------------------------
        searcher = Searcher(self.prims, self.concepts, self.beam, self.max_nodes, self.time_budget_s)
        cands, nodes = searcher.search(combined, input_type, c.ret,
                                       max_cost=min(self.max_cost + 1, 6), probes=probes)
        exact = [x for x in cands if x.n_ok == len(combined)]
        if val:
            exact = [x for x in exact if _accuracy_term(x.term, val, self.prims, self.concepts) == 1.0]
        if exact:
            best = min(exact, key=lambda x: (tree_edit_distance(x.term, c.body), x.cost))
            dist = tree_edit_distance(best.term, c.body)
            old_ok = _accuracy_term(best.term, old, self.prims, self.concepts)
            if old_ok == 1.0 and dist <= 2:
                before = str(c.body)
                c.body = best.term
                c.validation = combined
                c.embedding = embed_term(best.term, self.dim)
                c.note = "globally repaired"
                return {"mode": "global", "before": before, "after": str(c.body),
                        "distance": dist, "retention": old_ok, "nodes": nodes,
                        "elapsed_s": time.perf_counter() - t0}

        # --- 2. scoped repair ------------------------------------------
        sc = Searcher(self.prims, self.concepts, self.beam, self.max_nodes, self.time_budget_s)
        ncands, nnodes = sc.search(new, input_type, c.ret,
                                   max_cost=self.max_cost, probes=probes)
        exact_new = [x for x in ncands if x.n_ok == len(new)]
        if val:
            exact_new = [x for x in exact_new
                         if _accuracy_term(x.term, val, self.prims, self.concepts) == 1.0]
        if not exact_new:
            return {"mode": "rejected", "reason": "no body explains the new evidence consistently",
                    "nodes": nodes + nnodes, "elapsed_s": time.perf_counter() - t0}
        body = min(exact_new, key=lambda x: (x.cost, str(x.term))).term

        guard = self.find_guard([x for x, _ in new], [x for x, _ in old], input_type)
        if guard is None:
            return {"mode": "rejected",
                    "reason": "no context predicate separates new from old evidence",
                    "nodes": nodes + nnodes, "elapsed_s": time.perf_counter() - t0}

        name = f"c{len(self.concepts) + len(self.hypotheses):03d}"
        child = Concept(
            name=name,
            params=("x",),
            body=body,
            ret=c.ret,
            input_type=input_type,
            arg_types=(input_type,),
            guard=guard,
            confidence=0.85,
            support=len(new),
            created=len(self.concepts),
            embedding=embed_term(body, self.dim),
            validation=new + val,
            parent=c.name,
            note="scoped repair",
        )
        self.concepts[name] = child
        self.active.append(name)
        # locally restrict the parent: one guard node, no body change
        old_guard = c.guard
        negated = Term("not", (guard,))
        c.guard = negated if old_guard is None else Term("and", (old_guard, negated))
        c.note = "restricted by scoped repair"
        parent_ok = _accuracy_term(c.body, old, self.prims, self.concepts)
        parent_applicable = all(c.accepts(x, self.prims, self.concepts) for x, _ in old)
        self.reward(child, 1.0)
        return {"mode": "scoped", "new_concept": name, "guard": str(guard),
                "parent_guard": str(c.guard), "retention": parent_ok,
                "parent_still_applicable": parent_applicable,
                "nodes": nodes + nnodes, "elapsed_s": time.perf_counter() - t0}

    # -- retention ------------------------------------------------------
    def retention(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, c in self.concepts.items():
            if not c.validation:
                continue
            out[name] = _accuracy_term(c.body, c.validation, self.prims, self.concepts)
        return out

    def forgetting(self, before: dict[str, float]) -> float:
        after = self.retention()
        if not before:
            return 0.0
        drops = [before[k] - after.get(k, 0.0) for k in before]
        return max(0.0, max(drops) if drops else 0.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise(cands: list[Candidate], examples: list[tuple[Any, Any]], eps: float,
               beta: float) -> None:
    targets = [y for _, y in examples]
    k_out = max(2, len({value_key(y) for y in targets}))
    logs = []
    for cand in cands:
        ll = 0.0
        for v, y in zip(cand.outputs, targets):
            if v is UNDEF:
                ll += math.log(max(eps, 1e-9))
            elif same_value(v, y):
                ll += math.log(max(1.0 - eps, 1e-9))
            else:
                ll += math.log(max(eps / max(1, k_out - 1), 1e-9))
        logs.append(ll - beta * cand.cost)
    m = max(logs)
    ws = [math.exp(v - m) for v in logs]
    s = sum(ws) or 1.0
    for cand, w in zip(cands, ws):
        cand.posterior = w / s


def _contains_concept_call(t: Term) -> bool:
    if t.op == "concept":
        return True
    return any(_contains_concept_call(a) for a in t.args)


def _accuracy_term(term: Term, examples: list[tuple[Any, Any]], prims: dict[str, Prim],
                   concepts: dict[str, Concept]) -> float:
    if not examples:
        return 1.0
    ok = 0
    for x, y in examples:
        try:
            if same_value(eval_term(term, {"x": x}, prims, concepts), y):
                ok += 1
        except EvalError:
            pass
    return ok / len(examples)


def _flatten(v: Any) -> list[Any]:
    if isinstance(v, Scene):
        out: list[Any] = []
        for e in v.entities:
            out.extend([e.x, e.y])
        out.append(v.query)
        return out
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            out.extend(_flatten(x))
        return out
    return [v]
