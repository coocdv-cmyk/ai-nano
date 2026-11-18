"""Core representations for OMEGA Constructive Intelligence (OCI).

A small typed term language over relational scenes, integer lists and numbers.
This module contains no learning: it defines values, terms, primitives and
evaluation. The constructive learner lives in ``model.py``.

Design constraints:
  * standard library only (no numpy in this module) so the semantics are exact;
  * every program is a pure function of its input;
  * evaluation errors are explicit (``EvalError``), never silently replaced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Entity:
    id: str
    color: str
    shape: str
    size: str
    x: int
    y: int


@dataclass(frozen=True)
class Scene:
    entities: tuple[Entity, ...]
    query: str
    mirror: bool = False

    def by_id(self, eid: str) -> Entity:
        for e in self.entities:
            if e.id == eid:
                return e
        raise EvalError(f"no entity {eid!r}")


class EvalError(Exception):
    """Raised when a term is undefined on an input (type-correct but no value)."""


COLORS = ("red", "blue", "green", "yellow")
SHAPES = ("circle", "square", "triangle", "star")
SIZES = ("small", "large")
SIZE_RANK = {"small": 0, "large": 1}


def type_of_value(v: Any) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, str):
        return "str"
    if isinstance(v, Entity):
        return "entity"
    if isinstance(v, Scene):
        return "scene"
    if isinstance(v, tuple):
        return "pair"
    if isinstance(v, list):
        return "list"
    return "any"


def value_key(v: Any) -> Any:
    """Hashable, type-sensitive identity used for example bookkeeping."""
    if isinstance(v, bool):
        return ("bool", v)
    if isinstance(v, Entity):
        return ("entity", v.id)
    if isinstance(v, Scene):
        return ("scene", tuple((e.id, e.color, e.shape, e.size, e.x, e.y) for e in v.entities), v.query, v.mirror)
    if isinstance(v, list):
        return ("list", tuple(value_key(x) for x in v))
    if isinstance(v, tuple):
        return ("pair", tuple(value_key(x) for x in v))
    return v


def same_value(a: Any, b: Any) -> bool:
    return value_key(a) == value_key(b)


# ---------------------------------------------------------------------------
# Terms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Term:
    op: str
    args: tuple["Term", ...] = ()
    value: Any = None

    def __str__(self) -> str:  # human readable, deterministic
        if self.op == "var":
            return str(self.value)
        if self.op == "lit":
            return repr(self.value)
        if self.op == "concept":
            inner = ", ".join(str(a) for a in self.args)
            return f"{self.value}({inner})"
        inner = ", ".join(str(a) for a in self.args)
        return f"{self.op}({inner})"

    def size(self) -> int:
        return 1 + sum(a.size() for a in self.args)

    def depth(self) -> int:
        return 1 + (max((a.depth() for a in self.args), default=0))


def V(name: str = "x") -> Term:
    return Term("var", value=name)


def L(value: Any) -> Term:
    return Term("lit", value=value)


def C(name: str, *args: Term) -> Term:
    return Term("concept", args=tuple(args), value=name)


def subterms(t: Term) -> Iterable[Term]:
    yield t
    for a in t.args:
        yield from subterms(a)


def replace(t: Term, target: Term, replacement: Term) -> Term:
    if t == target:
        return replacement
    return Term(t.op, tuple(replace(a, target, replacement) for a in t.args), t.value)


def tree_edit_distance(a: Term, b: Term) -> int:
    """A cheap structural distance: node replacement + child insert/delete."""
    if a.op == b.op and a.value == b.value and len(a.args) == len(b.args):
        return sum(tree_edit_distance(x, y) for x, y in zip(a.args, b.args))
    if a.op == b.op and a.value == b.value:
        n = min(len(a.args), len(b.args))
        d = sum(tree_edit_distance(a.args[i], b.args[i]) for i in range(n))
        return d + abs(len(a.args) - len(b.args))
    return 1 + sum(tree_edit_distance(x, b) for x in a.args) + sum(tree_edit_distance(a, y) for y in b.args)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Prim:
    name: str
    args: tuple[str, ...]
    ret: str
    fn: Callable[..., Any]
    cost: int = 1


def _p(name: str, args: tuple[str, ...], ret: str, fn: Callable[..., Any], cost: int = 1) -> Prim:
    return Prim(name, args, ret, fn, cost)


def _manhattan(a: Entity, b: Entity) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)


def _others(scene: Scene, e: Entity) -> list[Entity]:
    return [o for o in scene.entities if o.id != e.id]


def _nearest(scene: Scene, e: Entity) -> Entity:
    cands = _others(scene, e)
    if not cands:
        raise EvalError("no other entity")
    return min(cands, key=lambda o: (_manhattan(e, o), o.id))


def _left_of(scene: Scene, e: Entity) -> Entity:
    cands = [o for o in _others(scene, e) if (o.x, o.y) < (e.x, e.y)]
    if not cands:
        raise EvalError("nothing to the left")
    return max(cands, key=lambda o: (o.x, o.y))


def _right_of(scene: Scene, e: Entity) -> Entity:
    cands = [o for o in _others(scene, e) if (o.x, o.y) > (e.x, e.y)]
    if not cands:
        raise EvalError("nothing to the right")
    return min(cands, key=lambda o: (o.x, o.y))


def _above(scene: Scene, e: Entity) -> Entity:
    cands = [o for o in _others(scene, e) if o.y > e.y]
    if not cands:
        raise EvalError("nothing above")
    return min(cands, key=lambda o: (o.y, o.x))


def _below(scene: Scene, e: Entity) -> Entity:
    cands = [o for o in _others(scene, e) if o.y < e.y]
    if not cands:
        raise EvalError("nothing below")
    return max(cands, key=lambda o: (o.y, o.x))


def _same_color(scene: Scene, e: Entity) -> Entity:
    cands = [o for o in _others(scene, e) if o.color == e.color]
    if not cands:
        raise EvalError("no same-colour entity")
    return min(cands, key=lambda o: (_manhattan(e, o), o.id))


def _same_shape(scene: Scene, e: Entity) -> Entity:
    cands = [o for o in _others(scene, e) if o.shape == e.shape]
    if not cands:
        raise EvalError("no same-shape entity")
    return min(cands, key=lambda o: (_manhattan(e, o), o.id))


def _count_same_color(scene: Scene, e: Entity) -> int:
    return sum(1 for o in scene.entities if o.color == e.color)


def _count_same_shape(scene: Scene, e: Entity) -> int:
    return sum(1 for o in scene.entities if o.shape == e.shape)


def _largest(scene: Scene) -> Entity:
    return max(scene.entities, key=lambda o: (SIZE_RANK[o.size], o.x, o.y, o.id))


def _smallest(scene: Scene) -> Entity:
    return min(scene.entities, key=lambda o: (SIZE_RANK[o.size], o.x, o.y, o.id))


def _range(xs: list[int]) -> int:
    return max(xs) - min(xs) if xs else 0


def _gcd(a: int, b: int) -> int:
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def _sum_digits(n: int) -> int:
    return sum(int(c) for c in str(abs(n)))


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def _fib(n: int) -> int:
    if n < 0:
        raise EvalError("negative fib")
    a, b = 0, 1
    for _ in range(min(n, 40)):
        a, b = b, a + b
    return a


def _fact(n: int) -> int:
    if n < 0:
        raise EvalError("negative factorial")
    out = 1
    for i in range(2, min(n, 15) + 1):
        out *= i
    return out


def build_prims(int_constants: Iterable[int] = (0, 1, 2, 3), string_constants: Iterable[str] = COLORS) -> dict[str, Prim]:
    """Build the primitive registry for one learning episode.

    Integer/string constants are instantiated as primitives so that a rule such
    as ``+2`` is a single cost unit. This is a deliberate inductive bias: it
    matches how a learner would consider a small set of salient constants.
    """
    prims: dict[str, Prim] = {}
    add = lambda p: prims.__setitem__(p.name, p)

    # --- arithmetic -------------------------------------------------------
    add(_p("add1", ("int",), "int", lambda x: x + 1))
    add(_p("sub1", ("int",), "int", lambda x: x - 1))
    add(_p("double", ("int",), "int", lambda x: x * 2))
    add(_p("square", ("int",), "int", lambda x: x * x))
    add(_p("neg", ("int",), "int", lambda x: -x))
    add(_p("abs", ("int",), "int", abs))
    add(_p("sum_digits", ("int",), "int", _sum_digits))
    add(_p("digital_root", ("int",), "int", lambda x: 0 if x == 0 else 1 + (abs(x) - 1) % 9))
    add(_p("is_prime", ("int",), "bool", _is_prime))
    add(_p("even", ("int",), "bool", lambda x: x % 2 == 0))
    add(_p("odd", ("int",), "bool", lambda x: x % 2 == 1))
    add(_p("fib", ("int",), "int", _fib))
    add(_p("fact", ("int",), "int", _fact))

    for k in sorted({int(c) for c in int_constants if -20 <= int(c) <= 20}):
        add(_p(f"add({k})", ("int",), "int", lambda x, k=k: x + k))
        add(_p(f"mul({k})", ("int",), "int", lambda x, k=k: x * k))
        add(_p(f"mod({k})", ("int",), "int", lambda x, k=k: x % k if k else 0))
        add(_p(f"gt({k})", ("int",), "bool", lambda x, k=k: x > k))
        add(_p(f"lt({k})", ("int",), "bool", lambda x, k=k: x < k))

    # --- lists ------------------------------------------------------------
    add(_p("sum_list", ("list",), "int", lambda xs: sum(xs)))
    add(_p("max_list", ("list",), "int", lambda xs: max(xs) if xs else 0))
    add(_p("min_list", ("list",), "int", lambda xs: min(xs) if xs else 0))
    add(_p("len_list", ("list",), "int", lambda xs: len(xs)))
    add(_p("head", ("list",), "int", lambda xs: xs[0] if xs else 0))
    add(_p("last", ("list",), "int", lambda xs: xs[-1] if xs else 0))
    add(_p("range_list", ("list",), "int", _range))
    add(_p("count_even", ("list",), "int", lambda xs: sum(1 for x in xs if x % 2 == 0)))
    add(_p("count_odd", ("list",), "int", lambda xs: sum(1 for x in xs if x % 2 == 1)))
    add(_p("sum_squares", ("list",), "int", lambda xs: sum(x * x for x in xs)))
    add(_p("reverse", ("list",), "list", lambda xs: list(reversed(xs))))
    add(_p("sort", ("list",), "list", lambda xs: sorted(xs)))
    add(_p("sort_desc", ("list",), "list", lambda xs: sorted(xs, reverse=True)))
    add(_p("unique", ("list",), "list", lambda xs: list(dict.fromkeys(xs))))
    add(_p("evens", ("list",), "list", lambda xs: [x for x in xs if x % 2 == 0]))
    add(_p("odds", ("list",), "list", lambda xs: [x for x in xs if x % 2 == 1]))

    for k in sorted({int(c) for c in int_constants if 0 <= int(c) <= 12}):
        add(_p(f"take({k})", ("list",), "list", lambda xs, k=k: xs[:k]))
        add(_p(f"drop({k})", ("list",), "list", lambda xs, k=k: xs[k:]))
        add(_p(f"filter_gt({k})", ("list",), "list", lambda xs, k=k: [x for x in xs if x > k]))
        add(_p(f"map_add({k})", ("list",), "list", lambda xs, k=k: [x + k for x in xs]))
        add(_p(f"map_mul({k})", ("list",), "list", lambda xs, k=k: [x * k for x in xs]))
        add(_p(f"count_gt({k})", ("list",), "int", lambda xs, k=k: sum(1 for x in xs if x > k)))

    # --- pairs ------------------------------------------------------------
    add(_p("first", ("pair",), "int", lambda p: p[0]))
    add(_p("second", ("pair",), "int", lambda p: p[1]))
    add(_p("sum_pair", ("pair",), "int", lambda p: p[0] + p[1]))
    add(_p("diff_pair", ("pair",), "int", lambda p: abs(p[0] - p[1])))
    add(_p("gcd_pair", ("pair",), "int", lambda p: _gcd(p[0], p[1])))
    add(_p("max_pair", ("pair",), "int", lambda p: max(p)))
    add(_p("min_pair", ("pair",), "int", lambda p: min(p)))

    # --- scenes -----------------------------------------------------------
    add(_p("query", ("scene",), "entity", lambda s: s.by_id(s.query)))
    add(_p("left_of", ("scene", "entity"), "entity", _left_of))
    add(_p("right_of", ("scene", "entity"), "entity", _right_of))
    add(_p("above", ("scene", "entity"), "entity", _above))
    add(_p("below", ("scene", "entity"), "entity", _below))
    add(_p("same_color", ("scene", "entity"), "entity", _same_color))
    add(_p("same_shape", ("scene", "entity"), "entity", _same_shape))
    add(_p("nearest", ("scene", "entity"), "entity", _nearest))
    add(_p("farthest", ("scene", "entity"), "entity", lambda s, e: max(_others(s, e), key=lambda o: (_manhattan(e, o), o.id))))
    add(_p("biggest", ("scene",), "entity", _largest))
    add(_p("smallest", ("scene",), "entity", _smallest))
    add(_p("count_same_color", ("scene", "entity"), "int", _count_same_color))
    add(_p("count_same_shape", ("scene", "entity"), "int", _count_same_shape))
    add(_p("mirror", ("scene",), "bool", lambda s: s.mirror))
    add(_p("attr_color", ("entity",), "str", lambda e: e.color))
    add(_p("attr_shape", ("entity",), "str", lambda e: e.shape))
    add(_p("attr_size", ("entity",), "str", lambda e: e.size))
    add(_p("coord_x", ("entity",), "int", lambda e: e.x))
    add(_p("coord_y", ("entity",), "int", lambda e: e.y))

    for c in sorted(set(string_constants)):
        add(_p(f"count_color({c})", ("scene",), "int", lambda s, c=c: sum(1 for e in s.entities if e.color == c)))
        add(_p(f"count_shape({c})", ("scene",), "int", lambda s, c=c: sum(1 for e in s.entities if e.shape == c)))

    # --- logic ------------------------------------------------------------
    add(_p("not", ("bool",), "bool", lambda b: not b))
    add(_p("and", ("bool", "bool"), "bool", lambda a, b: bool(a and b)))
    add(_p("or", ("bool", "bool"), "bool", lambda a, b: bool(a or b)))
    return prims


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def eval_term(t: Term, env: dict[str, Any], prims: dict[str, Prim], concepts: dict[str, Any] | None = None) -> Any:
    if t.op == "var":
        try:
            return env[t.value]
        except KeyError as exc:
            raise EvalError(f"unbound variable {t.value!r}") from exc
    if t.op == "lit":
        return t.value
    if t.op == "concept":
        if concepts is None or t.value not in concepts:
            raise EvalError(f"unknown concept {t.value!r}")
        c = concepts[t.value]
        args = [eval_term(a, env, prims, concepts) for a in t.args]
        try:
            return c.apply(args, prims, concepts)
        except EvalError:
            raise
        except Exception as exc:  # a concept body error is an undefined application
            raise EvalError(f"concept {t.value!r} failed: {exc}") from exc
    if t.op not in prims:
        raise EvalError(f"unknown primitive {t.op!r}")
    p = prims[t.op]
    args = [eval_term(a, env, prims, concepts) for a in t.args]
    try:
        return p.fn(*args)
    except EvalError:
        raise
    except Exception as exc:
        raise EvalError(str(exc)) from exc


def infer_type(t: Term, env_types: dict[str, str], prims: dict[str, Prim], concepts: dict[str, Any] | None = None) -> str:
    if t.op == "var":
        return env_types.get(t.value, "any")
    if t.op == "lit":
        return type_of_value(t.value)
    if t.op == "concept":
        if concepts is None or t.value not in concepts:
            return "any"
        return concepts[t.value].ret
    if t.op not in prims:
        return "any"
    return prims[t.op].ret


def type_ok(actual: str, expected: str) -> bool:
    if expected == "any" or actual == "any":
        return True
    if actual == expected:
        return True
    # lists are homogeneous enough for this DSL; element types are checked at runtime
    if expected.startswith("list") and actual.startswith("list"):
        return True
    return False


def well_typed(t: Term, env_types: dict[str, str], prims: dict[str, Prim], concepts: dict[str, Any] | None = None) -> bool:
    if t.op == "var":
        return t.value in env_types
    if t.op == "lit":
        return True
    if t.op == "concept":
        if concepts is None or t.value not in concepts:
            return False
        c = concepts[t.value]
        if len(c.params) != len(t.args):
            return False
        expected = getattr(c, "arg_types", ()) or tuple("any" for _ in c.params)
        if len(expected) != len(t.args):
            expected = tuple("any" for _ in c.params)
        return all(
            type_ok(infer_type(a, env_types, prims, concepts), p)
            for a, p in zip(t.args, expected)
        )
    if t.op not in prims:
        return False
    p = prims[t.op]
    if len(p.args) != len(t.args):
        return False
    for a, expected in zip(t.args, p.args):
        if not type_ok(infer_type(a, env_types, prims, concepts), expected):
            return False
        if not well_typed(a, env_types, prims, concepts):
            return False
    return True


def evaluate_examples(
    t: Term,
    examples: list[tuple[Any, Any]],
    prims: dict[str, Prim],
    concepts: dict[str, Any] | None = None,
    env_name: str = "x",
) -> tuple[Any, ...] | None:
    """Return the tuple of outputs on all examples, or None if undefined."""
    out = []
    for x, _y in examples:
        try:
            out.append(eval_term(t, {env_name: x}, prims, concepts))
        except EvalError:
            return None
    return tuple(out)


def matches(t: Term, examples: list[tuple[Any, Any]], prims: dict[str, Prim], concepts: dict[str, Any] | None = None) -> int:
    """Number of examples on which the term is defined and correct."""
    n = 0
    for x, y in examples:
        try:
            v = eval_term(t, {"x": x}, prims, concepts)
            if same_value(v, y):
                n += 1
        except EvalError:
            pass
    return n


def term_cost(t: Term) -> int:
    if t.op in ("var",):
        return 0
    if t.op == "lit":
        return 1
    if t.op == "concept":
        return 1 + sum(term_cost(a) for a in t.args)
    return 1 + sum(term_cost(a) for a in t.args)
