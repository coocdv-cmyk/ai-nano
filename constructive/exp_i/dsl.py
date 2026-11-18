"""EXP-I-B — an independent, strictly typed, extensible expression DSL.

No CCM dependency. Programs are immutable trees; types are explicit; execution is
step-, depth- and magnitude-bounded. The DSL deliberately supports the operators
EXP-H could not express: conditionals, boolean predicates, comparisons, logical
connectives, operator exclusion (bitwise), bounded arithmetic, parameterized
list transforms and structured list operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .types import Expressiveness, stable_key


# ---------------------------------------------------------------------------
# Types and operator registry
# ---------------------------------------------------------------------------

INT = "int"
BOOL = "bool"
LIST = "list"
PAIR = "pair"
ANY = "any"

_PRIMITIVE_TYPES = {INT, BOOL, LIST, PAIR}


class DslError(Exception):
    """Type error, budget failure or undefined operation. Never silently absorbed."""


@dataclass(frozen=True)
class OpSpec:
    children: tuple[str, ...]      # required child types (ANY = polymorphic slot)
    out: str                        # output type ("SAME" = polymorphic from children)
    eval: Callable[..., Any] | None = None
    commutative: bool = False
    has_param: bool = False
    category: str = "core"


def _need_int(v: Any) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise DslError(f"expected int, got {type(v).__name__}")
    return v


def _need_bool(v: Any) -> bool:
    if not isinstance(v, bool):
        raise DslError(f"expected bool, got {type(v).__name__}")
    return v


def _need_list(v: Any) -> list[int]:
    if not isinstance(v, list):
        raise DslError(f"expected list, got {type(v).__name__}")
    out = []
    for x in v:
        if isinstance(x, bool) or not isinstance(x, int):
            raise DslError("list contains a non-int")
        out.append(x)
    return out


def _need_pair(v: Any) -> tuple[int, int]:
    if not isinstance(v, tuple) or len(v) != 2:
        raise DslError(f"expected pair, got {v!r}")
    a, b = v
    if isinstance(a, bool) or isinstance(b, bool):
        raise DslError("pair holds a bool")
    return int(a), int(b)


def _is_prime(n: int) -> bool:
    n = _need_int(n)
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


def _bounded_fact(n: int, cap: int) -> int:
    n = min(_need_int(n), 16)
    acc = 1
    for i in range(2, n + 1):
        acc *= i
        if abs(acc) > cap:
            raise DslError("integer magnitude exceeded (fact)")
    return acc


def _digits(n: int) -> list[int]:
    return [int(c) for c in str(abs(_need_int(n)))]


def _need_list_sum(xs: list[int]) -> int:
    return sum(_need_list(xs))


def _fib_bounded(n: int) -> int:
    a, b = 0, 1
    for _ in range(max(0, min(_need_int(n), 30))):
        a, b = b, a + b
    return a


# fmt: off
OPS: dict[str, OpSpec] = {
    # variables / literals
    "var":        OpSpec((), ANY, None, category="core"),        # type carried in param
    "lit":        OpSpec((), INT, None, has_param=True, category="core"),
    "bool":       OpSpec((), BOOL, None, has_param=True, category="core"),
    # unbounded-free unary integer ops
    "add1":       OpSpec((INT,), INT, lambda a: a + 1),
    "sub1":       OpSpec((INT,), INT, lambda a: a - 1),
    "neg":        OpSpec((INT,), INT, lambda a: -a),
    "abs":        OpSpec((INT,), INT, abs),
    "double":     OpSpec((INT,), INT, lambda a: a * 2),
    "square":     OpSpec((INT,), INT, lambda a: a * a),
    "sum_digits": OpSpec((INT,), INT, lambda a: sum(_digits(a))),
    "digital_root": OpSpec((INT,), INT,
                           lambda a: 0 if a == 0 else 1 + (abs(a) - 1) % 9),
    "fib":        OpSpec((INT,), INT, lambda a: _fib_bounded(a), category="numeric"),
    "fact":       OpSpec((INT,), INT, None),   # magnitude handled in evaluate
    # boolean predicates
    "is_prime":   OpSpec((INT,), BOOL, lambda a: _is_prime(a), category="predicate"),
    "even":       OpSpec((INT,), BOOL, lambda a: a % 2 == 0, category="predicate"),
    "odd":        OpSpec((INT,), BOOL, lambda a: a % 2 == 1, category="predicate"),
    "is_zero":    OpSpec((INT,), BOOL, lambda a: a == 0, category="predicate"),
    "pos":        OpSpec((INT,), BOOL, lambda a: a > 0, category="predicate"),
    # parameterized bounded arithmetic
    "add":        OpSpec((INT,), INT, lambda a, k: a + k, has_param=True),
    "sub":        OpSpec((INT,), INT, lambda a, k: a - k, has_param=True),
    "mul":        OpSpec((INT,), INT, lambda a, k: a * k, has_param=True),
    "mod":        OpSpec((INT,), INT, lambda a, k: (a % k) if k else 0, has_param=True),
    "div":        OpSpec((INT,), INT,
                         lambda a, k: _trunc_div(a, k) if k else 0, has_param=True),
    # comparisons (boolean predicates)
    "lt":         OpSpec((INT,), BOOL, lambda a, k: a < k, has_param=True, category="predicate"),
    "le":         OpSpec((INT,), BOOL, lambda a, k: a <= k, has_param=True, category="predicate"),
    "gt":         OpSpec((INT,), BOOL, lambda a, k: a > k, has_param=True, category="predicate"),
    "ge":         OpSpec((INT,), BOOL, lambda a, k: a >= k, has_param=True, category="predicate"),
    "eq":         OpSpec((INT,), BOOL, lambda a, k: a == k, has_param=True, category="predicate"),
    "ne":         OpSpec((INT,), BOOL, lambda a, k: a != k, has_param=True, category="predicate"),
    # logic / conditionals
    "not":        OpSpec((BOOL,), BOOL, lambda a: not a, category="logic"),
    "and":        OpSpec((BOOL, BOOL), BOOL, lambda a, b: a and b, commutative=True, category="logic"),
    "or":         OpSpec((BOOL, BOOL), BOOL, lambda a, b: a or b, commutative=True, category="logic"),
    "if":         OpSpec((BOOL, ANY, ANY), "SAME", None, category="conditional"),
    # operator exclusion / bitwise
    "band":       OpSpec((INT,), INT, lambda a, k: a & k, has_param=True, category="exclusion"),
    "bor":        OpSpec((INT,), INT, lambda a, k: a | k, has_param=True, category="exclusion"),
    "bxor":       OpSpec((INT,), INT, lambda a, k: a ^ k, has_param=True, category="exclusion"),
    "shl":        OpSpec((INT,), INT,
                         lambda a, k: (a << (k % 32)) & 0xFFFFFFFF, has_param=True, category="exclusion"),
    "shr":        OpSpec((INT,), INT,
                         lambda a, k: (a & 0xFFFFFFFF) >> (k % 32), has_param=True, category="exclusion"),
    # parameterized list transforms
    "map_add":    OpSpec((LIST,), LIST, lambda xs, k: [v + k for v in xs], has_param=True, category="list"),
    "map_mul":    OpSpec((LIST,), LIST, lambda xs, k: [v * k for v in xs], has_param=True, category="list"),
    "map_square": OpSpec((LIST,), LIST, lambda xs: [v * v for v in xs], category="list"),
    "map_cube":   OpSpec((LIST,), LIST, lambda xs: [v * v * v for v in xs], category="list"),
    "filter_gt":  OpSpec((LIST,), LIST, lambda xs, k: [v for v in xs if v > k], has_param=True, category="list"),
    "count_gt":   OpSpec((LIST,), INT,
                         lambda xs, k: sum(1 for v in xs if v > k), has_param=True, category="list"),
    "take":       OpSpec((LIST,), LIST, lambda xs, k: xs[:k], has_param=True, category="list"),
    "drop":       OpSpec((LIST,), LIST, lambda xs, k: xs[k:], has_param=True, category="list"),
    # structured list operations
    "sort":       OpSpec((LIST,), LIST, lambda xs: sorted(xs), category="list"),
    "sort_desc":  OpSpec((LIST,), LIST, lambda xs: sorted(xs, reverse=True), category="list"),
    "reverse":    OpSpec((LIST,), LIST, lambda xs: list(reversed(xs)), category="list"),
    "unique":     OpSpec((LIST,), LIST, lambda xs: list(dict.fromkeys(xs)), category="list"),
    "evens":      OpSpec((LIST,), LIST, lambda xs: [v for v in xs if v % 2 == 0], category="list"),
    "odds":       OpSpec((LIST,), LIST, lambda xs: [v for v in xs if v % 2 == 1], category="list"),
    # list reductions
    "sum_list":   OpSpec((LIST,), INT, _need_list_sum, category="list"),
    "max_list":   OpSpec((LIST,), INT, lambda xs: max(xs) if xs else 0, category="list"),
    "min_list":   OpSpec((LIST,), INT, lambda xs: min(xs) if xs else 0, category="list"),
    "len_list":   OpSpec((LIST,), INT, len, category="list"),
    "head":       OpSpec((LIST,), INT, lambda xs: xs[0] if xs else 0, category="list"),
    "last":       OpSpec((LIST,), INT, lambda xs: xs[-1] if xs else 0, category="list"),
    "range_list": OpSpec((LIST,), INT, lambda xs: (max(xs) - min(xs)) if xs else 0, category="list"),
    "count_even": OpSpec((LIST,), INT, lambda xs: sum(1 for v in xs if v % 2 == 0), category="list"),
    "count_odd":  OpSpec((LIST,), INT, lambda xs: sum(1 for v in xs if v % 2 == 1), category="list"),
    "sum_squares": OpSpec((LIST,), INT, lambda xs: sum(v * v for v in xs), category="list"),
    # pairs
    "mkpair":     OpSpec((INT, INT), PAIR, lambda a, b: (a, b), category="pair"),
    "first":      OpSpec((PAIR,), INT, lambda p: p[0], category="pair"),
    "second":     OpSpec((PAIR,), INT, lambda p: p[1], category="pair"),
    "sum_pair":   OpSpec((PAIR,), INT, lambda p: p[0] + p[1], category="pair"),
    "diff_pair":  OpSpec((PAIR,), INT, lambda p: abs(p[0] - p[1]), category="pair"),
    "max_pair":   OpSpec((PAIR,), INT, lambda p: max(p), category="pair"),
    "min_pair":   OpSpec((PAIR,), INT, lambda p: min(p), category="pair"),
}
# fmt: on


def _trunc_div(a: int, b: int) -> int:
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Node:
    op: str
    params: tuple[Any, ...] = ()
    children: tuple["Node", ...] = ()
    out_type: str = ANY          # explicit for var; inferred otherwise

    def __post_init__(self) -> None:
        if self.op not in OPS:
            raise DslError(f"unknown operator {self.op!r}")

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def depth(self) -> int:
        return 1 + max((c.depth() for c in self.children), default=0)

    def __str__(self) -> str:
        p = ",".join(str(x) for x in self.params)
        args = ",".join(str(c) for c in self.children)
        parts = [s for s in (p, args) if s]
        return f"{self.op}({';'.join(parts)})" if parts else self.op


def var(type_name: str = INT) -> Node:
    if type_name not in _PRIMITIVE_TYPES:
        raise DslError(f"invalid var type {type_name!r}")
    return Node("var", (type_name,), (), type_name)


def lit(value: int) -> Node:
    return Node("lit", (int(value),), (), INT)


def boolean(value: bool) -> Node:
    return Node("bool", (bool(value),), (), BOOL)


def call(op: str, *children: Node, params: tuple[Any, ...] = ()) -> Node:
    """Build a node with inferred output type (no static validation)."""
    node = Node(op, tuple(params), tuple(children))
    return Node(op, tuple(params), tuple(children), infer_type(node))


# ---------------------------------------------------------------------------
# Type inference and static validation
# ---------------------------------------------------------------------------


def infer_type(node: Node, env: dict[str, str] | None = None) -> str:
    spec = OPS[node.op]
    if node.op == "var":
        t = node.params[0] if node.params else ((env or {}).get("x", INT))
        if t not in _PRIMITIVE_TYPES:
            raise DslError(f"var has invalid type {t!r}")
        return t
    if node.op in ("lit", "bool"):
        return spec.out
    expected = spec.children
    if len(node.children) != len(expected):
        raise DslError(f"{node.op}: expected {len(expected)} children, got {len(node.children)}")
    for child, want in zip(node.children, expected):
        got = infer_type(child, env)
        if want not in (ANY, got):
            raise DslError(f"{node.op}: child type {got} != expected {want}")
    if spec.out == "SAME":
        a, b = infer_type(node.children[1], env), infer_type(node.children[2], env)
        if a != b:
            raise DslError(f"if: branch types disagree ({a} vs {b})")
        return a
    return spec.out


def validate(node: Node, env: dict[str, str] | None = None, *,
             max_depth: int = 8, max_nodes: int = 64,
             max_magnitude: int = 10 ** 9) -> list[str]:
    """Static validation; returns a list of issues (empty == well-formed)."""
    issues: list[str] = []

    def walk(n: Node, depth: int) -> None:
        if depth > max_depth:
            issues.append(f"depth {depth} > {max_depth}")
        if n.size() > max_nodes:
            issues.append(f"node count {n.size()} > {max_nodes}")
        if n.op in ("lit",) and n.params and abs(int(n.params[0])) > max_magnitude:
            issues.append(f"literal magnitude exceeds {max_magnitude}")
        if n.op in ("add", "sub", "mul", "mod", "div", "map_add", "map_mul",
                    "filter_gt", "count_gt", "take", "drop", "band", "bor",
                    "bxor", "shl", "shr", "lt", "le", "gt", "ge", "eq", "ne") \
                and n.params and abs(int(n.params[0])) > max_magnitude:
            issues.append(f"parameter magnitude exceeds {max_magnitude}")
        try:
            infer_type(n, env)
        except DslError as exc:
            issues.append(str(exc))
        for c in n.children:
            walk(c, depth + 1)

    walk(node, 1)
    return issues


def well_typed(node: Node, env: dict[str, str] | None = None) -> bool:
    try:
        infer_type(node, env)
        return True
    except DslError:
        return False


# ---------------------------------------------------------------------------
# Canonicalization, complexity, safety keys
# ---------------------------------------------------------------------------


def canonical(node: Node) -> tuple:
    """Stable structural key; commutative operands are sorted."""
    spec = OPS[node.op]
    kids = [canonical(c) for c in node.children]
    if spec.commutative:
        kids.sort(key=repr)
    return (node.op, tuple(node.params), tuple(kids))


def canonical_key(node: Node) -> str:
    return stable_key(canonical(node))


def complexity(node: Node) -> float:
    """Cost estimate used by search. Parameterized ops cost slightly more."""
    total = 1.0
    for c in node.children:
        total += complexity(c)
    if OPS[node.op].has_param:
        total += 0.5
    if node.op in ("if", "band", "bor", "bxor", "shl", "shr"):
        total += 1.0
    return total


def cycles_possible() -> bool:
    """Immutable trees cannot contain cycles; explicit for documentation/tests."""
    return False


# ---------------------------------------------------------------------------
# Safe execution
# ---------------------------------------------------------------------------


def evaluate(node: Node, x: Any, *, max_steps: int = 10_000,
             max_depth: int = 24, max_magnitude: int = 10 ** 9) -> Any:
    """Bounded, side-effect-free evaluation. Raises DslError on any violation."""
    steps = [0]

    def tick() -> None:
        steps[0] += 1
        if steps[0] > max_steps:
            raise DslError("step budget exhausted")

    def check(v: Any) -> Any:
        if isinstance(v, bool):
            return v
        if isinstance(v, int) and abs(v) > max_magnitude:
            raise DslError("integer magnitude exceeded")
        if isinstance(v, list):
            if len(v) > 10000:
                raise DslError("list length exceeded")
            for e in v:
                if isinstance(e, bool) or not isinstance(e, int) or abs(e) > max_magnitude:
                    raise DslError("list element out of bounds")
        return v

    def ev(n: Node, d: int) -> Any:
        tick()
        if d > max_depth:
            raise DslError("depth budget exhausted")
        spec = OPS[n.op]
        if n.op == "var":
            return x
        if n.op == "lit":
            return check(_need_int(n.params[0]))
        if n.op == "bool":
            return bool(n.params[0])
        args = [ev(c, d + 1) for c in n.children]
        if n.op == "if":
            return check(args[1] if _need_bool(args[0]) else args[2])
        if n.op == "fact":
            return check(_bounded_fact(_need_int(args[0]), max_magnitude))
        try:
            if spec.has_param:
                k = n.params[0] if n.params else 0
                result = spec.eval(*args, k)  # type: ignore[misc]
            else:
                result = spec.eval(*args)  # type: ignore[misc]
        except DslError:
            raise
        except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
            raise DslError(f"{n.op} runtime type/value failure: {exc}") from exc
        # normalize list ops that return generators/sorted lists of ints
        if isinstance(result, list):
            result = [check(e) for e in result]
        return check(result)

    return ev(node, 0)


def behavioral_fingerprint(node: Node, probes: list[Any], **kw: Any) -> tuple:
    """Outputs on a fixed diagnostic set; used by retrieval ambiguity detection."""
    out: list[Any] = []
    for p in probes:
        try:
            out.append(evaluate(node, p, **kw))
        except DslError as exc:
            out.append(f"ERR:{exc}")
    return tuple(out)


# ---------------------------------------------------------------------------
# Expressiveness classification and hidden-program translation (read-only)
# ---------------------------------------------------------------------------

_EXCLUSION_OPS = {"band", "bor", "bxor", "shl", "shr"}
_CONDITIONAL_OPS = {"if", "gt", "lt", "le", "ge", "eq", "ne", "and", "or", "not"}


_OP_ALIASES = {"if_else": "if", "shl": "shl", "shr": "shr"}


def expressiveness_of_ops(ops: set[str], *, budget_ok: bool = True,
                          identifiable: bool = True) -> Expressiveness:
    normalized = {_OP_ALIASES.get(op, op) for op in ops}
    if not normalized <= set(OPS):
        return Expressiveness.NOT_EXPRESSIBLE
    if not identifiable:
        return Expressiveness.NOT_IDENTIFIABLE
    if not budget_ok:
        return Expressiveness.COMPUTE_LIMITED
    return Expressiveness.EXPRESSIBLE_DISCOVERED


def from_hidden(hidden: Any, env: dict[str, str] | None = None) -> Node:
    """Translate an EXP-H-style hidden program (op/param/children) into a DSL Node.

    Accepts any object with ``.op``, ``.param`` and ``.children`` attributes. This
    is a *read-only* structural import: the source program is never modified.
    ``if_else`` maps to ``if``; bitwise and conditional ops map directly.
    """
    env = env or {"x": INT}

    def conv(h: Any) -> Node:
        op = h.op
        if op == "var":
            return var(env.get("x", INT))
        if op in ("not", "and", "or"):
            kids = tuple(conv(c) for c in h.children)
            return call(op, *kids)
        if op == "if_else":
            kids = tuple(conv(c) for c in h.children)
            return call("if", *kids)
        if op in ("gt", "lt"):
            return call(op, conv(h.children[0]), params=(int(h.param or 0),))
        if op in OPS and OPS[op].has_param:
            return call(op, conv(h.children[0]), params=(int(h.param or 0),))
        if op in OPS:
            return call(op, *(conv(c) for c in h.children))
        raise DslError(f"hidden op {op!r} has no DSL equivalent")

    return conv(hidden)


def hidden_op_set(hidden: Any) -> set[str]:
    out: set[str] = set()

    def walk(h: Any) -> None:
        out.add(h.op)
        for c in getattr(h, "children", ()):
            walk(c)

    walk(hidden)
    return out
