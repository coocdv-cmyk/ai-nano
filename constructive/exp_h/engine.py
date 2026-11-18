"""Hidden-program engine for EXP-H.

This is *not* a second implementation of the CCM DSL; it is a deliberately
richer target language used to generate the hidden programs of H1/H2/H6/H7.
Programs are inert data (typed tuples), evaluated by a step-bounded recursive
interpreter -- never by ``eval`` or ``exec``.

Type system: int, bool, list, pair, entity, scene are not all supported here;
supported types are int, bool and list (of int), pair.  Anything else is a
generator bug, not silent coercion.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

INT = "int"
BOOL = "bool"
LIST = "list"
PAIR = "pair"


@dataclass(frozen=True)
class HNode:
    op: str
    param: Any = None
    children: tuple["HNode", ...] = ()

    def __str__(self) -> str:
        inner = ",".join(str(c) for c in self.children)
        return f"{self.op}({self.param},{inner})" if self.param is not None else f"{self.op}({inner})"


class HiddenError(Exception):
    """Undefined or unsupported operation on a hidden program."""


MAX_NODES = 200
MAX_DEPTH = 24
MAX_STEPS = 20_000


# ---------------------------------------------------------------------------
# Op registry
# ---------------------------------------------------------------------------


def _bound_fact(n: int) -> int:
    acc = 1
    for i in range(2, n + 1):
        acc *= i
    return acc


_INT_UNARY = {
    "add1": lambda x: x + 1,
    "sub1": lambda x: x - 1,
    "double": lambda x: x * 2,
    "square": lambda x: x * x,
    "neg": lambda x: -x,
    "abs": abs,
    "sum_digits": lambda x: sum(int(c) for c in str(abs(x))),
    "digital_root": lambda x: 0 if x == 0 else 1 + (abs(x) - 1) % 9,
    "fact": lambda x: 1 if x <= 0 else _bound_fact(min(x, 16)),
}


_LIST_INT = {
    "sum_list": sum,
    "max_list": lambda xs: max(xs) if xs else 0,
    "min_list": lambda xs: min(xs) if xs else 0,
    "len_list": len,
    "head": lambda xs: xs[0] if xs else 0,
    "last": lambda xs: xs[-1] if xs else 0,
    "range_list": lambda xs: (max(xs) - min(xs) if xs else 0),
    "count_even": lambda xs: sum(1 for x in xs if x % 2 == 0),
    "count_odd": lambda xs: sum(1 for x in xs if x % 2 == 1),
    "sum_squares": lambda xs: sum(x * x for x in xs),
}

_LIST_LIST = {
    "reverse": lambda xs: list(reversed(xs)),
    "sort": sorted,
    "sort_desc": lambda xs: sorted(xs, reverse=True),
    "unique": lambda xs: list(dict.fromkeys(xs)),
    "evens": lambda xs: [x for x in xs if x % 2 == 0],
    "odds": lambda xs: [x for x in xs if x % 2 == 1],
}

_BITWISE = ("shl", "shr", "band", "bor", "bxor")

# CCM-expressible subset tag (informational only; never used to influence search)
CCM_OPS = {
    "add1", "sub1", "double", "square", "neg", "abs", "sum_digits",
    "digital_root", "is_prime", "even", "odd",
    "sum_list", "max_list", "min_list", "len_list", "head", "last",
    "range_list", "count_even", "count_odd", "sum_squares",
    "reverse", "sort", "sort_desc", "unique", "evens", "odds",
}


class _StepCounter:
    def __init__(self, budget: int = MAX_STEPS):
        self.left = budget

    def tick(self) -> None:
        self.left -= 1
        if self.left < 0:
            raise HiddenError("step budget exhausted")


def eval_hidden(node: Any, x: Any, steps: _StepCounter | None = None, depth: int = 0) -> Any:
    """Evaluate a hidden program on one input, with bounded steps and depth."""
    steps = steps or _StepCounter()
    steps.tick()
    if depth > MAX_DEPTH:
        raise HiddenError("depth budget exhausted")
    if not isinstance(node, HNode):
        raise HiddenError("hidden program is not an HNode")
    op = node.op
    if node.children:
        args = [eval_hidden(c, x, steps, depth + 1) for c in node.children]
    else:
        args = ()
    if op == "var":
        return x

    # --- integer unary ---
    if op in _INT_UNARY:
        return _INT_UNARY[op](_need_int(args[0]))
    if op in ("add", "mul", "sub", "mod"):
        k = _need_int(node.param)
        a = _need_int(args[0])
        if op == "add":
            return a + k
        if op == "sub":
            return a - k
        if op == "mul":
            return a * k
        return a % k if k else 0

    # --- bitwise (not in CCM's vocabulary) ---
    if op in _BITWISE:
        k = int(node.param or 0)
        a = int(args[0])
        if k > 31:
            k = k % 32
        if op == "shl":
            return (a << k) & 0xFFFFFFFF
        if op == "shr":
            return (a & 0xFFFFFFFF) >> k
        if op == "band":
            return a & k
        if op == "bor":
            return a | k
        return a ^ k

    # --- lists ---
    if op in _LIST_INT:
        xs = _need_list(args[0])
        return _LIST_INT[op](xs)
    if op in _LIST_LIST:
        xs = args[0] if isinstance(args[0], list) else _need_list(args[0])
        return _LIST_LIST[op](list(xs))
    if op in ("take", "drop", "filter_gt", "count_gt", "map_add", "map_mul", "map_square", "map_cube"):
        xs = _need_list(args[0])
        k = int(node.param or 0)
        if op == "take":
            return xs[:k]
        if op == "drop":
            return xs[k:]
        if op == "filter_gt":
            return [v for v in xs if v > k]
        if op == "count_gt":
            return sum(1 for v in xs if v > k)
        if op == "map_add":
            return [v + k for v in xs]
        if op == "map_mul":
            return [v * k for v in xs]
        if op == "map_square":
            return [v * v for v in xs]
        if op == "map_cube":
            return [v * v * v for v in xs]

    # --- bools ---
    if op == "even":
        return _need_int(args[0]) % 2 == 0
    if op == "odd":
        return _need_int(args[0]) % 2 == 1
    if op == "is_prime":
        return _is_prime(_need_int(args[0]))
    if op in ("gt", "lt"):
        k = int(node.param or 0)
        a = _need_int(args[0])
        return (a > k) if op == "gt" else (a < k)
    if op == "not":
        return not bool(args[0])
    if op == "and":
        return bool(args[0]) and bool(args[1])
    if op == "or":
        return bool(args[0]) or bool(args[1])

    # --- pairs ---
    if op == "mkpair":
        return (_need_int(args[0]), _need_int(args[1]))
    if op in ("first", "second", "sum_pair", "diff_pair", "max_pair", "min_pair",
              "gcd_pair"):
        a, b = _need_pair(args[0])
        if op == "first":
            return a
        if op == "second":
            return b
        if op == "sum_pair":
            return a + b
        if op == "diff_pair":
            return abs(a - b)
        if op == "max_pair":
            return max(a, b)
        if op == "min_pair":
            return min(a, b)
        return _gcd(a, b)

    # --- conditional (not in CCM's vocabulary) ---
    if op == "if_else":
        return args[1] if bool(args[0]) else args[2]

    raise HiddenError(f"unknown hidden op {op!r}")


def _need_int(v: Any) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise HiddenError(f"expected int, got {type(v).__name__}")
    return v


def _need_list(v: Any) -> list[int]:
    if not isinstance(v, list):
        raise HiddenError(f"expected list, got {type(v).__name__}")
    for x in v:
        if isinstance(x, bool) or not isinstance(x, int):
            raise HiddenError("list contains a non-int")
    return list(v)


def _need_pair(v: Any) -> tuple[int, int]:
    if not isinstance(v, tuple) or len(v) != 2:
        raise HiddenError(f"expected pair, got {v!r}")
    a, b = v
    if isinstance(a, bool) or isinstance(b, bool):
        raise HiddenError("pair holds a bool")
    return int(a), int(b)


def _gcd(a: int, b: int) -> int:
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


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


# ---------------------------------------------------------------------------
# Ast size
# ---------------------------------------------------------------------------


def hidden_size(node: Any) -> int:
    if not isinstance(node, HNode):
        return 0
    return 1 + sum(hidden_size(c) for c in node.children)
