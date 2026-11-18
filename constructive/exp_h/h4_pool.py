"""Parametrized concept pool for H4 — continuous-learning concept families.

The pool only contains concepts CCM can express.  Parametric entries keep
their parameter visible to example generation.  Constants stay within the
learner's default budget (0-12 for lists, 2-7 for int add/mul) on purpose;
discovering a ``add(11)`` from "y is exactly x+11 for three examples" is a
search question, not a mechanism question, and it is not credited here.
"""

from __future__ import annotations

import random
from typing import Any

from ..core import build_prims, eval_term, Term, V
from ..worlds import Task

_PRIMS_SAMPLE = build_prims(tuple(range(-10, 13)))


def _p(name: str) -> Term:
    return Term(name, (V("x"),))


def int_rule_task(name: str, fn, seed: int, k: int, out_type: str = "int",
                  n_test: int = 20, n_val: int = 1, lo: int = 0,
                  hi: int = 30) -> Task:
    rng = random.Random(seed)
    if hi - lo + 1 < k + n_val + n_test:
        raise ValueError(f"range too small for {name}")
    vals = random.Random(seed).sample(range(lo, hi + 1), k + n_val + n_test)
    train = [(x, fn(x)) for x in vals[:k]]
    val = [(x, fn(x)) for x in vals[k:k + n_val]]
    test = [(x, fn(x)) for x in vals[k + n_val:]]
    probes_pool = [v for v in vals[k + n_val:][:4]]
    if len(set(y for _, y in train + val)) < 2 and k >= 2:
        raise ValueError(f"degenerate labels for {name}")
    return Task("h4", name, name, train, val, test, "int", out_type,
                probes_pool, {"pool_item": name})


def pair_rule_task(name: str, fn, seed: int, k: int, n_test: int = 20,
                   n_val: int = 1) -> Task:
    rng = random.Random(seed)
    pairs = rng.sample([(a, b) for a in range(13) for b in range(13)], k + n_val + n_test)
    train = [(x, fn(x)) for x in pairs[:k]]
    val = [(x, fn(x)) for x in pairs[k:k + n_val]]
    test = [(x, fn(x)) for x in pairs[k + n_val:]]
    probes = [(rng.randint(0, 12), rng.randint(0, 12)) for _ in range(4)]
    if len(set(y for _, y in train + val)) < 2 and k >= 2:
        raise ValueError(f"degenerate labels for {name}")
    return Task("h4", name, name, train, val, test, "pair", "int",
                probes, {"pool_item": name})


def list_expr_task(name: str, term: Term, out_type: str, seed: int, k: int,
                   n_test: int = 20, n_val: int = 1) -> Task:
    rng = random.Random(seed)
    seen: set[tuple[int, ...]] = set()
    batch: list[list[int]] = []
    while len(batch) < k + n_val + n_test:
        x = tuple(rng.randint(0, 9) for _ in range(rng.randint(5, 9)))
        if x in seen:
            continue
        seen.add(x)
        batch.append(list(x))

    def lab(xs):
        try:
            v = eval_term(term, {"x": xs}, _PRIMS_SAMPLE)
            if v is None:
                raise ValueError("none")
            return v
        except Exception:
            return None

    labels = [lab(x) for x in batch]
    if any(y is None for y in labels):
        raise ValueError(f"hidden term cannot be evaluated: {name}")
    if len(set(repr(y) for y in labels)) < 2 and k >= 2:
        raise ValueError(f"degenerate labels for {name}")
    train = [(x, y) for x, y in zip(batch[:k], labels[:k])]
    val = [(x, y) for x, y in zip(batch[k:k + n_val], labels[k:k + n_val])]
    test = [(x, y) for x, y in zip(batch[k + n_val:], labels[k + n_val:])]
    probes = batch[:4]
    return Task("h4", name, name, train, val, test, "list", out_type,
                probes, {"pool_item": name})


def build_pool(n_cap: int = 120) -> list[tuple[str, Any]]:
    """Fixed-order pool.  Every builder is a no-arg seed -> Task factory."""
    from ..worlds import list_rule_task, relational_task

    pool: list[tuple[str, Any]] = []

    pool.append(("math-square", lambda s, k=4: int_rule_task("square", lambda x: x * x, s, k)))
    pool.append(("math-sum_digits", lambda s, k=4: int_rule_task(
        "sum_digits", lambda x: sum(int(c) for c in str(abs(x))), s, k)))
    pool.append(("math-digital_root", lambda s, k=4: int_rule_task(
        "digital_root", lambda x: 0 if x == 0 else 1 + (abs(x) - 1) % 9, s, k)))
    pool.append(("math-fib", lambda s, k=4: int_rule_task("fib", _fib, s, k)))
    pool.append(("math-is_prime", lambda s, k=4: int_rule_task(
        "is_prime", _is_prime, s, k, out_type="bool")))
    pool.append(("math-even", lambda s, k=4: int_rule_task(
        "even", lambda x: x % 2 == 0, s, k, out_type="bool")))
    pool.append(("math-odd", lambda s, k=4: int_rule_task(
        "odd", lambda x: x % 2 == 1, s, k, out_type="bool")))
    pool.append(("math-add1", lambda s, k=4: int_rule_task("add1", lambda x: x + 1, s, k)))
    pool.append(("math-sub1", lambda s, k=4: int_rule_task("sub1", lambda x: x - 1, s, k)))
    pool.append(("math-double", lambda s, k=4: int_rule_task("double", lambda x: x * 2, s, k)))

    for kk in range(2, 8):
        pool.append((f"math-add-{kk}", lambda s, kk=kk, k=4: int_rule_task(
            f"add({kk})", lambda x, kk=kk: x + kk, s + kk, k)))
    for kk in range(2, 6):
        pool.append((f"math-mul-{kk}", lambda s, kk=kk, k=4: int_rule_task(
            f"mul({kk})", lambda x, kk=kk: x * kk, s + kk, k)))

    for rule in ("sum_list", "max_list", "min_list", "range_list", "count_even",
                 "count_odd", "sum_squares", "head", "last", "len_list",
                 "sort", "sort_desc", "unique", "reverse", "evens", "odds"):
        pool.append((f"list-{rule}",
                     lambda s, rule=rule, k=4: list_rule_task(rule, seed=s + 81000, k=k)))

    pool.append(("math-neg", lambda s, k=4: int_rule_task("neg", lambda x: -x, s, k, lo=-15, hi=15)))
    pool.append(("math-abs", lambda s, k=4: int_rule_task("abs", abs, s, k, lo=-15, hi=15)))
    for kk in (2, 5, 8):
        pool.append((f"math-sub-{kk}", lambda s, kk=kk, k=4: int_rule_task(
            f"sub({kk})", lambda x, kk=kk: x - kk, s + kk, k)))
    for kk in (3, 4, 5):
        pool.append((f"math-mod-{kk}", lambda s, kk=kk, k=4: int_rule_task(
            f"mod({kk})", lambda x, kk=kk: x % kk, s + kk, k)))

    for kk in range(1, 7):
        pool.append((f"list-count_gt-{kk}",
                     lambda s, kk=kk, k=4: list_expr_task(
                         f"count_gt({kk})", _p(f"count_gt({kk})"), "int",
                         s + kk, k)))
    for kk in range(1, 7):
        pool.append((f"list-take-{kk}",
                     lambda s, kk=kk, k=4: list_expr_task(
                         f"take({kk})", _p(f"take({kk})"), "list", s + kk, k)))
    for kk in range(2, 7):
        pool.append((f"list-filter_gt-{kk}",
                     lambda s, kk=kk, k=4: list_expr_task(
                         f"filter_gt({kk})", _p(f"filter_gt({kk})"), "list", s + kk, k)))
    for kk in range(7, 10):
        pool.append((f"list-count_gt-{kk}",
                     lambda s, kk=kk, k=4: list_expr_task(
                         f"count_gt({kk})", _p(f"count_gt({kk})"), "int",
                         s + 7000 + kk, k)))
    for kk in (1, 2, 3, 4):
        pool.append((f"list-map_add-{kk}",
                     lambda s, kk=kk, k=4: list_expr_task(
                         f"map_add({kk})", _p(f"map_add({kk})"), "list",
                         s + 6000 + kk, k)))
    for kk in (1, 2, 3):
        pool.append((f"list-map_mul-{kk}",
                     lambda s, kk=kk, k=4: list_expr_task(
                         f"map_mul({kk})", _p(f"map_mul({kk})"), "list",
                         s + 5000 + kk, k)))

    # scene relations
    for rule in ("left_of", "right_of", "above", "below", "same_color",
                 "same_shape", "nearest", "farthest", "biggest", "smallest"):
        pool.append((f"scene-{rule}",
                     lambda s, rule=rule, k=4: relational_task(rule, seed=s + 82000, k=k)))

    # pair arithmetic
    for name, fn in (("sum", lambda p: p[0] + p[1]),
                     ("diff", lambda p: abs(p[0] - p[1])),
                     ("gcd", lambda p: _gcd(p[0], p[1])),
                     ("max", max),
                     ("min", min)):
        pool.append((f"pair-{name}_pair",
                     lambda s, name=name, fn=fn, k=4: pair_rule_task(
                         f"{name}_pair", fn, s, k)))
    return pool[:n_cap]


def _gcd(a: int, b: int) -> int:
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def _fib(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
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
