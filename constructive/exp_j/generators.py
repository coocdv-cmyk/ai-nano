"""EXP-J independent task generators.

Written from scratch for EXP-J. No generator, label function or construction
shortcut is imported from EXP-H or EXP-I. The only EXP-I code used here is the
*DSL constructor* (``call``/``var``), which is the neutral interface through
which any hypothesis is presented to the mechanism under test.

Every rule family has:
  * an independent pure-Python reference label function,
  * a set of candidate programs (one semantically correct, several confusable),
  * disjoint train / validation / test domains,
  * optional numerical and structural distribution shift.
"""

from __future__ import annotations

import random
from typing import Any, Callable

from constructive.exp_i import dsl
from constructive.exp_i.dsl import Node, call, var
from constructive.exp_i.types import Candidate, Example, Task

from .types import ShiftSpec

INT_X = var("int")
LIST_X = var("list")


# ---------------------------------------------------------------------------
# Independent sampling helpers (own implementations)
# ---------------------------------------------------------------------------


def _uniq_ints(rng: random.Random, n: int, lo: int, hi: int,
               exclude: set[int] | None = None) -> list[int]:
    excluded = set(exclude or ())
    span = hi - lo + 1
    if n <= span:
        pool = [v for v in range(lo, hi + 1) if v not in excluded]
        rng.shuffle(pool)
        return pool[:n]
    out: list[int] = []
    guard = 0
    while len(out) < n and guard < 100000:
        guard += 1
        v = rng.randint(lo, hi)
        if v not in excluded and v not in out:
            out.append(v)
    return out


def _uniq_lists(rng: random.Random, n: int, lo_len: int, hi_len: int,
                lo: int, hi: int, exclude: set[tuple[int, ...]] | None = None,
                allow_empty: bool = False) -> list[list[int]]:
    excluded = set(exclude or ())
    out: list[list[int]] = []
    guard = 0
    while len(out) < n and guard < 200000:
        guard += 1
        length = rng.randint(max(1 if not allow_empty else 0, lo_len), hi_len)
        xs = tuple(rng.randint(lo, hi) for _ in range(length))
        if xs in excluded or xs in {tuple(o) for o in out}:
            continue
        out.append(list(xs))
    return out


def _uniq_pairs(rng: random.Random, n: int, lo: int, hi: int,
                exclude: set[tuple[int, int]] | None = None) -> list[tuple[int, int]]:
    excluded = set(exclude or ())
    out: list[tuple[int, int]] = []
    guard = 0
    while len(out) < n and guard < 100000:
        guard += 1
        p = (rng.randint(lo, hi), rng.randint(lo, hi))
        if p not in excluded and p not in out:
            out.append(p)
    return out


def disjoint(*groups: list[Any]) -> bool:
    keys = [{repr(x) for x in g} for g in groups]
    return all(not (keys[i] & keys[j]) for i in range(len(keys))
               for j in range(i + 1, len(keys)))


# ---------------------------------------------------------------------------
# Independent reference label functions (no import from EXP-I generators)
# ---------------------------------------------------------------------------


def _prime(n: int) -> bool:
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


def _digits(n: int) -> list[int]:
    return [int(ch) for ch in str(abs(n))]


def _fib(n: int) -> int:
    a, b = 0, 1
    for _ in range(max(0, min(n, 30))):
        a, b = b, a + b
    return a


def _range(xs: list[int]) -> int:
    return (max(xs) - min(xs)) if xs else 0


def _unique(xs: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for v in xs:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


# Each label function takes (x, params)->label. These are independent of the
# DSL evaluator; a mismatch would be a generator bug and is asserted at build.
LABELS: dict[str, Callable[[Any, dict[str, int]], Any]] = {
    "affine_mod": lambda x, p: (p["a"] * x + p["b"]) % p["m"],
    "digit_sum_parity": lambda x, p: (sum(_digits(x)) % 2 == 0),
    "threshold_parity": lambda x, p: (x > p["k"]) and (x % 2 == 1),
    "prime_and_odd": lambda x, p: _prime(x) and (x % 2 == 1),
    "fib_mod": lambda x, p: _fib(x) % p["m"],
    "square_digit_sum": lambda x, p: sum(_digits(x * x)),
    "range_of_evens": lambda xs, p: _range([v for v in xs if v % 2 == 0]),
    "sum_top_k": lambda xs, p: sum(sorted(xs, reverse=True)[:p["k"]]),
    "count_distinct_gt": lambda xs, p: sum(1 for v in _unique(xs) if v > p["k"]),
    "prefix_sum_gt": lambda xs, p: (sum(xs[:p["k"]]) > p["t"]),
    "evens_of_even": lambda xs, p: sum(1 for v in xs if v % 2 == 0),
    "weighted_pair": lambda xy, p: p["a"] * xy[0] + p["b"],
}

# Candidate programs: (id, node). The entry whose id == "true" is the correct
# hypothesis; the rest are plausible confusables. Operator combinations marked
# here are deliberately unseen in EXP-H/EXP-I generator families.
UNSEEN_OPS: dict[str, tuple[str, ...]] = {
    "affine_mod": ("mul", "add", "mod"),
    "threshold_parity": ("gt", "and", "odd"),
    "prime_and_odd": ("is_prime", "not", "and"),
    "fib_mod": ("fib", "mod"),
    "square_digit_sum": ("square", "sum_digits"),
    "range_of_evens": ("evens", "range_list"),
    "sum_top_k": ("sort_desc", "take", "sum_list"),
    "count_distinct_gt": ("unique", "count_gt"),
    "prefix_sum_gt": ("take", "sum_list", "gt"),
    "weighted_pair": ("first", "mul", "add"),
    "digit_sum_parity": ("sum_digits", "even"),
    "evens_of_even": ("evens", "count_even"),
}

INPUT_TYPE: dict[str, str] = {
    "affine_mod": "int", "digit_sum_parity": "int",
    "threshold_parity": "int", "prime_and_odd": "int",
    "fib_mod": "int", "square_digit_sum": "int",
    "range_of_evens": "list", "sum_top_k": "list",
    "count_distinct_gt": "list", "prefix_sum_gt": "list",
    "evens_of_even": "list", "weighted_pair": "pair",
}

OUTPUT_TYPE: dict[str, str] = {
    "digit_sum_parity": "bool", "threshold_parity": "bool",
    "prime_and_odd": "bool", "prefix_sum_gt": "bool",
}


def _programs(rule: str, p: dict[str, int]) -> list[tuple[str, Node]]:
    x, xs = INT_X, LIST_X
    if rule == "affine_mod":
        return [
            ("true", call("mod", call("add", call("mul", x, params=(p["a"],)),
                                      params=(p["b"],)), params=(p["m"],))),
            ("ignore_affine", call("mod", x, params=(p["m"],))),
            ("off_by_one", call("mod", call("add", call("mul", x, params=(p["a"],)),
                                            params=(p["b"] + 1,)), params=(p["m"],))),
        ]
    if rule == "digit_sum_parity":
        return [
            ("true", call("even", call("sum_digits", x))),
            ("digital_root_parity", call("even", call("digital_root", x))),
            ("odd_digit_sum", call("odd", call("sum_digits", x))),
        ]
    if rule == "threshold_parity":
        return [
            ("true", call("and", call("gt", x, params=(p["k"],)), call("odd", x))),
            ("or_variant", call("or", call("gt", x, params=(p["k"],)), call("odd", x))),
            ("threshold_only", call("gt", x, params=(p["k"],))),
        ]
    if rule == "prime_and_odd":
        return [
            ("true", call("and", call("is_prime", x),
                          call("not", call("even", x)))),
            ("prime_only", call("is_prime", x)),
            ("odd_only", call("odd", x)),
        ]
    if rule == "fib_mod":
        return [
            ("true", call("mod", call("fib", x), params=(p["m"],))),
            ("square_mod", call("mod", call("square", x), params=(p["m"],))),
            ("raw_fib", call("fib", x)),
        ]
    if rule == "square_digit_sum":
        return [
            ("true", call("sum_digits", call("square", x))),
            ("plain_digit_sum", call("sum_digits", x)),
            ("double_digit_sum", call("sum_digits", call("double", x))),
        ]
    if rule == "range_of_evens":
        return [
            ("true", call("range_list", call("evens", xs))),
            ("full_range", call("range_list", xs)),
            ("max_even", call("max_list", call("evens", xs))),
        ]
    if rule == "sum_top_k":
        return [
            ("true", call("sum_list", call("take", call("sort_desc", xs),
                                           params=(p["k"],)))),
            ("sum_bottom_k", call("sum_list", call("take", call("sort", xs),
                                                   params=(p["k"],)))),
            ("max_only", call("max_list", xs)),
        ]
    if rule == "count_distinct_gt":
        return [
            ("true", call("count_gt", call("unique", xs), params=(p["k"],))),
            ("count_all_gt", call("count_gt", xs, params=(p["k"],))),
            ("distinct_count", call("len_list", call("unique", xs))),
        ]
    if rule == "prefix_sum_gt":
        return [
            ("true", call("gt", call("sum_list", call("take", xs, params=(p["k"],))),
                          params=(p["t"],))),
            ("full_sum_gt", call("gt", call("sum_list", xs), params=(p["t"],))),
            ("last_gt", call("gt", call("last", xs), params=(p["t"],))),
        ]
    if rule == "evens_of_even":
        return [
            ("true", call("count_even", xs)),
            ("odds_count", call("count_odd", xs)),
            ("len_all", call("len_list", xs)),
        ]
    if rule == "weighted_pair":
        pr = var("pair")
        return [
            ("true", call("add", call("mul", call("first", pr), params=(p["a"],)),
                          params=(p["b"],))),
            ("plain_sum", call("sum_pair", pr)),
            ("first_scaled", call("mul", call("first", pr), params=(p["a"],))),
        ]
    raise ValueError(f"unknown rule {rule}")


ALL_RULES = tuple(LABELS)


def _params(rule: str, rng: random.Random) -> dict[str, int]:
    if rule == "affine_mod":
        return {"a": rng.randint(2, 5), "b": rng.randint(-6, 6), "m": rng.randint(7, 13)}
    if rule == "threshold_parity":
        return {"k": rng.randint(0, 20)}
    if rule == "fib_mod":
        return {"m": rng.randint(3, 11)}
    if rule == "sum_top_k":
        return {"k": rng.randint(2, 3)}
    if rule == "count_distinct_gt":
        return {"k": rng.randint(1, 6)}
    if rule == "prefix_sum_gt":
        return {"k": rng.randint(2, 4), "t": rng.randint(4, 18)}
    if rule == "weighted_pair":
        return {"a": rng.randint(2, 4), "b": rng.randint(-5, 5)}
    return {}


def _sample(rule: str, rng: random.Random, n: int, domain: dict[str, Any],
            exclude: list[Any] | None = None) -> list[Any]:
    """Sample n unique inputs in a domain description."""
    in_t = INPUT_TYPE[rule]
    excluded = exclude or []
    if in_t == "int":
        lo, hi = domain["lo"], domain["hi"]
        return _uniq_ints(rng, n, lo, hi, {int(v) for v in excluded})
    if in_t == "pair":
        lo, hi = domain["lo"], domain["hi"]
        return _uniq_pairs(rng, n, lo, hi, {tuple(v) for v in excluded})
    lo_len, hi_len = domain["len"], domain["len_hi"]
    lo, hi = domain["lo"], domain["hi"]
    return _uniq_lists(rng, n, lo_len, hi_len, lo, hi,
                       {tuple(v) for v in excluded})


def default_domains(rule: str, shift: str = "none") -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (train_domain, test_domain) for a named distribution shift."""
    in_t = INPUT_TYPE[rule]
    if in_t == "int":
        train = {"lo": 0, "hi": 9}
        if shift == "magnitude":
            return train, {"lo": 40, "hi": 120}
        if shift == "sign":
            return train, {"lo": -120, "hi": -20}
        if shift == "wide":
            return train, {"lo": -60, "hi": 140}
        return train, {"lo": 10, "hi": 60}
    if in_t == "pair":
        train = {"lo": 0, "hi": 9}
        if shift == "magnitude":
            return train, {"lo": 50, "hi": 120}
        if shift == "sign":
            return train, {"lo": -60, "hi": -10}
        return train, {"lo": 10, "hi": 50}
    train = {"len": 2, "len_hi": 5, "lo": 0, "hi": 9}
    if shift == "length":
        return train, {"len": 8, "len_hi": 14, "lo": 0, "hi": 9}
    if shift == "magnitude":
        return train, {"len": 2, "len_hi": 5, "lo": 40, "hi": 120}
    if shift == "both":
        return train, {"len": 8, "len_hi": 14, "lo": 40, "hi": 120}
    return train, {"len": 3, "len_hi": 7, "lo": 0, "hi": 15}


def build_task(rule: str, seed: int, train_d: dict[str, Any],
               val_d: dict[str, Any], test_d: dict[str, Any], *, k: int = 4,
               n_val: int = 4, n_test: int = 12, tag: str = "domains") -> Task:
    """Build a task from explicit train / validation / test domains."""
    if rule not in LABELS:
        raise ValueError(f"unknown rule {rule}")
    rng = random.Random(seed * 1000003 + sum(ord(c) for c in rule))
    p = _params(rule, rng)
    train_x = _sample(rule, rng, k, train_d)
    val_x = _sample(rule, rng, n_val, val_d, train_x)
    test_x = _sample(rule, rng, n_test, test_d, train_x + val_x)
    if not (disjoint(train_x, val_x, test_x)):
        raise AssertionError("generator split overlap")
    label = LABELS[rule]
    train = [Example(x, label(x, p)) for x in train_x]
    val = [Example(x, label(x, p)) for x in val_x]
    test = [Example(x, label(x, p)) for x in test_x]

    programs = _programs(rule, p)
    # Generator self-check: the semantically correct program and the independent
    # reference label function must coincide on every sampled point. Distractor
    # programs are deliberately *not* checked (they are confusables).
    true_prog = dict(programs)["true"]
    for ex in train + val + test:
        got = dsl.evaluate(true_prog, ex.x)
        if got != ex.y:
            raise AssertionError(
                f"generator mismatch {rule}/true at {ex.x!r}: {got!r} != {ex.y!r}")
    # Distractors are deliberately plausible: they may coincide on the observed
    # support and only separate off-domain. That is exactly what J1 measures.

    cands = [Candidate(id=cid, program=prog, cost=dsl.complexity(prog))
             for cid, prog in programs]
    # Splits and candidates are attached as metadata; test labels are never
    # exposed to any candidate builder (candidates are fixed by rule+params).
    task = Task(
        name=f"{rule}-{tag}-s{seed}", family=rule,
        train=train, validation_pool=val, test=test,
        input_type=INPUT_TYPE[rule],
        output_type=OUTPUT_TYPE.get(rule, "int"),
        meta={"rule": rule, "params": p, "shift": tag,
              "unseen_ops": list(UNSEEN_OPS.get(rule, ())),
              "candidates": cands, "n_train": len(train),
              "n_val": len(val), "n_test": len(test),
              "train_domain": dict(train_d), "test_domain": dict(test_d)},
    )
    return task


def make_task(rule: str, seed: int, *, shift: str = "none", k: int = 4,
              n_val: int = 4, n_test: int = 12,
              spec: ShiftSpec | None = None) -> Task:
    """Build one task with independently generated disjoint splits."""
    train_d, test_d = default_domains(rule, shift)
    if spec is not None:
        train_d, test_d = spec.train_domain, spec.test_domain
    return build_task(rule, seed, train_d, train_d, test_d,
                      k=k, n_val=n_val, n_test=n_test, tag=shift)


def candidates_for(task: Task) -> list[Candidate]:
    """Return a *copy* of the fixed candidate set. No test labels involved."""
    return [Candidate(id=c.id, program=c.program, cost=c.cost)
            for c in task.meta["candidates"]]


def correct_id(task: Task) -> str:
    return "true"


# ---------------------------------------------------------------------------
# Parameterized concepts under different input distributions (J1/J2)
# ---------------------------------------------------------------------------


def parameterized_stream(rule: str, seeds: tuple[int, ...], shifts: tuple[str, ...],
                         k: int = 4, n_test: int = 12) -> list[Task]:
    """Same concept family, varying parameters and input distributions."""
    out: list[Task] = []
    for seed in seeds:
        for shift in shifts:
            out.append(make_task(rule, seed * 17 + len(shift), shift=shift,
                                 k=k, n_test=n_test))
    return out


# ---------------------------------------------------------------------------
# Abstraction streams (J4): shared-core vs unrelated task families
# ---------------------------------------------------------------------------


def abstraction_stream_j(seed: int, family: str = "shared-core",
                         n_tasks: int = 6, k: int = 4, n_test: int = 8) -> list[Task]:
    """Independent stream with a shared transform core or unrelated cores.

    ``shared-core`` tasks all reuse ``sort_desc -> take(k)``; reducers vary.
    ``unrelated`` tasks each use a different core, so reuse should not help.
    """
    rng = random.Random(seed * 7919 + sum(ord(c) for c in family))
    kparam = rng.randint(2, 3)
    cores = {
        "shared-core": lambda xs: call("evens", xs),
        "unrelated": None,
    }
    if family not in cores:
        raise ValueError(family)
    reducers = ("sum_list", "len_list", "max_list", "min_list", "sum_squares", "range_list")
    tasks: list[Task] = []
    for idx in range(n_tasks):
        if family == "shared-core":
            core = cores[family](LIST_X)
        else:
            core_ops = (
                call("unique", LIST_X), call("map_add", LIST_X, params=(1,)),
                call("odds", LIST_X), call("take", LIST_X, params=(3,)),
                call("sort", LIST_X), call("reverse", LIST_X),
            )
            core = core_ops[idx % len(core_ops)]
        red = reducers[idx % len(reducers)]
        prog = call(red, core)
        xs = _uniq_lists(rng, k + 4 + n_test, 3, 7, 0, 9)
        label = lambda v: dsl.evaluate(prog, v)
        train = [Example(x, label(x)) for x in xs[:k]]
        val = [Example(x, label(x)) for x in xs[k:k + 4]]
        test = [Example(x, label(x)) for x in xs[k + 4:]]
        tasks.append(Task(
            name=f"absj-{family}-{red}-s{seed}", family=f"abstraction-{family}",
            train=train, validation_pool=val, test=test,
            input_type="list", output_type="int",
            meta={"hidden": str(prog), "core": str(core), "reducer": red,
                  "index": idx, "family": family},
        ))
    return tasks
