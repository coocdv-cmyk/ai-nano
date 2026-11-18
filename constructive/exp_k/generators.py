"""EXP-K independent generators.

Written from scratch for EXP-K. No generator, label function or construction
shortcut is imported from EXP-H, EXP-I or EXP-J. The only reused EXP-I code is
the neutral DSL constructor used to express hypotheses to the system under test.
"""

from __future__ import annotations

import random
from typing import Any, Callable

from constructive.exp_i import dsl
from constructive.exp_i.dsl import Node, call, var
from constructive.exp_i.types import Candidate, Example, Task

INT_X, LIST_X, PAIR_X = var("int"), var("list"), var("pair")


def _uniq_ints(rng, n, lo, hi, exclude=()):
    ex = set(exclude)
    pool = [v for v in range(lo, hi + 1) if v not in ex]
    if n <= len(pool):
        rng.shuffle(pool)
        return pool[:n]
    out, guard = [], 0
    while len(out) < n and guard < 100000:
        guard += 1
        v = rng.randint(lo, hi)
        if v not in ex and v not in out:
            out.append(v)
    return out


def _uniq_lists(rng, n, lo_len, hi_len, lo, hi, exclude=()):
    ex = {tuple(x) for x in exclude}
    out, guard = [], 0
    while len(out) < n and guard < 200000:
        guard += 1
        xs = tuple(rng.randint(lo, hi) for _ in range(rng.randint(lo_len, hi_len)))
        if xs in ex or xs in {tuple(o) for o in out}:
            continue
        out.append(list(xs))
    return out


def _uniq_pairs(rng, n, lo, hi, exclude=()):
    ex = set(exclude)
    out, guard = [], 0
    while len(out) < n and guard < 100000:
        guard += 1
        p = (rng.randint(lo, hi), rng.randint(lo, hi))
        if p not in ex and p not in out:
            out.append(p)
    return out


def disjoint(*groups):
    keys = [{repr(x) for x in g} for g in groups]
    return all(not (keys[i] & keys[j]) for i in range(len(keys))
               for j in range(i + 1, len(keys)))


def _droot(n: int) -> int:
    return 0 if n == 0 else 1 + (abs(n) - 1) % 9


# Independent reference label functions.
LABELS: dict[str, Callable[[Any, dict[str, int]], Any]] = {
    "mod_double": lambda x, p: 2 * (x % p["m"]),
    "digital_root_square": lambda x, p: _droot(x * x),
    "gt_mod": lambda x, p: (x % p["m"]) > p["k"],
    "if_even": lambda x, p: (2 * x + 1) if x % 2 == 0 else (x + 1),
    "neg_abs_mod": lambda x, p: -(abs(x) % p["m"]),
    "band_shift": lambda x, p: ((x & p["k"]) << (p["s"] % 32)) & 0xFFFFFFFF,
    "max_map_add": lambda xs, p: (max(v + p["k"] for v in xs) if xs else 0),
    "reverse_take_sum": lambda xs, p: sum(list(reversed(xs))[:p["k"]]),
    "count_even_unique": lambda xs, p: sum(1 for v in dict.fromkeys(xs) if v % 2 == 0),
    "sum_squares_evens": lambda xs, p: sum(v * v for v in xs if v % 2 == 0),
    "filter_gt_max": lambda xs, p: (max(v for v in xs if v > p["k"]) if any(
        v > p["k"] for v in xs) else 0),
    "diff_scaled": lambda xy, p: p["k"] * abs(xy[0] - xy[1]),
    "abs_double_mod": lambda x, p: (2 * abs(x)) % p["m"],
    "sumdigits_add": lambda x, p: sum(int(c) for c in str(abs(x))) + p["k"],
    "sub_square": lambda x, p: (x - p["k"]) ** 2,
    "odd_and_gt": lambda x, p: (x % 2 == 1) and (x > p["k"]),
}

INPUT_TYPE = {
    **{r: "int" for r in ("mod_double", "digital_root_square", "gt_mod",
                          "if_even", "neg_abs_mod", "band_shift",
                          "abs_double_mod", "sumdigits_add", "sub_square",
                          "odd_and_gt")},
    **{r: "list" for r in ("max_map_add", "reverse_take_sum", "count_even_unique",
                           "sum_squares_evens", "filter_gt_max")},
    "diff_scaled": "pair",
}
OUTPUT_TYPE = {r: "bool" for r in ("gt_mod", "odd_and_gt")}


def _programs(rule: str, p: dict[str, int]) -> list[tuple[str, Node]]:
    x, xs, pr = INT_X, LIST_X, PAIR_X
    if rule == "mod_double":
        return [("true", call("double", call("mod", x, params=(p["m"],)))),
                ("mod_only", call("mod", x, params=(p["m"],))),
                ("double_only", call("double", x))]
    if rule == "digital_root_square":
        return [("true", call("digital_root", call("square", x))),
                ("droot_only", call("digital_root", x)),
                ("sumdigits_square", call("sum_digits", call("square", x)))]
    if rule == "gt_mod":
        return [("true", call("gt", call("mod", x, params=(p["m"],)), params=(p["k"],))),
                ("gt_raw", call("gt", x, params=(p["k"],))),
                ("mod_only", call("mod", x, params=(p["m"],)))]
    if rule == "if_even":
        return [("true", call("if", call("even", x),
                              call("add1", call("double", x)), call("add1", x))),
                ("if_odd", call("if", call("odd", x),
                                call("add1", call("double", x)), call("add1", x))),
                ("double_only", call("double", x))]
    if rule == "neg_abs_mod":
        return [("true", call("neg", call("mod", call("abs", x), params=(p["m"],)))),
                ("mod_only", call("mod", x, params=(p["m"],))),
                ("neg_only", call("neg", x))]
    if rule == "band_shift":
        return [("true", call("shl", call("band", x, params=(p["k"],)), params=(p["s"],))),
                ("band_only", call("band", x, params=(p["k"],))),
                ("bor_only", call("bor", x, params=(p["k"],)))]
    if rule == "max_map_add":
        return [("true", call("max_list", call("map_add", xs, params=(p["k"],)))),
                ("max_only", call("max_list", xs)),
                ("sum_map_add", call("sum_list", call("map_add", xs, params=(p["k"],))))]
    if rule == "reverse_take_sum":
        return [("true", call("sum_list", call("take", call("reverse", xs), params=(p["k"],)))),
                ("take_sum", call("sum_list", call("take", xs, params=(p["k"],)))),
                ("sum_only", call("sum_list", xs))]
    if rule == "count_even_unique":
        return [("true", call("count_even", call("unique", xs))),
                ("count_even_all", call("count_even", xs)),
                ("unique_len", call("len_list", call("unique", xs)))]
    if rule == "sum_squares_evens":
        return [("true", call("sum_squares", call("evens", xs))),
                ("sum_squares_all", call("sum_squares", xs)),
                ("sum_evens", call("sum_list", call("evens", xs)))]
    if rule == "filter_gt_max":
        return [("true", call("max_list", call("filter_gt", xs, params=(p["k"],)))),
                ("max_only", call("max_list", xs)),
                ("count_gt", call("count_gt", xs, params=(p["k"],)))]
    if rule == "diff_scaled":
        return [("true", call("mul", call("diff_pair", pr), params=(p["k"],))),
                ("diff_only", call("diff_pair", pr)),
                ("sum_pair", call("sum_pair", pr))]
    if rule == "abs_double_mod":
        return [("true", call("mod", call("abs", call("double", x)), params=(p["m"],))),
                ("mod_only", call("mod", x, params=(p["m"],))),
                ("abs_only", call("abs", x))]
    if rule == "sumdigits_add":
        return [("true", call("add", call("sum_digits", x), params=(p["k"],))),
                ("sumdigits_only", call("sum_digits", x)),
                ("add_only", call("add", x, params=(p["k"],)))]
    if rule == "sub_square":
        return [("true", call("square", call("sub", x, params=(p["k"],)))),
                ("square_only", call("square", x)),
                ("sub_only", call("sub", x, params=(p["k"],)))]
    if rule == "odd_and_gt":
        return [("true", call("and", call("odd", x), call("gt", x, params=(p["k"],)))),
                ("odd_only", call("odd", x)),
                ("gt_only", call("gt", x, params=(p["k"],)))]
    raise ValueError(rule)


ALL_RULES = tuple(LABELS)


def params_for(rule: str, rng: random.Random) -> dict[str, int]:
    if rule in ("mod_double", "neg_abs_mod", "abs_double_mod"):
        return {"m": rng.randint(5, 11)}
    if rule == "sumdigits_add":
        return {"k": rng.randint(1, 5)}
    if rule == "sub_square":
        return {"k": rng.randint(1, 6)}
    if rule == "odd_and_gt":
        return {"k": rng.randint(0, 8)}
    if rule == "gt_mod":
        return {"m": rng.randint(5, 11), "k": rng.randint(1, 6)}
    if rule == "band_shift":
        return {"k": rng.randint(3, 15), "s": rng.randint(1, 4)}
    if rule in ("max_map_add",):
        return {"k": rng.randint(1, 5)}
    if rule in ("reverse_take_sum",):
        return {"k": rng.randint(2, 3)}
    if rule == "filter_gt_max":
        return {"k": rng.randint(1, 6)}
    if rule == "diff_scaled":
        return {"k": rng.randint(2, 5)}
    return {}


def domains(rule: str, shift: str = "none") -> tuple[dict[str, Any], dict[str, Any]]:
    t = INPUT_TYPE[rule]
    if t == "int":
        train = {"lo": 0, "hi": 9}
        return train, {"lo": 40, "hi": 120} if shift == "magnitude" else \
            {"lo": -80, "hi": -20} if shift == "sign" else {"lo": 10, "hi": 60}
    if t == "pair":
        train = {"lo": 0, "hi": 9}
        return train, {"lo": 40, "hi": 120} if shift == "magnitude" else \
            {"lo": 10, "hi": 60}
    train = {"len": 2, "len_hi": 5, "lo": 0, "hi": 9}
    if shift == "length":
        return train, {"len": 8, "len_hi": 14, "lo": 0, "hi": 9}
    if shift == "both":
        return train, {"len": 8, "len_hi": 14, "lo": 30, "hi": 90}
    return train, {"len": 3, "len_hi": 7, "lo": 0, "hi": 15}


def _sample(rule, rng, n, d, exclude=()):
    t = INPUT_TYPE[rule]
    if t == "int":
        return _uniq_ints(rng, n, d["lo"], d["hi"], exclude)
    if t == "pair":
        return _uniq_pairs(rng, n, d["lo"], d["hi"], exclude)
    return _uniq_lists(rng, n, d["len"], d["len_hi"], d["lo"], d["hi"], exclude)


def build_task(rule: str, seed: int, train_d, val_d, test_d, *,
               k: int = 4, n_val: int = 4, n_test: int = 12,
               tag: str = "none") -> Task:
    if rule not in LABELS:
        raise ValueError(rule)
    rng = random.Random(seed * 1000003 + sum(ord(c) for c in rule))
    p = params_for(rule, rng)
    tx = _sample(rule, rng, k, train_d)
    vx = _sample(rule, rng, n_val, val_d, tx)
    ex = _sample(rule, rng, n_test, test_d, tx + vx)
    if not disjoint(tx, vx, ex):
        raise AssertionError("EXP-K generator split overlap")
    label = LABELS[rule]
    train = [Example(x, label(x, p)) for x in tx]
    val = [Example(x, label(x, p)) for x in vx]
    test = [Example(x, label(x, p)) for x in ex]
    programs = _programs(rule, p)
    true_prog = dict(programs)["true"]
    for ex_ in train + val + test:
        got = dsl.evaluate(true_prog, ex_.x)
        if got != ex_.y:
            raise AssertionError(f"EXP-K generator mismatch {rule} at {ex_.x!r}: {got} != {ex_.y!r}")
    cands = [Candidate(id=cid, program=prog, cost=dsl.complexity(prog))
             for cid, prog in programs]
    return Task(
        name=f"{rule}-{tag}-s{seed}", family=rule, train=train,
        validation_pool=val, test=test, input_type=INPUT_TYPE[rule],
        output_type=OUTPUT_TYPE.get(rule, "int"),
        meta={"rule": rule, "params": p, "shift": tag, "candidates": cands,
              "train_domain": dict(train_d), "test_domain": dict(test_d),
              "unseen_ops": tuple(dict(programs)["true"].op for _ in [0])},
    )


def make_task(rule: str, seed: int, shift: str = "none", *,
              k: int = 4, n_val: int = 4, n_test: int = 12) -> Task:
    tr, te = domains(rule, shift)
    return build_task(rule, seed, tr, tr, te, k=k, n_val=n_val,
                      n_test=n_test, tag=shift)


def candidates_for(task: Task) -> list[Candidate]:
    return [Candidate(id=c.id, program=c.program, cost=c.cost)
            for c in task.meta["candidates"]]


def seed_registry(seeds: tuple[int, ...]) -> dict[str, Any]:
    """Deterministic fingerprint of the complete seed/generator registry."""
    rows = []
    for seed in seeds:
        for rule in ALL_RULES:
            t = make_task(rule, seed)
            rows.append({
                "seed": seed, "rule": rule, "params": t.meta["params"],
                "train": [e.x for e in t.train],
                "val": [e.x for e in t.validation_pool],
                "test": [e.x for e in t.test],
            })
    return {"n": len(rows), "rows": rows}
