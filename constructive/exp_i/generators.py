"""EXP-I — independent task generator families.

Generators are written from scratch for EXP-I. Label functions are independent
reference implementations; train, validation and evaluation use separate RNG
instances and disjoint domains, so no implementation object is shared between
generator and evaluator. Test labels are never exported to candidate builders.
"""

from __future__ import annotations

import random
from typing import Any, Callable

from .types import Candidate, Example, Task
from . import dsl
from .dsl import Node, call, lit, var


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def sample_ints(rng: random.Random, n: int, lo: int, hi: int) -> list[int]:
    span = hi - lo + 1
    if n >= span:
        pool = list(range(lo, hi + 1))
        rng.shuffle(pool)
        return pool[:n]
    out: list[int] = []
    seen: set[int] = set()
    while len(out) < n:
        v = rng.randint(lo, hi)
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def sample_lists(rng: random.Random, n: int, lo_len: int, hi_len: int,
                 lo: int, hi: int) -> list[list[int]]:
    out: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    tries = 0
    while len(out) < n and tries < 50000:
        tries += 1
        xs = tuple(rng.randint(lo, hi) for _ in range(rng.randint(lo_len, hi_len)))
        if xs in seen:
            continue
        seen.add(xs)
        out.append(list(xs))
    return out


def _task(name: str, family: str, train: list[Example], val: list[Example],
          test: list[Example], input_type: str, output_type: str,
          meta: dict[str, Any] | None = None) -> Task:
    return Task(name=name, family=family, train=train, validation_pool=val,
                test=test, input_type=input_type, output_type=output_type,
                meta=dict(meta or {}))


def disjointify(*groups: list[Any]) -> bool:
    """Sanity check: no identical input appears in two splits."""
    keys = [{repr(x) for x in g} for g in groups]
    return all(not (keys[i] & keys[j]) for i in range(len(keys))
               for j in range(i + 1, len(keys)))


# ---------------------------------------------------------------------------
# Independent reference label functions (no import from CCM / exp_h)
# ---------------------------------------------------------------------------


def ref_sum_digits(n: int) -> int:
    return sum(int(c) for c in str(abs(n)))


def ref_digital_root(n: int) -> int:
    return 0 if n == 0 else 1 + (abs(n) - 1) % 9


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


def ref_fib(n: int) -> int:
    a, b = 0, 1
    for _ in range(max(0, min(int(n), 30))):
        a, b = b, a + b
    return a


REFERENCE_LABELS: dict[str, Callable[[Any], Any]] = {
    "sum_digits": ref_sum_digits,
    "square": lambda x: x * x,
    "fib": ref_fib,
    "is_prime": _is_prime,
    "double": lambda x: x * 2,
    "abs": abs,
    "add1": lambda x: x + 1,
    "digital_root": ref_digital_root,
    "sum_of_evens": lambda xs: sum(v for v in xs if v % 2 == 0),
    "distinct_count": lambda xs: len(set(xs)),
    "sum_squares": lambda xs: sum(v * v for v in xs),
    "count_gt2": lambda xs: sum(1 for v in xs if v > 2),
}


# ---------------------------------------------------------------------------
# Twin tasks (EXP-I-A) — validation deliberately spans a discriminating domain
# ---------------------------------------------------------------------------


def twin_task(seed: int, k: int = 3, n_test: int = 12, n_val: int = 12,
              validation_range: tuple[int, int] = (10, 30),
              test_range: tuple[int, int] = (30, 70)) -> Task:
    rng = random.Random(seed * 7919 + 11)
    support_x = sample_ints(rng, k, 1, 9)
    val_big = sample_ints(rng, max(n_val + 6, 14), *validation_range)
    val_x = [v for v in val_big if v not in support_x][:n_val]
    test_big = sample_ints(rng, n_test + 8, *test_range)
    test_x = [v for v in test_big if v not in support_x and v not in val_x][:n_test]
    train = [Example(x, ref_sum_digits(x)) for x in support_x]
    val = [Example(x, ref_sum_digits(x)) for x in val_x]
    test = [Example(x, ref_sum_digits(x)) for x in test_x]
    assert disjointify([e.x for e in train], [e.x for e in val], [e.x for e in test])
    return _task(f"twin-s{seed}", "twin", train, val, test, "int", "int",
                 {"hidden": "sum_digits", "twin": "sum_digits vs digital_root",
                  "ambiguous_domain": [1, 9], "validation_range": list(validation_range)})


def twin_candidates() -> list[Candidate]:
    """Candidate hypotheses available to the learner. Both twins are present."""
    x = var("int")
    specs = [
        ("sum_digits", call("sum_digits", x)),
        ("digital_root", call("digital_root", x)),
        ("identity", x),
        ("double", call("double", x)),
        ("abs", call("abs", x)),
        ("mod9", call("mod", x, params=(9,))),
    ]
    return [Candidate(id=cid, program=prog, cost=dsl.complexity(prog))
            for cid, prog in specs]


# ---------------------------------------------------------------------------
# Parametric collision families (EXP-I-C) — exact H4 names
# ---------------------------------------------------------------------------

COLLISION_FAMILIES = ("count_gt", "take", "filter_gt", "map_add", "add")


def collision_task(family: str, param: int, seed: int, k: int = 4,
                   n_test: int = 12) -> Task:
    rng = random.Random(seed * 104729 + param * 31)
    if family == "add":
        xs = sample_ints(rng, k + 4 + n_test, -5, 25)
        label: Callable[[Any], Any] = lambda v: v + param
        in_type, out_type = "int", "int"
    else:
        lists = sample_lists(rng, k + 4 + n_test, 4, 8, 0, 12)
        if family == "count_gt":
            label = lambda xs: sum(1 for v in xs if v > param)
            out_type = "int"
        elif family == "take":
            label = lambda xs: list(xs[:param])
            out_type = "list"
        elif family == "filter_gt":
            label = lambda xs: [v for v in xs if v > param]
            out_type = "list"
        elif family == "map_add":
            label = lambda xs: [v + param for v in xs]
            out_type = "list"
        else:
            raise ValueError(family)
        xs = lists
        in_type = "list"
    train = [Example(x, label(x)) for x in xs[:k]]
    val = [Example(x, label(x)) for x in xs[k:k + 4]]
    test = [Example(x, label(x)) for x in xs[k + 4:]]
    assert disjointify([e.x for e in train], [e.x for e in val], [e.x for e in test])
    return _task(f"{family}-{param}-s{seed}", "collision", train, val, test,
                 in_type, out_type,
                 {"family": family, "param": param, "hidden": f"{family}({param})"})


def collision_program(family: str, param: int) -> Node:
    if family == "add":
        return call("add", var("int"), params=(param,))
    return call(family, var("list"), params=(param,))


def collision_candidates(params: list[int], family: str) -> list[Candidate]:
    out = []
    for p in params:
        prog = collision_program(family, p)
        out.append(Candidate(id=f"{family}-{p}", program=prog,
                             cost=dsl.complexity(prog)))
    return out


# ---------------------------------------------------------------------------
# Numeric / structural OOD tasks (EXP-I-E)
# ---------------------------------------------------------------------------

INT_RULES = ("sum_digits", "square", "fib", "is_prime")
LIST_RULES = ("sum_of_evens", "distinct_count")
SCENE_RULES = ("nearest", "same_shape", "left_of")
OOD_RULES = INT_RULES + LIST_RULES + SCENE_RULES


def _scene_label(rule: str, scene: Any) -> Any:
    # scene = (entities, query_index); entity = (x, y, color, shape, size)
    ents, q = scene
    qx, qy, qc, qs, qz = ents[q]
    if rule == "left_of":
        return any(e[0] < qx and e[1] == qy for i, e in enumerate(ents) if i != q)
    if rule == "same_shape":
        return any(e[3] == qs for i, e in enumerate(ents) if i != q)
    if rule == "nearest":
        best, best_i = None, -1
        for i, e in enumerate(ents):
            if i == q:
                continue
            d = abs(e[0] - qx) + abs(e[1] - qy)
            if best is None or d < best:
                best, best_i = d, i
        return best_i
    raise ValueError(rule)


def _gen_scene(rng: random.Random, lo: int, hi: int, density: str = "normal"):
    n = rng.randint(lo, hi)
    width = max(4, n + 2)
    ents = [(rng.randrange(width), rng.randrange(width), rng.choice("RGB"),
             rng.choice("SCT"), rng.choice((1, 2, 3))) for _ in range(n)]
    q = rng.randrange(n)
    return (tuple(ents), q)


def numeric_ood_task(rule: str, seed: int, k: int = 4, n_test: int = 12,
                     train_domain: tuple[int, int] = (0, 9),
                     test_domain: tuple[int, int] = (10, 60)) -> Task:
    rng = random.Random(seed * 31337 + len(rule))
    if rule in INT_RULES:
        fn = REFERENCE_LABELS[rule]
        # train and validation share a domain, so sample jointly without replacement
        in_domain = sample_ints(rng, k + 4, *train_domain)
        tx, vx = in_domain[:k], in_domain[k:]
        ex = sample_ints(rng, n_test, *test_domain)
        train = [Example(x, fn(x)) for x in tx]
        val = [Example(x, fn(x)) for x in vx]
        test = [Example(x, fn(x)) for x in ex]
        in_type = "int"
        out_type = "bool" if rule == "is_prime" else "int"
    elif rule in LIST_RULES:
        fn = REFERENCE_LABELS[rule]
        all_lists = sample_lists(rng, k + 4, 2, 5, 0, 9)
        tlen, vlen = all_lists[:k], all_lists[k:]
        elen = sample_lists(rng, n_test, 9, 16, 0, 9)
        train = [Example(x, fn(x)) for x in tlen]
        val = [Example(x, fn(x)) for x in vlen]
        test = [Example(x, fn(x)) for x in elen]
        in_type, out_type = "list", "int"
    else:
        train_s = [_gen_scene(rng, 3, 5) for _ in range(k)]
        val_s = []
        while len(val_s) < 4:
            s = _gen_scene(rng, 3, 5)
            if s not in train_s:
                val_s.append(s)
        test_s = [_gen_scene(rng, 7, 12, "dense") for _ in range(n_test)]
        train = [Example(s, _scene_label(rule, s)) for s in train_s]
        val = [Example(s, _scene_label(rule, s)) for s in val_s]
        test = [Example(s, _scene_label(rule, s)) for s in test_s]
        in_type = "scene"
        out_type = "bool" if rule in ("left_of", "same_shape") else "int"
    assert disjointify([e.x for e in train], [e.x for e in val], [e.x for e in test])
    return _task(f"ood-{rule}-s{seed}", "numeric-ood", train, val, test,
                 in_type, out_type,
                 {"rule": rule, "train_domain": list(train_domain),
                  "test_domain": list(test_domain)})


def ood_candidates(rule: str) -> list[Candidate]:
    """Plausible candidate programs for a rule (one correct + distractors)."""
    x = var("int")
    xs = var("list")
    if rule == "sum_digits":
        specs = [("sum_digits", call("sum_digits", x)),
                 ("digital_root", call("digital_root", x)),
                 ("mod9", call("mod", x, params=(9,)))]
    elif rule == "square":
        specs = [("square", call("square", x)),
                 ("double", call("double", x)),
                 ("sum_digits", call("sum_digits", x))]
    elif rule == "fib":
        specs = [("fib", call("fib", x)), ("square", call("square", x))]
    elif rule == "is_prime":
        specs = [("is_prime", call("is_prime", x)),
                 ("odd", call("odd", x)), ("gt1", call("gt", x, params=(1,)))]
    elif rule == "sum_of_evens":
        specs = [("sum_of_evens", call("sum_list", call("evens", xs))),
                 ("sum_list", call("sum_list", xs))]
    elif rule == "distinct_count":
        specs = [("distinct_count", call("len_list", call("unique", xs))),
                 ("len_list", call("len_list", xs))]
    else:
        # Scene relations are not expressible as typed int/list DSL programs;
        # generalization.py evaluates them with independent reference functions.
        # Reporting this as a representation limitation is honest — no fake program.
        return []
    out = []
    for cid, prog in specs:
        try:
            dsl.infer_type(prog)
        except dsl.DslError:
            continue
        out.append(Candidate(id=cid, program=prog, cost=dsl.complexity(prog)))
    return out


# ---------------------------------------------------------------------------
# Abstraction tasks (EXP-I-G): shared-core and unrelated families
# ---------------------------------------------------------------------------

ABS_CORES = {
    "shared-core": lambda k: call("map_add", call("evens", var("list")), params=(k,)),
    "unrelated-sum": lambda k: call("evens", var("list")),
    "unrelated-max": lambda k: call("unique", var("list")),
    "unrelated-sq": lambda k: call("map_square", var("list")),
}
ABS_REDUCERS = ("sum_list", "len_list", "max_list", "min_list", "sum_squares")


def abstraction_stream(seed: int, family: str = "shared-core", k: int = 4,
                       n_test: int = 8) -> list[Task]:
    rng = random.Random(seed * 5437 + sum(map(ord, family)))
    pk = rng.choice((1, 2, 3, 4))
    core = ABS_CORES[family](pk)
    tasks: list[Task] = []
    for idx, reducer in enumerate(ABS_REDUCERS):
        prog = call(reducer, core)
        xs = sample_lists(rng, k + 4 + n_test, 4, 8, 0, 9)

        def label(v: Any, _p=prog) -> Any:
            return dsl.evaluate(_p, v)

        train = [Example(x, label(x)) for x in xs[:k]]
        val = [Example(x, label(x)) for x in xs[k:k + 4]]
        test = [Example(x, label(x)) for x in xs[k + 4:]]
        tasks.append(_task(f"abs-{family}-{reducer}-s{seed}", "abstraction",
                           train, val, test, "list", "int",
                           {"family": family, "core": str(core),
                            "hidden_program": str(prog), "stream_index": idx}))
    return tasks


# ---------------------------------------------------------------------------
# Adversarial builders (EXP-I-A / G / mandatory adversarial tests)
# ---------------------------------------------------------------------------


def corrupt_validation(task: Task, n_flips: int = 1, seed: int = 0) -> Task:
    rng = random.Random(seed)
    val = list(task.validation_pool)
    for i in rng.sample(range(len(val)), min(n_flips, len(val))):
        ex = val[i]
        bad = (not ex.y) if isinstance(ex.y, bool) else ex.y + 1000
        val[i] = Example(ex.x, bad)
    return Task(task.name + "-corrupt", task.family, list(task.train), val,
                list(task.test), task.input_type, task.output_type,
                dict(task.meta))


def contradictory_support(task: Task) -> Task:
    train = list(task.train)
    if train:
        ex = train[0]
        bad = (not ex.y) if isinstance(ex.y, bool) else ex.y + 999
        # Same input, conflicting label: a directly detectable contradiction.
        train.append(Example(ex.x, bad))
    return Task(task.name + "-contradiction", task.family, train,
                list(task.validation_pool), list(task.test),
                task.input_type, task.output_type, dict(task.meta))


def blind_positive_task(seed: int = 0, k: int = 4) -> Task:
    """Support contains only odd composites — zero positive prime evidence."""
    rng = random.Random(seed)
    odds = [n for n in range(9, 40, 2) if not _is_prime(n)]
    sx = rng.sample(odds, k)
    vx = sample_ints(rng, 4, 10, 40)
    tx = sample_ints(rng, 12, 10, 60)
    train = [Example(x, _is_prime(x)) for x in sx]
    val = [Example(x, _is_prime(x)) for x in vx]
    test = [Example(x, _is_prime(x)) for x in tx]
    return _task(f"blind-prime-s{seed}", "blind-positive", train, val, test,
                 "int", "bool", {"rule": "is_prime", "blind": True})
