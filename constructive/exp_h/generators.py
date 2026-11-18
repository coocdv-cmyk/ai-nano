"""Task generators for EXP-H.

Every generator produces an independent ``Task``-like object (``constructed
from the hidden program, never from the evaluator``).  Deterministic under
``random.Random`` seeds.  The learner sees only examples; labels come from a
hidden program in the richer ``exp_h.engine`` language.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from ..core import COLORS, SHAPES, SIZES, Entity, Scene
from ..worlds import Task
from .engine import HNode, eval_hidden


@dataclass
class ExpHTask(Task):
    """Constructed by a hidden program; carries ground-truth for taxonomies."""

    hidden: Any = None
    family_tag: str = ""

    def meta_dict(self) -> dict[str, Any]:
        out = dict(self.meta)
        out["family_tag"] = self.family_tag
        return out


def _x() -> HNode:
    return HNode("var")


# ---------------------------------------------------------------------------
# Hidden vocabulary family (CCM-expressible, unseen instance)
# ---------------------------------------------------------------------------

#: Templates: hidden program tuples with a hole ``h`` anywhere in them.
#: Each entry is (hidden AST builder, target input type, output type, CCM cost)
_VOCAB_INT = [
    (lambda r: HNode("double", None, (HNode("abs", None, (_x(),)),)), 1, "int"),
    (lambda r: HNode("square", None, (HNode("sub1", None, (_x(),)),)), 1, "int"),
    (lambda r: HNode("abs", None, (HNode("double", None, (_x(),)),)), 1, "int"),
    (lambda r: HNode("mod", r.choice((3, 4, 5, 7)), (HNode("square", None, (_x(),)),)), 2, "int"),
    (lambda r: HNode("sub1", None, (HNode("double", None, (_x(),)),)), 1, "int"),
    (lambda r: HNode("add1", None, (HNode("double", None, (_x(),)),)), 1, "int"),
    (lambda r: HNode("neg", None, (HNode("abs", None, (_x(),)),)), 2, "int"),
    (lambda r: HNode("double", None, (HNode("double", None, (_x(),)),)), 2, "int"),
    (lambda r: HNode("square", None, (HNode("double", None, (_x(),)),)), 2, "int"),
    (lambda r: HNode("sum_digits", None, (HNode("double", None, (_x(),)),)), 1, "int"),
    (lambda r: HNode("digital_root", None, (HNode("double", None, (_x(),)),)), 2, "int"),
    (lambda r: HNode("abs", None, (HNode("neg", None, (HNode("double", None, (_x(),)),)),)), 3, "int"),
    (lambda r: HNode("double", None, (HNode("abs", None, (HNode("neg", None, (_x(),)),)),)), 3, "int"),
]

_VOCAB_LIST = [
    (lambda r: HNode("sum_list", None, (HNode("evens", None, (_x(),)),)), 2, "int"),
    (lambda r: HNode("len_list", None, (HNode("odds", None, (_x(),)),)), 2, "int"),
    (lambda r: HNode("max_list", None, (HNode("evens", None, (_x(),)),)), 2, "int"),
    (lambda r: HNode("sum_list", None, (HNode("map_add", k := r.choice((1, 2, 3)),
                                               (HNode("evens", None, (_x(),)),)),)), 3, "int"),
    (lambda r: HNode("count_even", None, (HNode("sort_desc", None, (_x(),)),)), 2, "int"),
    (lambda r: HNode("len_list", None, (HNode("unique", None, (_x(),)),)), 2, "int"),
    (lambda r: HNode("sum_squares", None, (HNode("odds", None, (_x(),)),)), 2, "int"),
    (lambda r: HNode("head", None, (HNode("sort_desc", None, (_x(),)),)), 1, "int"),
    (lambda r: HNode("last", None, (HNode("sort", None, (_x(),)),)), 1, "int"),
    (lambda r: HNode("range_list", None, (HNode("evens", None, (_x(),)),)), 2, "int"),
    (lambda r: HNode("sum_list", None, (HNode("filter_gt", k := r.choice((1, 2)),
                                               (_x(),)),)), 2, "int"),
]


def gen_vocab_target(rng: random.Random, input_type: str) -> tuple[HNode, str]:
    if input_type == "list":
        builder, _, _ = _VOCAB_LIST[rng.randrange(len(_VOCAB_LIST))]
    else:
        builder, _, _ = _VOCAB_INT[rng.randrange(len(_VOCAB_INT))]
    return builder(rng), "vocab"


# ---------------------------------------------------------------------------
# Vocabulary-exclusion family (bitwise / cube / conditional): no CCM equivalent
# ---------------------------------------------------------------------------


def gen_exclusion_target(rng: random.Random, input_type: str) -> tuple[HNode, str]:
    """Targets with no CCM-equivalent: bitwise ops, or sums of cubes.

    Chosen deliberately so no CCM composition can express the semantics
    (CCM has no cube, no bitwise op, and no conditional).
    """
    if input_type == "list":
        choice = rng.random()
        if choice < 0.34:
            return HNode("sum_list", None, (HNode("map_cube", None, (_x(),)),)), "exclusion"
        if choice < 0.67:
            return HNode("sum_list", None, (HNode("map_cube", None, (HNode("evens", None, (_x(),)),)),)), "exclusion"
        return HNode("max_list", None, (HNode("map_cube", None, (_x(),)),)), "exclusion"
    op = rng.choice(("band", "bor", "bxor", "shl", "shr"))
    k = rng.choice((1, 2, 3, 5, 7, 9, 13))
    return HNode(op, k, (_x(),)), "exclusion"


# ---------------------------------------------------------------------------
# Conditional family (CCM enumerates no conditionals -> representation limit)
# ---------------------------------------------------------------------------


def gen_conditional_target(rng: random.Random, input_type: str) -> tuple[HNode, str]:
    branch_a = rng.choice((HNode("double", None, (_x(),)),
                           HNode("square", None, (_x(),)),
                           HNode("neg", None, (_x(),))))
    branch_b = rng.choice((HNode("add1", None, (_x(),)),
                           HNode("sub1", None, (_x(),)),
                           HNode("abs", None, (_x(),))))
    cond_k = rng.choice((0, 3, 5, 7))
    cond = rng.choice((HNode("gt", cond_k, (_x(),)),
                       HNode("lt", cond_k, (_x(),)),
                       HNode("even", None, (_x(),)),
                       HNode("odd", None, (_x(),))))
    return HNode("if_else", None, (cond, branch_a, branch_b)), "conditional"


# ---------------------------------------------------------------------------
# Abstraction family: tasks share a hidden core (reuse real, not arbitrary)
# ---------------------------------------------------------------------------


def gen_abstraction_stream(seed: int) -> list[tuple[HNode, str, str]]:
    """Task stream sharing the hidden core ``evens(map_add(c)(x))``.

    Returns a list of (hidden_program, family_tag, name).  The learner never
    sees the core; it is only behaviourally available through examples.
    """
    rng = random.Random(seed)
    k = rng.choice((1, 2, 3, 4))
    core = HNode("map_add", k, (HNode("evens", None, (_x(),)),))
    specs = [
        (HNode("sum_list", None, (core,)), "abs-sum", "sum_list(map_add(k)(evens(x)))"),
        (HNode("len_list", None, (core,)), "abs-len", "len_list(map_add(k)(evens(x)))"),
        (HNode("max_list", None, (core,)), "abs-max", "max_list(map_add(k)(evens(x)))"),
        (HNode("min_list", None, (core,)), "abs-min", "min_list(map_add(k)(evens(x)))"),
        (HNode("sum_squares", None, (core,)), "abs-sq", "sum_squares(map_add(k)(evens(x)))"),
        (HNode("sum_list", None, (HNode("unique", None, (core,)),)), "abs-uniq-sum",
         "sum_list(unique(map_add(k)(evens(x))))"),
    ]
    return [(node, "abstraction", name) for node, fam, name in specs]


# ---------------------------------------------------------------------------
# Non-identifiable twins: supports engineered so two semantically distinct
# rules coincide
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Domain builders
# ---------------------------------------------------------------------------


def sample_ints(rng: random.Random, n: int, lo: int, hi: int) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    span = hi - lo + 1
    if n >= span:
        pool = list(range(lo, hi + 1))
        rng.shuffle(pool)
        return pool[:n]
    while len(out) < n:
        v = rng.randint(lo, hi)
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def sample_lists(rng: random.Random, n: int, lo_len: int, hi_len: int,
                 lo: int, hi: int) -> list[list[int]]:
    out: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    while len(out) < n:
        ln = rng.randint(lo_len, hi_len)
        xs = tuple(rng.randint(lo, hi) for _ in range(ln))
        if xs in seen:
            continue
        seen.add(xs)
        out.append(list(xs))
    return out


def build_task(name: str, rule: str, train, val, test, input_type: str,
               output_type: str, probes, family: str, hidden: Any,
               meta: dict | None = None) -> ExpHTask:
    task_meta = dict(meta or {})
    task_meta["family_tag"] = family
    return ExpHTask("exp_h", name, rule, train, val, test, input_type,
                    output_type, list(probes or []), task_meta, hidden, family)


# ---------------------------------------------------------------------------
# H1 / H7 task builders
# ---------------------------------------------------------------------------


def build_h1_task(seed: int, k: int, family: str, n_test: int = 20,
                  n_probe: int = 4) -> ExpHTask:
    """Build one open-world discovery task from a hidden program."""
    rng = random.Random(seed)

    last_error = None
    node = fam = out_type = None
    input_type = None
    inputs = None
    for _attempt in range(12):
        if family == "vocab":
            input_type = "list" if rng.random() < 0.55 else "int"
            node, fam = gen_vocab_target(rng, input_type)
        elif family == "exclusion":
            if rng.random() < 0.5:
                input_type = "int"
            else:
                input_type = "list"
            node, fam = gen_exclusion_target(rng, input_type)
        elif family == "conditional":
            node, fam = gen_conditional_target(rng, "int")
            input_type = "int"
        else:
            raise ValueError(f"unknown H1 family {family!r}")

        out_type = _infer_out_type(node)

        if input_type == "int":
            attempt_inputs = sample_ints(rng, k + 1 + n_test, -15, 30)
        else:
            attempt_inputs = sample_lists(rng, k + 1 + n_test, 3, 6, 0, 9)

        def safe_label(v):
            try:
                return eval_hidden(node, v)
            except Exception:
                return None

        labelled = [(x, safe_label(x)) for x in attempt_inputs]
        if any(y is None for _, y in labelled):
            last_error = ValueError(
                f"hidden program failed on generated input (seed={seed}, family={family})")
            continue
        if any(isinstance(y, bool) != (out_type == "bool") for _, y in labelled):
            last_error = ValueError("output type mismatch")
            continue
        # non-degenerate support: at least two distinct outputs required
        train_labels = [y for _, y in labelled[:k]]
        if len(set(train_labels)) < 2 and k >= 2:
            last_error = ValueError(
                f"degenerate support labels (seed={seed}, family={family})")
            continue
        inputs = attempt_inputs
        break
    else:
        raise last_error or ValueError(f"could not generate {family} task")
    train = labelled[:k]
    val = labelled[k:k + 1]
    test = labelled[k + 1:]
    if input_type == "int":
        probes = sample_ints(rng, n_probe, -15, 30)
    else:
        probes = sample_lists(rng, n_probe, 3, 6, 0, 9)
    rule = str(node)
    name = f"h1-{family}-{input_type}-s{seed}"
    return build_task(name, rule, train, val, test, input_type, out_type, probes,
                      family, node, {"hidden_program": rule, "k": k})


def _infer_out_type(node: HNode) -> str:
    op = node.op
    if op in ("even", "odd", "is_prime", "not", "and", "or", "gt", "lt"):
        return "bool"
    if op in ("sum_list", "max_list", "min_list", "len_list", "head", "last",
              "range_list", "count_even", "count_odd", "sum_squares", "count_gt"):
        return "int"
    if op in ("if_else",):
        return _infer_out_type(node.children[1])
    for lst in (("reverse", "sort", "sort_desc", "unique", "evens", "odds",
                 "take", "drop", "filter_gt", "map_add", "map_mul", "map_square", "map_cube"),):
        if op in lst:
            return "list"
    return "int"


def build_twin_task(seed: int, k: int, n_test: int = 20,
                    n_probe: int = 4) -> ExpHTask:
    """Support built on single-digit ints; twins coincide everywhere on it."""
    rng = random.Random(seed)
    inputs = sample_ints(rng, k + 1 + n_test, 0, 9)
    node_a = HNode("sum_digits", None, (_x(),))
    node_b = HNode("digital_root", None, (_x(),))
    train = [(x, eval_hidden(node_a, x)) for x in inputs[:k]]
    val = [(x, eval_hidden(node_a, x)) for x in inputs[k:k + 1]]
    # test extends past the ambiguous domain on purpose (structural OOD)
    test = [(x, eval_hidden(node_a, x)) for x in inputs[k + 1:]] + [
        (x, eval_hidden(node_a, x)) for x in range(10, 30)][: n_test - (len(inputs) - k - 1)]
    test = test[:n_test]
    probes = sample_ints(rng, n_probe, 0, 9)
    rule = "sum_digits(x)"
    name = f"h1-twin-s{seed}"
    return build_task(name, rule, train, val, test, "int", "int", probes,
                      "nonidentifiable", node_a,
                      {"hidden_program": rule, "twin": "sum_digits vs digital_root",
                       "k": k})


def build_list_twin_task(seed: int, k: int, n_test: int = 20) -> ExpHTask:
    """Constant lists: max/min/head/last coincide on support, differ on test."""
    rng = random.Random(seed)
    consts = [sample_ints(rng, 1, 3, 9)[0] for _ in range(k + 1 + n_test)]
    node = HNode("max_list", None, (_x(),))
    train_lists = [[v] * 4 for v in consts[:k]]
    val_list = [[consts[k]] * 4]
    train = [(x, eval_hidden(node, x)) for x in train_lists]
    val = [(x, eval_hidden(node, x)) for x in val_list]
    # test blends constant lists and "one element bigger by 1" lists,
    # which distinguish the twins
    test_lists: list[list[int]] = []
    for i, v in enumerate(consts[k + 1:]):
        test_lists.append([v] * 4 if i % 2 == 0 else [v, v, v, v + 1])
    test = [(x, eval_hidden(node, x)) for x in test_lists[:n_test]]
    probes = [list(x) for x in train_lists[:4]]
    rule = "max_list(x)"
    name = f"h1-ltwin-s{seed}"
    return build_task(name, rule, train, val, test, "list", "int", probes,
                      "nonidentifiable", node,
                      {"hidden_program": rule, "twin": "max/min/head/last",
                       "k": k})


# ---------------------------------------------------------------------------
# H3: OOD distributions (list sizes; relation density; numerical range)
# ---------------------------------------------------------------------------


def gen_scene_counts(rng: random.Random, n: int, grid: int, width: int, mirror: bool = False):
    ents = []
    cells = [(x, y) for x in range(width) for y in range(grid)]
    rng.shuffle(cells)
    for i, (x, y) in enumerate(cells[:n]):
        if mirror:
            x = width - 1 - x
        ents.append(Entity(f"e{i}", rng.choice(COLORS), rng.choice(SHAPES),
                           rng.choice(SIZES), x, y))
    return Scene(tuple(ents), rng.choice([e.id for e in ents]), mirror)


def build_ood_scene_task(rule: str, seed: int, k: int, train_size: tuple[int, int],
                         test_size: tuple[int, int], width: int = 5,
                         n_test: int = 20) -> ExpHTask:
    """Train on train_size entities, test on test_size (structural OOD)."""
    from ..core import build_prims
    prims = build_prims()
    rng = random.Random(seed)

    def gen(lo, hi):
        for _ in range(4000):
            n = rng.randint(lo, hi)
            s = gen_scene_counts(rng, n, width, width)
            try:
                if rule in ("biggest", "smallest"):
                    y = prims[rule].fn(s)
                else:
                    y = prims[rule].fn(s, s.by_id(s.query))
                return s, y
            except Exception:
                continue
        raise RuntimeError(f"cannot build scene for {rule}")

    train = [gen(*train_size) for _ in range(k)]
    val = [gen(*train_size)]
    test = [gen(*test_size) for _ in range(n_test)]
    probes = [gen(*train_size)[0] for _ in range(4)]
    name = f"h3-{rule}-ood-s{seed}"
    return build_task(name, rule, train, val, test, "scene", "entity", probes,
                      "structure-ood", None,
                      {"train_entities": train_size, "test_entities": test_size})


def build_ood_numeric_task(rule: str, seed: int, k: int,
                           lo: int, hi: int, test_hi: int,
                           n_test: int = 20) -> ExpHTask:
    from ..core import build_prims
    prims = build_prims()
    rng = random.Random(seed)
    train_pool = list(range(lo, hi + 1))
    rng.shuffle(train_pool)
    if len(train_pool) < k + 1:
        raise ValueError("train range too small for support+validation")
    train_vals = train_pool[:k + 1]
    test_vals = list(range(hi + 1, test_hi + 1))
    rng.shuffle(test_vals)
    test_vals = test_vals[:n_test]
    if len(test_vals) < n_test:
        raise ValueError("test range too small")
    train = [(x, prims[rule].fn(x)) for x in train_vals[:k]]
    val = [(x, prims[rule].fn(x)) for x in train_vals[k:k + 1]]
    test = [(x, prims[rule].fn(x)) for x in test_vals]
    probes = [rng.randint(lo, hi) for _ in range(4)]
    name = f"h3-{rule}-numeric-s{seed}"
    return build_task(name, rule, train, val, test, "int",
                      "bool" if rule in ("is_prime", "even", "odd") else "int", probes,
                      "numeric-ood", None,
                      {"train_range": [lo, hi], "test_range": [hi + 1, test_hi]})


def build_ood_list_task(rule: str, seed: int, k: int,
                        train_len: tuple[int, int], test_len: tuple[int, int],
                        n_test: int = 20, n_probe: int = 4) -> ExpHTask:
    """Train on short lists, test on long lists (lengths up to 100)."""
    from constructive.core import build_prims, eval_term, Term, V, same_value
    from constructive.worlds import DERIVED_LIST_TERMS
    prims = build_prims()
    rng = random.Random(seed)

    def label(xs):
        if rule in prims:
            return prims[rule].fn(xs)
        return eval_term(DERIVED_LIST_TERMS[rule], {"x": xs}, prims)

    def gen(lo_len, hi_len):
        out = []
        seen = set()
        tries = 0
        while len(out) < n_test + 1 and tries < 20000:
            tries += 1
            ln = rng.randint(lo_len, hi_len)
            xs = [rng.randint(0, 9) for _ in range(ln)]
            t = tuple(xs)
            if t in seen:
                continue
            seen.add(t)
            out.append(xs)
        return out

    train_xs = gen(*train_len)[:k]
    val_xs = gen(*train_len)[:1]
    test_xs = []
    while len(test_xs) < n_test:
        ln = rng.randint(*test_len)
        test_xs.append([rng.randint(0, 9) for _ in range(ln)])
    train = [(x, label(x)) for x in train_xs]
    val = [(x, label(x)) for x in val_xs]
    test = [(x, label(x)) for x in test_xs]
    probes = gen(*train_len)[:n_probe]
    name = f"h3-{rule}-len-s{seed}"
    return build_task(name, rule, train, val, test, "list",
                      "list" if rule in ("unique", "sort", "sort_desc", "reverse") else "int",
                      probes, "length-ood", None,
                      {"train_len": list(train_len), "test_len": list(test_len)})


# ---------------------------------------------------------------------------
# H2: deep compositions
# ---------------------------------------------------------------------------


def _unary_chain(rng: random.Random, length: int) -> HNode:
    """Random walk of unary ops; no immediate inverse/identity pairs."""
    ops = ["double", "add1", "sub1", "abs", "neg", "mod", "sum_digits"]
    for _ in range(400):
        ops_used: list[Any] = []
        node = _x()
        ok = True
        for _ in range(length):
            pool = ["double", "add1", "sub1", "abs", "neg", "mod(3)", "mod(7)", "sum_digits", "digital_root"]
            if length > 6:
                pool.remove("square") if "square" in pool else None
            prev = ops_used[-1] if ops_used else None
            choices = [o for o in pool if o != prev]
            # avoid add1/sub1 ping-pong
            if prev in ("add1",):
                choices = [o for o in choices if o != "sub1"]
            if prev in ("sub1",):
                choices = [o for o in choices if o != "add1"]
            op = rng.choice(choices)
            if isinstance(op, str) and op.startswith("mod("):
                k = int(op[4:-1])
                node = HNode("mod", k, (node,))
                ops_used.append(f"mod({k})")
            else:
                node = HNode(op, None, (node,))
                ops_used.append(op)
        return node
    raise RuntimeError("chain generation failed")


def build_h2_task(seed: int, depth: int, family: str, k: int = 4,
                  n_test: int = 12, n_probe: int = 4) -> ExpHTask:
    rng = random.Random(seed * 991 + depth)
    if family == "unary_chain":
        node = _unary_chain(rng, depth)
        inputs = sample_ints(rng, k + 1 + n_test, 0, 20)
    elif family == "conditional_tree":
        # nested if_else tree with a discriminating condition; unreachable by CCM
        op_choices = ["double", "neg", "add1", "sub1"]
        def build(d: int) -> HNode:
            if d <= 1:
                return HNode(rng.choice(op_choices), None, (_x(),))
            cond = HNode("gt", rng.choice((0, 2, 3, 5)), (_x(),))
            return HNode("if_else", None,
                         (cond, build(d - 1), build(d - 1)))
        node = build(depth)
        inputs = sample_ints(rng, k + 1 + n_test, -10, 20)
    elif family == "pair_tree":
        def build(d: int, want: str) -> HNode:
            if want == "pair":
                if d <= 0:
                    return HNode("var")
                return HNode("mkpair", None, (build(d - 1, "int"),
                                               build(d - 1, "int")))
            if d <= 0:
                return HNode(rng.choice(("first", "second")), None,
                             (HNode("var"),))
            op = rng.choice(("sum_pair", "diff_pair", "max_pair", "min_pair",
                             "gcd_pair"))
            return HNode(op, None, (build(d - 1, "pair"), build(d - 1, "pair")))
        node = build(depth, "int")
        pairs = [(rng.randint(0, 12), rng.randint(0, 12)) for _ in range(k + 1 + n_test)]
        labels = [eval_hidden(node, p) for p in pairs]
        train = [(p, y) for p, y in list(zip(pairs, labels))[:k]]
        val = [(pairs[k], labels[k])]
        test = list(zip(pairs[k + 1:], labels[k + 1:]))
        probes = pairs[:n_probe]
        rule = str(node)
        name = f"h2-pair_tree-d{depth}-s{seed}"
        return build_task(name, rule, train, val, test, "pair", "int", probes,
                          "h2-pair_tree", node,
                          {"depth": depth, "hidden_program": rule})
    else:
        raise ValueError(f"unknown H2 family {family!r}")

    labels = [eval_hidden(node, x) for x in inputs]
    train = [(x, y) for x, y in zip(inputs[:k], labels[:k])]
    val = [(inputs[k], labels[k])]
    test = list(zip(inputs[k + 1:], labels[k + 1:]))
    probes = sample_ints(rng, n_probe, 0, 20)
    rule = str(node)
    name = f"h2-{family}-d{depth}-s{seed}"
    return build_task(name, rule, train, val, test, "int", "int", probes,
                      "h2-" + family, node,
                      {"depth": depth, "hidden_program": rule})


# ---------------------------------------------------------------------------
# H1/H6 record helpers used by the abstraction stream
# ---------------------------------------------------------------------------


def build_abstraction_tasks(seed: int, k: int = 4, n_test: int = 20,
                            n_probe: int = 4) -> list[ExpHTask]:
    """The shared-core stream from gen_abstraction_stream, as CCM tasks."""
    specs = gen_abstraction_stream(seed)
    out: list[ExpHTask] = []
    for i, (node, tag, name) in enumerate(specs):
        task_rng = random.Random(seed * 1000 + i)
        inputs = sample_lists(task_rng, k + 1 + n_test, 4, 9, 0, 9)

        def safe_label(v):
            try:
                return eval_hidden(node, v)
            except Exception:
                # inputs with no evens: fall back to label of max(0-list)=0 case
                return 0

        labels = [safe_label(v) for v in inputs]
        train = [(x, y) for x, y in zip(inputs[:k], labels[:k])]
        val = [(inputs[k], labels[k])]
        test = list(zip(inputs[k + 1:], labels[k + 1:]))
        probes = sample_lists(task_rng, n_probe, 4, 9, 0, 9)
        rule = str(node)
        tname = f"h6-{name.replace(' ', '_')}-s{seed}"
        out.append(build_task(tname, rule, train, val, test, "list", "int",
                              probes, tag, node,
                              {"stream_index": i, "k": k, "hidden_program": rule}))
    return out
