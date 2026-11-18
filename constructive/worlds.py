"""Synthetic worlds for the constructive-learning experiments.

Every task is generated from a hidden program expressed in the same DSL the
learner uses. Training, validation and test inputs come from one RNG stream but
are disjoint index ranges, and duplicates are rejected. Each task also carries
a list of *unlabelled* probe inputs from the same distribution: the learner uses
them to keep observationally distinct hypotheses apart without ever seeing a
label.

Task generators reject data that a different single primitive already explains,
so the intended rule is identifiable among the simple alternatives.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

from .core import COLORS, SHAPES, SIZES, Entity, Scene, Term, V, build_prims, eval_term, same_value

_PRIMS = build_prims()
# A deliberately *stronger* primitive set than the learner's default, used only
# to reject training data that a different single primitive already explains.
# Constants up to +/-20 and 0..12 mean `lt(12)`, `mod(9)` or `take(4)` cannot
# masquerade as an identifiable hidden rule.
_DISTRACTOR_PRIMS = build_prims(int_constants=tuple(range(-20, 21)))


@dataclass
class Task:
    family: str
    name: str
    rule: str
    train: list[tuple[Any, Any]]
    val: list[tuple[Any, Any]]
    test: list[tuple[Any, Any]]
    input_type: str
    output_type: str
    probes: list[Any] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def support(self) -> list[tuple[Any, Any]]:
        return list(self.train)


REL_RULES = ("left_of", "right_of", "above", "below", "same_color", "same_shape",
             "nearest", "farthest", "biggest", "smallest")
LIST_PRIMS = ("sum_list", "max_list", "min_list", "range_list", "count_even",
              "count_odd", "sum_squares", "head", "last", "len_list")
MATH_RULES = ("square", "double", "sum_digits", "digital_root", "fib",
              "is_prime", "even", "odd")


def _t(name: str, *args: Term) -> Term:
    return Term(name, tuple(args))


def _x() -> Term:
    return V("x")


#: Hidden list programs with primitive cost 2-3.  Stored as terms, never as
#: Python lambdas, so the hidden rule really lives in the learner's own language.
DERIVED_LIST_TERMS: dict[str, Term] = {
    "sum_of_evens": _t("sum_list", _t("evens", _x())),
    "distinct_count": _t("len_list", _t("unique", _x())),
    "max_shifted": _t("max_list", _t("map_add(3)", _x())),
    "sum_top3": _t("sum_list", _t("take(3)", _t("sort_desc", _x()))),
    "even_square_sum": _t("sum_list", _t("map_mul(2)", _t("evens", _x()))),
}

#: Composition primitives: two independently teachable concepts.
COMPOSE_A = _t("evens", _x())
COMPOSE_B = _t("sum_list", _t("take(3)", _t("sort_desc", _x())))
COMPOSE_TARGET = _t("sum_list", _t("take(3)", _t("sort_desc", _t("evens", _x()))))


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------


def gen_scene(rng: random.Random, n: int | None = None, mirror: bool = False) -> Scene:
    n = n or rng.randint(4, 6)
    positions = rng.sample([(x, y) for x in range(5) for y in range(5)], n)
    ents = []
    for i, (x, y) in enumerate(positions):
        if mirror:
            x = 4 - x
        ents.append(Entity(f"e{i}", rng.choice(COLORS), rng.choice(SHAPES), rng.choice(SIZES), x, y))
    q = rng.choice(ents).id
    return Scene(tuple(ents), q, mirror)


def apply_rel(rule: str, scene: Scene) -> Entity:
    prim = _PRIMS[rule]
    if rule in ("biggest", "smallest"):
        return prim.fn(scene)
    return prim.fn(scene, scene.by_id(scene.query))


def _rel_defined(rule: str, scene: Scene) -> bool:
    try:
        apply_rel(rule, scene)
        return True
    except Exception:
        return False


def _rel_distractor(rule: str, examples: list[tuple[Scene, Entity]]) -> bool:
    for other, p in _DISTRACTOR_PRIMS.items():
        if other == rule or p.ret != "entity":
            continue
        try:
            if p.args == ("scene",):
                if all(p.fn(s).id == y.id for s, y in examples):
                    return True
            elif p.args == ("scene", "entity"):
                if all(p.fn(s, s.by_id(s.query)).id == y.id for s, y in examples):
                    return True
        except Exception:
            continue
    return False


def relational_task(rule: str, seed: int, k: int = 3, n_test: int = 20, n_val: int = 1,
                    n_probe: int = 4) -> Task:
    rng = random.Random(seed)
    for _attempt in range(200):
        scenes: list[Scene] = []
        for _ in range(k + n_val + n_test):
            for _ in range(30):
                s = gen_scene(rng)
                if _rel_defined(rule, s):
                    scenes.append(s)
                    break
            else:
                raise RuntimeError(f"cannot generate scene for {rule}")
        train = [(s, apply_rel(rule, s)) for s in scenes[:k]]
        val = [(s, apply_rel(rule, s)) for s in scenes[k:k + n_val]]
        test = [(s, apply_rel(rule, s)) for s in scenes[k + n_val:]]
        if k >= 2 and _rel_distractor(rule, train + val):
            continue
        if k >= 2 and len({y.id for _, y in train + val}) < 2:
            continue
        probes = [gen_scene(rng) for _ in range(n_probe)]
        return Task("relational", f"rel-{rule}", rule, train, val, test, "scene", "entity", probes)
    raise RuntimeError(f"could not disambiguate relational rule {rule}")


# ---------------------------------------------------------------------------
# Integer lists
# ---------------------------------------------------------------------------


def gen_list(rng: random.Random, n: int | None = None) -> list[int]:
    n = n or rng.randint(5, 9)
    return [rng.randint(0, 9) for _ in range(n)]


def _list_distractor(rule: str, examples: list[tuple[list[int], Any]]) -> bool:
    for other, p in _DISTRACTOR_PRIMS.items():
        if other == rule or p.args != ("list",):
            continue
        try:
            if all(same_value(p.fn(x), y) for x, y in examples):
                return True
        except Exception:
            continue
    for other, term in DERIVED_LIST_TERMS.items():
        if other == rule:
            continue
        try:
            if all(same_value(eval_term(term, {"x": x}, _DISTRACTOR_PRIMS), y)
                   for x, y in examples):
                return True
        except Exception:
            continue
    return False


def _list_label(rule: str, xs: list[int]) -> Any:
    if rule in _PRIMS:
        return _PRIMS[rule].fn(xs)
    return eval_term(DERIVED_LIST_TERMS[rule], {"x": xs}, _PRIMS)


def list_rule_task(rule: str, seed: int, k: int = 3, n_test: int = 20, n_val: int = 1,
                   n_probe: int = 4) -> Task:
    rng = random.Random(seed)
    for _attempt in range(300):
        xs: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        while len(xs) < k + n_val + n_test:
            x = gen_list(rng)
            t = tuple(x)
            if t in seen:
                continue
            seen.add(t)
            xs.append(x)
        train = [(x, _list_label(rule, x)) for x in xs[:k]]
        val = [(x, _list_label(rule, x)) for x in xs[k:k + n_val]]
        test = [(x, _list_label(rule, x)) for x in xs[k + n_val:]]
        if k >= 2 and _list_distractor(rule, train + val):
            continue
        if k >= 2 and len({repr(y) for _, y in train}) < 2:
            continue
        if len(val) >= 2 and len({repr(y) for _, y in val}) < 2:
            continue
        probes = [gen_list(rng) for _ in range(n_probe)]
        out_type = "list" if rule in ("unique", "sort", "sort_desc", "reverse") else "int"
        return Task("rule", f"list-{rule}", rule, train, val, test, "list", out_type, probes)
    raise RuntimeError(f"could not disambiguate list rule {rule}")


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------


def _math_apply(rule: str, x: Any) -> Any:
    return _PRIMS[rule].fn(x)


def _math_distractor(rule: str, examples: list[tuple[Any, Any]], input_type: str,
                     output_type: str) -> bool:
    for other, p in _DISTRACTOR_PRIMS.items():
        if other == rule or p.args != (input_type,) or p.ret != output_type:
            continue
        try:
            if all(same_value(p.fn(x), y) for x, y in examples):
                return True
        except Exception:
            continue
    return False


def math_task(rule: str, seed: int, k: int = 3, n_test: int = 20, pair: bool = False,
              n_probe: int = 4) -> Task:
    rng = random.Random(seed)
    constants = [0, 1, 2, 3]
    for _attempt in range(300):
        vals: list[Any] = []
        seen: set[Any] = set()
        while len(vals) < k + 1 + n_test:
            v = (rng.randint(0, 12), rng.randint(0, 12)) if pair else rng.randint(0, 30)
            if v in seen:
                continue
            seen.add(v)
            vals.append(v)
        train = [(x, _math_apply(rule, x)) for x in vals[:k]]
        val = [(x, _math_apply(rule, x)) for x in vals[k:k + 1]]
        test = [(x, _math_apply(rule, x)) for x in vals[k + 1:]]
        if k >= 2 and _math_distractor(rule, train + val, "pair" if pair else "int",
                                        "bool" if rule in ("is_prime", "even", "odd") else "int"):
            continue
        if k >= 2 and len({repr(y) for _, y in train}) < 2:
            continue
        if len(val) >= 2 and len({repr(y) for _, y in val}) < 2:
            continue
        probes = [(rng.randint(0, 12), rng.randint(0, 12)) if pair else rng.randint(0, 30)
                  for _ in range(n_probe)]
        out_type = "bool" if rule in ("is_prime", "even", "odd") else "int"
        return Task("math", f"math-{rule}", rule, train, val, test,
                    "pair" if pair else "int", out_type, probes)
    raise RuntimeError(f"could not disambiguate math rule {rule}")


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def _label_term(term: Term, x: Any) -> Any:
    return eval_term(term, {"x": x}, _PRIMS)


def _fresh_lists(rng: random.Random, count: int, seen: set[tuple[int, ...]]) -> list[list[int]]:
    out: list[list[int]] = []
    while len(out) < count:
        x = gen_list(rng, n=rng.randint(6, 9))
        t = tuple(x)
        if t in seen:
            continue
        seen.add(t)
        out.append(x)
    return out


def concept_a_task(seed: int, k: int = 4, n_val: int = 2) -> Task:
    """Teach concept A: evens (list -> list)."""
    rng = random.Random(seed)
    seen: set[tuple[int, ...]] = set()
    xs = _fresh_lists(rng, k + n_val + 12, seen)
    train = [(x, _label_term(COMPOSE_A, x)) for x in xs[:k]]
    val = [(x, _label_term(COMPOSE_A, x)) for x in xs[k:k + n_val]]
    test = [(x, _label_term(COMPOSE_A, x)) for x in xs[k + n_val:]]
    probes = _fresh_lists(rng, 4, seen)
    return Task("composition-part", "compose-A-evens", "evens", train, val, test, "list", "list", probes)


def concept_b_task(seed: int, k: int = 4, n_val: int = 2) -> Task:
    """Teach concept B: sum(take(3, sort_desc(x))) (list -> int)."""
    rng = random.Random(seed)
    seen: set[tuple[int, ...]] = set()
    xs = _fresh_lists(rng, k + n_val + 12, seen)
    train = [(x, _label_term(COMPOSE_B, x)) for x in xs[:k]]
    val = [(x, _label_term(COMPOSE_B, x)) for x in xs[k:k + n_val]]
    test = [(x, _label_term(COMPOSE_B, x)) for x in xs[k + n_val:]]
    probes = _fresh_lists(rng, 4, seen)
    return Task("composition-part", "compose-B-top3", "sum_top3", train, val, test, "list", "int", probes)


def composition_task(seed: int, k: int = 4, n_test: int = 20, n_val: int = 1) -> Task:
    """Target: sum(take(3, sort_desc(evens(x)))) - A then B, never trained together."""
    rng = random.Random(seed)
    seen: set[tuple[int, ...]] = set()
    for _ in range(200):
        xs = _fresh_lists(rng, k + n_val + n_test, seen)
        train = [(x, _label_term(COMPOSE_TARGET, x)) for x in xs[:k]]
        val = [(x, _label_term(COMPOSE_TARGET, x)) for x in xs[k:k + n_val]]
        test = [(x, _label_term(COMPOSE_TARGET, x)) for x in xs[k + n_val:]]
        # the target must differ from B alone and from simpler aggregates
        if any(_label_term(COMPOSE_B, x) == y for x, y in train + val + test):
            continue
        simple: list[Callable[[list[int]], Any]] = [
            lambda v: sum(v),
            lambda v: sum(sorted(v, reverse=True)[:3]),
            lambda v: sum(e for e in v if e % 2 == 0),
        ]
        if any(all(f(x) == y for x, y in train + val) for f in simple):
            continue
        probes = _fresh_lists(rng, 4, seen)
        return Task("composition", "compose-A-then-B", "sum_top3(evens(x))",
                    train, val, test, "list", "int", probes)
    raise RuntimeError("could not build a distinct composition task")


# ---------------------------------------------------------------------------
# Noise, contradiction, continual suites
# ---------------------------------------------------------------------------


def noisy_task(base: Task, seed: int, flip: int = 1) -> Task:
    """Corrupt a few training labels; validation and test stay clean."""
    rng = random.Random(seed)
    train = list(base.train)
    idx = rng.sample(range(len(train)), min(flip, len(train)))
    outputs = [y for _, y in base.train]
    flipped = 0
    flipped_idx: list[int] = []
    for i in idx:
        x, y = train[i]
        candidates = [o for o in outputs if not same_value(o, y)]
        if not candidates:
            continue
        train[i] = (x, rng.choice(candidates))
        flipped += 1
        flipped_idx.append(i)
    return Task("noise", base.name + "-noisy", base.rule, train, base.val, base.test,
                base.input_type, base.output_type, base.probes,
                {"clean_rule": base.rule, "flipped": flipped, "flipped_indices": sorted(flipped_idx)})


def contradictory_task(rule_a: str, rule_b: str, seed: int, k: int = 4, n_test: int = 20) -> Task:
    """Half the training labels follow rule_a, half follow rule_b.

    No single program in the DSL explains them, so a learner that accepts a
    hypothesis merely because it is the best available is wrong. Validation and
    test follow rule_a.
    """
    base_a = math_task(rule_a, seed, k=k, n_test=n_test)
    rng = random.Random(seed + 7)
    train = list(base_a.train)
    half = max(1, k // 2)
    for i in range(half):
        x, _ = train[i]
        train[i] = (x, _math_apply(rule_b, x))
    return Task("contradiction", f"contradict-{rule_a}-{rule_b}", rule_a, train, base_a.val,
                base_a.test, "int", base_a.output_type, base_a.probes,
                {"rules": [rule_a, rule_b], "split": half})


def continual_suite(seed: int) -> list[Task]:
    return [
        relational_task("left_of", seed + 1),
        list_rule_task("sum_list", seed + 2),
        math_task("square", seed + 3),
        relational_task("same_shape", seed + 4),
        list_rule_task("sum_of_evens", seed + 5),
        math_task("add(3)", seed + 6),
    ]


def retention_suite(seed: int) -> list[Task]:
    """A second, disjoint batch used only to test retention after new learning."""
    return [
        relational_task("same_shape", seed + 41, k=4),
        list_rule_task("sum_squares", seed + 42, k=4),
        math_task("double", seed + 43, k=4),
    ]


# ---------------------------------------------------------------------------
# Self-correction scenario
# ---------------------------------------------------------------------------


def _scene_batch(rng: random.Random, mirror: bool, n: int) -> list[tuple[Scene, Entity]]:
    out: list[tuple[Scene, Entity]] = []
    while len(out) < n:
        s = gen_scene(rng, mirror=mirror)
        try:
            y = _PRIMS["right_of" if mirror else "left_of"].fn(s, s.by_id(s.query))
        except Exception:
            continue
        out.append((s, y))
    return out


def mirrored_scene_task(seed: int, k: int = 4, n_test: int = 20) -> Task:
    """Learn left_of on normal scenes, then meet mirrored scenes needing right_of."""
    rng = random.Random(seed)
    train = _scene_batch(rng, False, k)
    val = _scene_batch(rng, False, 1)
    test = _scene_batch(rng, False, n_test)
    return Task("correction", "mirror-left-right", "left_of", train, val, test, "scene", "entity",
                [gen_scene(rng) for _ in range(4)],
                {"shift": "mirrored scenes require right_of"})


def mirrored_shift_task(seed: int, k: int = 4, n_test: int = 20) -> Task:
    """The shifted context: mirrored scenes labelled by right_of."""
    rng = random.Random(seed + 1000)
    train = _scene_batch(rng, True, k)
    val = _scene_batch(rng, True, 1)
    test = _scene_batch(rng, True, n_test)
    return Task("correction", "mirror-shift", "right_of", train, val, test, "scene", "entity",
                [gen_scene(rng, mirror=True) for _ in range(4)],
                {"context": "mirrored"})


def mirrored_test_examples(seed: int, n: int = 20) -> list[tuple[Scene, Entity]]:
    rng = random.Random(seed + 2000)
    return _scene_batch(rng, True, n)
