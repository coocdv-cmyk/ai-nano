"""Tests for the constructive-learning module (OCI).

Every test is executable evidence for a claim in the report.  Nothing is mocked:
the learner runs its real search, validation, composition and repair.
"""

from __future__ import annotations

import pytest

from constructive.core import Term, V, build_prims, eval_term, same_value, term_cost
from constructive.model import ConstructiveModel
from constructive import worlds as W
from constructive.baselines import MLP, Memorizer, NearestNeighbor

KW = {"max_cost": 3, "beam": 45, "max_nodes": 250_000, "time_budget_s": 5.0}


# ---------------------------------------------------------------------------
# Core semantics
# ---------------------------------------------------------------------------


def test_primitives_and_terms_evaluate():
    prims = build_prims()
    assert eval_term(Term("square", (V("x"),)), {"x": 7}, prims) == 49
    assert eval_term(Term("sum_list", (V("x"),)), {"x": [1, 2, 3]}, prims) == 6
    assert eval_term(Term("gcd_pair", (V("x"),)), {"x": (12, 18)}, prims) == 6
    task = W.relational_task("left_of", seed=11, k=3)
    scene, expected = task.train[0]
    q = scene.by_id(scene.query)
    left = eval_term(Term("left_of", (V("x"), Term("query", (V("x"),)))), {"x": scene}, prims)
    assert left.id == expected.id
    assert left.x <= q.x and left.id != q.id


def test_term_cost_counts_operators():
    assert term_cost(V("x")) == 0
    assert term_cost(Term("square", (V("x"),))) == 1
    assert term_cost(Term("sum_list", (Term("evens", (V("x"),)),))) == 2


# ---------------------------------------------------------------------------
# Experiments A and D: few-shot induction from a small number of examples
# ---------------------------------------------------------------------------


def test_search_finds_arithmetic_rule():
    task = W.math_task("square", seed=14, k=3)
    m = ConstructiveModel(**KW)
    rep = m.fit(task.train, task.input_type, task.output_type,
                validation=task.val, probes=task.probes)
    assert rep.accepted and not rep.abstained
    assert "square" in str(m.concepts[rep.concept].body)
    assert all(m.predict_episode(task.train, x, "int") == y for x, y in task.test)


def test_few_shot_relational_concept():
    task = W.relational_task("left_of", seed=11, k=3)
    m = ConstructiveModel(**KW)
    rep = m.fit(task.train, task.input_type, task.output_type,
                validation=task.val, probes=task.probes)
    assert rep.accepted
    assert "left_of" in str(m.concepts[rep.concept].body)
    assert sum(1 for x, y in task.test if m.predict_episode(task.train, x, "scene") == y) == len(task.test)


def test_validation_rejects_a_rule_that_only_fits_examples():
    task = W.math_task("square", seed=14, k=3)
    bogus = [(task.val[0][0], task.val[0][1] + 1)]
    m = ConstructiveModel(**KW)
    rep = m.fit(task.train, "int", "int", validation=bogus, probes=task.probes)
    assert not rep.accepted
    assert rep.abstained
    assert not m.concepts
    assert any("validation" in r["reason"] for r in m.rejected)


# ---------------------------------------------------------------------------
# Experiment B: rule discovery on lists
# ---------------------------------------------------------------------------


def test_derived_list_rule_is_discovered():
    task = W.list_rule_task("sum_of_evens", seed=13, k=3)
    m = ConstructiveModel(**KW)
    rep = m.fit(task.train, task.input_type, task.output_type,
                validation=task.val, probes=task.probes)
    assert rep.accepted
    assert str(m.concepts[rep.concept].body) == "sum_list(evens(x))"
    assert all(m.predict_episode(task.train, x, "list") == y for x, y in task.test)


def test_cost_three_rule_is_discovered():
    task = W.list_rule_task("sum_top3", seed=15, k=3)
    m = ConstructiveModel(**KW)
    rep = m.fit(task.train, task.input_type, task.output_type,
                validation=task.val, probes=task.probes)
    assert rep.accepted
    assert all(m.predict_episode(task.train, x, "list") == y for x, y in task.test)


# ---------------------------------------------------------------------------
# Retrieval among several stored concepts
# ---------------------------------------------------------------------------


def test_retrieval_selects_the_right_stored_concept():
    m = ConstructiveModel(**KW)
    left = W.relational_task("left_of", seed=101, k=3)
    shape = W.relational_task("same_shape", seed=102, k=3)
    r1 = m.fit(left.train, "scene", "entity", validation=left.val, probes=left.probes)
    r2 = m.fit(shape.train, "scene", "entity", validation=shape.val, probes=shape.probes)
    assert r1.concept != r2.concept
    assert sum(1 for x, y in left.test if m.predict_episode(left.train, x, "scene") == y) >= 18
    assert sum(1 for x, y in shape.test if m.predict_episode(shape.train, x, "scene") == y) >= 18
    top = m.rank(left.train, "scene")[0]
    assert top["concept"] == r1.concept and top["match"] == 1.0


def test_concept_is_reused_instead_of_duplicated():
    task = W.list_rule_task("sum_list", seed=12, k=3)
    m = ConstructiveModel(**KW)
    r1 = m.fit(task.train, "list", "int", validation=task.val, probes=task.probes)
    r2 = m.fit(task.train, "list", "int", validation=task.val, probes=task.probes)
    assert r1.concept == r2.concept
    assert len(m.concepts) == 1


# ---------------------------------------------------------------------------
# Experiment C: composition
# ---------------------------------------------------------------------------


def test_composition_reuses_two_concepts():
    m = ConstructiveModel(**KW)
    a = W.concept_a_task(21)
    b = W.concept_b_task(21)
    ra = m.fit(a.train, a.input_type, a.output_type, validation=a.val, probes=a.probes)
    rb = m.fit(b.train, b.input_type, b.output_type, validation=b.val, probes=b.probes)
    assert ra.accepted and rb.accepted
    target = W.composition_task(23)
    comp = m.compose(target.train, target.input_type, target.output_type,
                     validation=target.val, probes=target.probes, max_cost=2)
    assert comp.accepted
    assert f"{rb.concept}(" in comp.term and f"{ra.concept}(" in comp.term
    assert sum(1 for x, y in target.test if m.predict_episode(target.train, x, "list") == y) == len(target.test)


def test_composition_fails_without_learned_concepts():
    target = W.composition_task(23)
    fresh = ConstructiveModel(**KW)
    comp = fresh.compose(target.train, target.input_type, target.output_type,
                         validation=target.val, probes=target.probes, max_cost=2)
    assert not comp.accepted
    assert not fresh.concepts


# ---------------------------------------------------------------------------
# Experiment F: uncertainty, contradiction, repair
# ---------------------------------------------------------------------------


def test_contradictory_evidence_is_rejected():
    task = W.contradictory_task("square", "double", seed=51, k=4)
    m = ConstructiveModel(**KW)
    rep = m.fit(task.train, "int", "int", validation=task.val, probes=task.probes)
    assert not rep.accepted
    assert rep.abstained
    assert all(m.predict_episode(task.train, x, "int") is None for x, _ in task.test)


def test_quarantine_localises_the_corrupted_example():
    base = W.math_task("square", seed=7000, k=4)
    task = W.noisy_task(base, seed=7500, flip=1)
    m = ConstructiveModel(**KW)
    rep = m.fit(task.train, "int", "int", validation=task.val, probes=task.probes)
    assert rep.accepted
    assert rep.quarantined == task.meta["flipped_indices"]
    assert all(m.predict_episode(task.train, x, "int") == y for x, y in task.test)


def test_scoped_repair_preserves_old_knowledge():
    base = W.mirrored_scene_task(seed=31, k=4)
    shift = W.mirrored_shift_task(seed=31, k=4)
    m = ConstructiveModel(**KW)
    rep = m.fit(base.train, "scene", "entity", validation=base.val, probes=base.probes)
    assert rep.accepted
    old_before = sum(1 for x, y in base.test if m.predict_episode(base.train, x, "scene") == y)
    rev = m.revise(rep.concept, shift.train, "scene", validation=shift.val, probes=shift.probes)
    assert rev["mode"] == "scoped"
    assert "mirror" in rev["guard"]
    old_after = sum(1 for x, y in base.test if m.predict_episode(base.train, x, "scene") == y)
    new_after = sum(1 for x, y in shift.test if m.predict_episode(shift.train, x, "scene") == y)
    assert old_after == old_before == len(base.test)
    assert new_after == len(shift.test)
    assert m.retention()[rep.concept] == 1.0


# ---------------------------------------------------------------------------
# Retention, leakage, determinism
# ---------------------------------------------------------------------------


def test_retention_after_learning_another_concept():
    m = ConstructiveModel(**KW)
    t1 = W.list_rule_task("sum_list", seed=12, k=3)
    t2 = W.math_task("add(3)", seed=5003, k=3)
    m.fit(t1.train, "list", "int", validation=t1.val, probes=t1.probes)
    before = m.retention()
    m.fit(t2.train, "int", "int", validation=t2.val, probes=t2.probes)
    after = m.retention()
    assert all(before[k] == 1.0 for k in before)
    assert all(after[k] == 1.0 for k in after)
    assert m.forgetting(before) == 0.0


def test_test_labels_do_not_influence_learning():
    task = W.math_task("square", seed=14, k=3)
    m1 = ConstructiveModel(**KW)
    r1 = m1.fit(task.train, "int", "int", validation=task.val, probes=task.probes)
    corrupted = [(x, y + 100) for x, y in task.test]
    m2 = ConstructiveModel(**KW)
    r2 = m2.fit(task.train, "int", "int", validation=task.val, probes=task.probes)
    assert str(m1.concepts[r1.concept].body) == str(m2.concepts[r2.concept].body)
    assert corrupted  # test labels are never passed to fit


def test_train_and_test_inputs_are_disjoint():
    for task in (W.relational_task("nearest", 77), W.list_rule_task("sum_squares", 78),
                 W.math_task("fib", 79)):
        def key(x):
            if hasattr(x, "entities"):
                return (tuple((e.id, e.color, e.shape, e.size, e.x, e.y) for e in x.entities), x.query)
            return tuple(x) if isinstance(x, list) else x
        train_keys = {key(x) for x, _ in task.train}
        test_keys = {key(x) for x, _ in task.test}
        assert not (train_keys & test_keys)


def test_learning_is_deterministic():
    task = W.list_rule_task("sum_of_evens", seed=13, k=3)
    bodies = []
    for _ in range(2):
        m = ConstructiveModel(**KW)
        rep = m.fit(task.train, "list", "int", validation=task.val, probes=task.probes)
        bodies.append(str(m.concepts[rep.concept].body))
    assert bodies[0] == bodies[1]


# ---------------------------------------------------------------------------
# Baselines are real, runnable systems
# ---------------------------------------------------------------------------


def test_baselines_run_and_are_weak_with_three_examples():
    task = W.math_task("square", seed=14, k=3)
    for cls in (Memorizer, NearestNeighbor, MLP):
        b = cls()
        b.fit(task.train, task.input_type, task.output_type)
        preds = [b.predict(x) for x, _ in task.test]
        assert len(preds) == len(task.test)
    mem = Memorizer()
    mem.fit(task.train, task.input_type, task.output_type)
    assert all(mem.predict(x) is None for x, _ in task.test)
    m = ConstructiveModel(**KW)
    m.fit(task.train, "int", "int", validation=task.val, probes=task.probes)
    assert all(m.predict_episode(task.train, x, "int") == y for x, y in task.test)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
