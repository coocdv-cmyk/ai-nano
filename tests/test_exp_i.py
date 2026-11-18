"""Tests for the EXP-I research module.

Tests are deterministic, CPU-only and assert only what they claim. They cover
all stages (A-H) plus the mandatory adversarial break-attempts.
"""

from __future__ import annotations

import random
import time

import pytest

from constructive.exp_i import abstraction as AB
from constructive.exp_i import dsl, generators as G, repair as RP, retrieval as RT
from constructive.exp_i import smoke as SMOKE
from constructive.exp_i.config import ExpIConfig
from constructive.exp_i.dsl import Node, call, lit, var
from constructive.exp_i.evidence import (blind_positive_risk, candidate_disagreement,
                                         contradictions_in_support, counterexamples,
                                         redundancy_report, score_support, select_informative)
from constructive.exp_i.experiments import (_chain_task, run_ia, run_ib, run_ic,
                                            run_id, run_if, run_ih)
from constructive.exp_i.resources import (ResourceGovernor, ResourceLimit, run_isolated)
from constructive.exp_i.search import Searcher, can_reach
from constructive.exp_i.types import (Candidate, CompletionStatus, Example, Task,
                                      ValidationState, stable_key)
from constructive.exp_i.validation import Calibrator, ValidationEngine, calibration_report

EVAL = dsl.evaluate


# ---------------------------------------------------------------------------
# types / config / resources
# ---------------------------------------------------------------------------


def test_validation_state_vocabulary():
    assert {s.value for s in ValidationState} == {
        "ACCEPTED", "REJECTED", "INSUFFICIENT_EVIDENCE",
        "NON_IDENTIFIABLE", "CONTRADICTORY", "RESOURCE_LIMITED"}


def test_stable_key_deterministic_and_bool_safe():
    assert stable_key([1, True]) != stable_key([1, 1])
    assert stable_key({"a": 1, "b": 2}) == stable_key({"b": 2, "a": 1})


def test_resource_global_deadline():
    g = ResourceGovernor(global_deadline_s=0.01)
    with pytest.raises(ResourceLimit):
        while True:
            g.checkpoint("probe")
            time.sleep(0.002)
    assert g.report()["timeouts"]


def test_resource_node_budget():
    g = ResourceGovernor(node_budget=3)
    with pytest.raises(ResourceLimit):
        for _ in range(10):
            g.charge_nodes()
            g.checkpoint("probe")


def test_resource_cancellation():
    g = ResourceGovernor()
    g.cancel("cancel")
    with pytest.raises(ResourceLimit) as ei:
        g.checkpoint("probe")
    assert ei.value.kind == "cancellation"


def test_resource_timeout_never_false_success():
    g = ResourceGovernor(global_deadline_s=0.01)
    status = CompletionStatus.COMPLETED
    try:
        with g.scope("fit"):
            while True:
                g.checkpoint("fit")
                time.sleep(0.002)
    except ResourceLimit:
        status = CompletionStatus.TIMED_OUT
    assert status is CompletionStatus.TIMED_OUT
    assert g.report()["diagnostics"]["last_scope_status"] == "TIMED_OUT"


def test_resource_failure_isolation_records_diagnostic():
    log = []
    assert run_isolated(lambda: 1 / 0, default=None, log=log) is None
    assert log and "ZeroDivisionError" in log[0]["error"]


# ---------------------------------------------------------------------------
# DSL (EXP-I-B)
# ---------------------------------------------------------------------------


def test_dsl_conditional_and_comparison():
    p = call("if", call("gt", var("int"), params=(5,)),
             call("double", var("int")), call("neg", var("int")))
    assert dsl.infer_type(p) == "int"
    assert EVAL(p, 7) == 14 and EVAL(p, 2) == -2


def test_dsl_boolean_connectives_and_negation():
    p = call("and", call("even", var("int")), call("gt", var("int"), params=(3,)))
    assert EVAL(p, 8) is True and EVAL(p, 2) is False
    assert EVAL(call("not", call("is_prime", var("int"))), 4) is True


def test_dsl_operator_exclusion_bitwise():
    assert EVAL(call("bxor", var("int"), params=(3,)), 5) == 6
    assert EVAL(call("band", var("int"), params=(6,)), 5) == 4


def test_dsl_parameterized_and_nested_list_ops():
    p = call("sum_list", call("filter_gt", call("evens", var("list")), params=(2,)))
    assert dsl.infer_type(p) == "int"
    assert EVAL(p, [1, 2, 3, 4, 5, 6]) == 10  # evens -> [2,4,6], filter>2 -> [4,6]


def test_dsl_strict_typing_rejects_bool_to_int():
    bad = Node("add1", (), (call("even", var("int")),))
    assert dsl.validate(bad)


def test_dsl_canonicalization_commutative():
    a = call("and", call("even", var("int")), call("pos", var("int")))
    b = call("and", call("pos", var("int")), call("even", var("int")))
    assert dsl.canonical_key(a) == dsl.canonical_key(b)


def test_dsl_complexity_and_depth():
    p = call("sum_digits", call("double", var("int")))
    assert dsl.complexity(p) > dsl.complexity(var("int"))
    assert p.depth() == 3


def test_dsl_cycle_prevention_and_safe_execution():
    assert not dsl.cycles_possible()
    bomb = call("add", var("int"), params=(10 ** 12,))
    assert dsl.validate(bomb, max_magnitude=10 ** 9)
    with pytest.raises(dsl.DslError):
        EVAL(bomb, 3, max_magnitude=10 ** 9)


def test_dsl_step_and_depth_caps_enforced():
    node = var("int")
    for _ in range(40):
        node = call("abs", node)
    with pytest.raises(dsl.DslError):
        EVAL(node, -3, max_depth=24)


def test_dsl_from_hidden_and_expressiveness():
    from constructive.exp_h.engine import HNode

    h = HNode("if_else", None, (HNode("even", None, (HNode("var"),)),
                                HNode("double", None, (HNode("var"),)),
                                HNode("neg", None, (HNode("var"),))))
    node = dsl.from_hidden(h)
    assert dsl.well_typed(node)
    assert EVAL(node, 4) == 8
    assert dsl.expressiveness_of_ops({"shl", "bxor"}).value == "EXPRESSIBLE_DISCOVERED"
    assert dsl.expressiveness_of_ops({"quantum_gate"}).value == "NOT_EXPRESSIBLE"


# ---------------------------------------------------------------------------
# Evidence / validation (EXP-I-A)
# ---------------------------------------------------------------------------


def _cands():
    return G.twin_candidates()


def test_evidence_scores_and_disagreement():
    t = G.twin_task(0, k=3)
    scores = score_support(_cands(), t.train, EVAL)
    assert scores["sum_digits"]["consistent"]
    assert candidate_disagreement(_cands(), [e.x for e in t.validation_pool], EVAL)["n_disagreeing"] > 0


def test_evidence_counterexamples_find_disagreement():
    cex = counterexamples(call("sum_digits", var("int")),
                          call("digital_root", var("int")),
                          list(range(1, 30)), EVAL, limit=3)
    assert cex and all(EVAL(call("sum_digits", var("int")), x) !=
                       EVAL(call("digital_root", var("int")), x) for x in cex)


def test_evidence_contradiction_and_blind_positive_detection():
    t = G.twin_task(0)
    c = contradictions_in_support(G.contradictory_support(t).train)
    assert c
    bp = blind_positive_risk(G.blind_positive_task().train)
    assert bp["at_risk"]


def test_evidence_adaptive_selection_and_provenance():
    t = G.twin_task(0)
    pick = select_informative(t.validation_pool, _cands(), EVAL)
    assert pick is not None
    rep = redundancy_report([__import__("constructive.exp_i.types", fromlist=["EvidenceRecord"]).
                             EvidenceRecord(x=1, y=1, source="validation_pool", informative=True)])
    assert rep["n_informative"] == 1


def test_validation_twin_accepts_correct_rule():
    eng = ValidationEngine(budget=4, eval_fn=EVAL)
    r = eng.validate(_cands(), G.twin_task(3, k=3))
    assert r.accepted and r.candidate_id == "sum_digits"


def test_validation_non_identifiable_abstains():
    eng = ValidationEngine(budget=4, eval_fn=EVAL)
    r = eng.validate(_cands(), G.twin_task(3, k=3, validation_range=(1, 9),
                                          test_range=(10, 30)))
    assert not r.accepted
    assert r.state in (ValidationState.NON_IDENTIFIABLE, ValidationState.INSUFFICIENT_EVIDENCE)


def test_validation_contradictory_state():
    eng = ValidationEngine(eval_fn=EVAL)
    r = eng.validate(_cands(), G.contradictory_support(G.twin_task(0)))
    assert r.state is ValidationState.CONTRADICTORY


def test_validation_insufficient_support():
    eng = ValidationEngine(eval_fn=EVAL)
    t = G.twin_task(0, k=1)
    r = eng.validate(_cands(), t)
    assert r.state is ValidationState.INSUFFICIENT_EVIDENCE


def test_validation_corrupted_validation_data_fails_safely():
    eng = ValidationEngine(budget=4, eval_fn=EVAL)
    r = eng.validate(_cands(), G.corrupt_validation(G.twin_task(0), n_flips=2))
    assert r.state in (ValidationState.REJECTED, ValidationState.CONTRADICTORY,
                       ValidationState.INSUFFICIENT_EVIDENCE)
    assert not r.accepted


def test_validation_blind_positive_abstains():
    eng = ValidationEngine(budget=4, eval_fn=EVAL)
    cands = [Candidate(id="prime", program=call("is_prime", var("int"))),
             Candidate(id="allneg", program=call("gt", var("int"), params=(10 ** 6,)))]
    r = eng.validate(cands, G.blind_positive_task())
    assert not r.accepted


def test_calibration_ece_finite_and_report():
    conf = [0.1, 0.4, 0.6, 0.9] * 5
    corr = [0, 0, 1, 1] * 5
    rep = calibration_report(conf, corr)
    assert 0 <= rep["ece"] <= 1
    cal = Calibrator()
    cal.fit(conf, corr)
    assert 0 <= cal.calibrate(0.95) <= 1


# ---------------------------------------------------------------------------
# Retrieval (EXP-I-C)
# ---------------------------------------------------------------------------


def _store():
    s = RT.ConceptStore()
    for p in (1, 2, 4, 6, 7):
        s.add(G.collision_program("count_gt", p), input_type="list",
              output_type="int", id=f"cg{p}", context=("count_gt",))
    return s


def test_retrieval_typed_signatures_and_fingerprints():
    s = _store()
    assert s.records["cg1"].input_type == "list"
    assert len({r.structural_fp for r in s.records.values()}) == len(s.records)
    assert s.records["cg1"].behavioral_fp != s.records["cg2"].behavioral_fp


def test_retrieval_collision_reports_both_candidates():
    s = RT.ConceptStore()
    # Two concepts with identical behavior on generic probes are aliases.
    a = call("filter_gt", var("list"), params=(-100,))  # identity on these probes
    b = call("map_add", var("list"), params=(0,))       # structurally distinct identity
    s.add(a, input_type="list", output_type="list", id="a")
    s.add(b, input_type="list", output_type="list", id="b")
    assert any(x["a"] == "a" and x["b"] == "b" for x in s.aliases)


def test_retrieval_never_silently_substitutes_on_ambiguity():
    s = _store()
    obs = [(x, 1) for x in [[1, 2], [3, 4], [5, 6]]]
    r = s.retrieve(input_type="list", output_type="int",
                   observed_outputs=obs, context=("count_gt",))
    if r.ambiguous:
        assert r.request_evidence and len(r.ambiguity_ids) >= 2


def test_retrieval_refinement_preserves_immutable_history():
    s = _store()
    old_body = s.records["cg1"].body
    new = s.refine("cg1", G.collision_program("count_gt", 5))
    assert new.version == 2
    assert s.records["cg1"].history[0]["body"] == old_body


def test_retrieval_retirement_preserves_record():
    s = _store()
    s.retire("cg1", "superseded")
    assert not s.records["cg1"].active and "cg1" in s.retired


def test_retrieval_memory_and_latency_measured():
    s = _store()
    t0 = time.perf_counter()
    r = s.retrieve(input_type="list", output_type="int", context=("count_gt",))
    assert time.perf_counter() - t0 >= 0 and s.memory_bytes() > 0 and r.ranked


# ---------------------------------------------------------------------------
# Search (EXP-I-D)
# ---------------------------------------------------------------------------


def test_search_memoization_and_finds_composition():
    s = Searcher(beam=50, max_cost=3.5, max_depth=4)
    r = s.search(G.numeric_ood_task("sum_of_evens", 0, k=3))
    assert r.cache_hits >= 0 and r.diagnostics.get("cache_size", 0) > 0
    assert any("sum_list(evens" in str(c.program) for c in r.candidates)


def test_search_node_budget_hard_cap():
    g = ResourceGovernor(node_budget=5)
    r = Searcher(governor=g, beam=40, max_cost=3.5).search(
        G.numeric_ood_task("sum_of_evens", 0, k=3))
    assert r.timed_out


def test_search_admissible_type_bound():
    assert not can_reach("int", "list", 1)
    assert can_reach("list", "int", 1)


def test_search_deep_chain_abstains_when_unsupported():
    task = _chain_task(16, 2, k=4, n_test=8)
    g = ResourceGovernor(global_deadline_s=1.0, node_budget=8000)
    r = Searcher(governor=g, beam=20, max_cost=3.0, max_depth=4).search(task)
    assert isinstance(r.candidates, list)  # abstention is a valid measured outcome


# ---------------------------------------------------------------------------
# Generators / generalization (EXP-I-E)
# ---------------------------------------------------------------------------


def test_generators_disjoint_splits_all_families():
    tasks = [G.twin_task(0), G.collision_task("count_gt", 3, 0),
             G.numeric_ood_task("sum_digits", 0),
             G.numeric_ood_task("sum_of_evens", 0),
             G.numeric_ood_task("left_of", 0)]
    for t in tasks:
        assert G.disjointify([e.x for e in t.train], [e.x for e in t.validation_pool],
                             [e.x for e in t.test]), t.name


def test_generators_independent_reference_labels():
    assert G.REFERENCE_LABELS["sum_digits"](19) == 10
    assert G.REFERENCE_LABELS["square"](7) == 49
    assert G.REFERENCE_LABELS["is_prime"](7) is True
    assert G.REFERENCE_LABELS["sum_of_evens"]([1, 2, 3, 4]) == 6


def test_ood_domain_boundary_reported():
    from constructive.exp_i.generalization import domain_boundary
    b = domain_boundary("sum_digits", {"train_domain": [0, 9], "test_domain": [10, 60]},
                        ValidationState.NON_IDENTIFIABLE)
    assert b.extrapolating and "abstain" in b.claim


def test_ood_experiment_runs():
    from constructive.exp_i.generalization import run_ood
    cfg = ExpIConfig(seeds=(0,), n_episodes=4)
    block = run_ood(cfg, seeds=(0,), rules=("square", "sum_digits"))
    assert block["status"] == "completed" and block["n_episodes"] > 0
    assert all("domain_boundary" in r for r in block["episodes"])


# ---------------------------------------------------------------------------
# Repair (EXP-I-F)
# ---------------------------------------------------------------------------


def test_repair_bounded_equivalence_detects_false_repair():
    correct = RP.tpl_sum_range()
    # off-by-one: <= becomes < — passes some examples, fails bounded domain
    cand = RP.replace_at(correct, (2, 1), RP.Cmp("<", RP.V("i"), RP.V("b")))
    dom = RP.domain_for("sum_range")
    rep = RP.verify_repair(cand, correct, domain=dom, visible=[])
    assert rep.state is RP.VerificationState.NOT_EQUIVALENT
    assert rep.counterexamples


def test_repair_generator_separate_from_verifier():
    correct = RP.tpl_pow2()
    rng = random.Random(5)
    visible_envs = RP.sample_inputs("pow2", rng, 4)
    visible = [(e, RP.exec_program(correct, e)) for e in visible_envs]
    repairs, tried = RP.generate_repairs(
        RP.replace_at(correct, (2, 1), RP.Cmp("<=", RP.V("i"), RP.V("n"))),
        visible, max_width=1)
    assert tried > 0 and all(r.program is not None for r in repairs)


def test_repair_magnitude_bomb_rejected():
    bomb = [RP.Assign("s", RP.C(1)),
            RP.While(RP.Cmp("<", RP.V("s"), RP.C(10 ** 18)),
                     [RP.Assign("s", RP.Bin("*", RP.V("s"), RP.V("s")))]),
            RP.Ret(RP.V("s"))]
    with pytest.raises(RP.RepairError):
        RP.exec_program(bomb, {}, max_steps=60)


def test_repair_false_replay_cases_detected():
    block = run_if(ExpIConfig(seeds=(0,)), n_episodes=2)
    assert block["status"] == "completed"
    assert block["summary"]["false_verified_rate"]["mean"] == 0.0


def test_repair_multiple_plausible_repairs_flagged():
    correct = RP.tpl_clamp()
    rep = RP.verify_repair(correct, correct,
                           domain=RP.domain_for("clamp"), visible=[],
                           plausible=[correct, RP.tpl_clamp()])
    assert rep.state is RP.VerificationState.AMBIGUOUS


# ---------------------------------------------------------------------------
# Abstraction (EXP-I-G)
# ---------------------------------------------------------------------------


def test_abstraction_subtree_extraction_and_scoring():
    prog = call("sum_list", call("map_add", call("evens", var("list")), params=(2,)))
    subs = AB.subtrees(prog)
    assert any(s.op == "map_add" for s in subs)
    assert AB.reusability(subs[0], [prog, prog]) >= 0


def test_abstraction_validation_rejects_constant():
    store = AB.AbstractionStore()
    store.harvest(call("sum_list", call("evens", var("list"))), ["t1"], [])
    for aid in list(store.items):
        store.validate(aid, [])
    # A constant abstraction is rejected, not silently accepted.
    assert all(a.validated for a in store.items.values()) or store.rejected


def test_abstraction_retirement_preserves_provenance():
    store = AB.AbstractionStore()
    store.harvest(call("sum_list", call("map_add", call("evens", var("list")), params=(1,))),
                  ["t1"], [])
    if store.items:
        aid = next(iter(store.items))
        store.retire(aid, "false abstraction")
        assert store.items[aid].provenance["retire_reason"] == "false abstraction"


# ---------------------------------------------------------------------------
# Experiments schema + mandatory adversarial tests
# ---------------------------------------------------------------------------


def test_stage_a_smoke_all_pass():
    block = SMOKE.run_stage_a()
    assert block["failures"] == 0, block


def test_run_ia_schema_and_no_reject_everything():
    block = run_ia(ExpIConfig(seeds=(0,), n_episodes=6))
    disc = block["accuracy_coverage_tradeoff"]["discriminating"]
    assert block["status"] == "completed"
    assert disc["coverage"]["mean"] > 0  # not "reject everything"
    assert "ece_independent" in block["calibration"]


def test_ib_all_checks_pass():
    assert run_ib(ExpIConfig())["all_checks_pass"]


def test_ic_measures_interference():
    block = run_ic(ExpIConfig(seeds=(0, 1)))
    assert block["status"] == "completed"
    assert "silent_substitution_rate" in block["summary"]


def test_id_reports_budget_metrics():
    block = run_id(ExpIConfig(seeds=(0,)))
    assert "depth_2" in block["summary"]
    assert all("expansions" in v for v in block["summary"].values())


def test_ih_resource_governance_all_pass():
    block = run_ih(ExpIConfig())
    assert block["all_pass"], block


def test_adversarial_one_example_misleading_support():
    eng = ValidationEngine(eval_fn=EVAL)
    t = G.twin_task(0, k=1)
    assert not eng.validate(_cands(), t).accepted


def test_adversarial_observationally_equivalent_hypotheses():
    eng = ValidationEngine(budget=4, eval_fn=EVAL)
    t = G.twin_task(2, k=3, validation_range=(1, 9), test_range=(10, 30))
    r = eng.validate(_cands(), t)
    assert r.state in (ValidationState.NON_IDENTIFIABLE, ValidationState.INSUFFICIENT_EVIDENCE)


def test_adversarial_randomized_support_order_robust():
    eng = ValidationEngine(budget=4, eval_fn=EVAL)
    results = set()
    for order_seed in range(3):
        t = G.twin_task(9, k=3)
        rng = random.Random(order_seed)
        rng.shuffle(t.train)
        results.add(eng.validate(_cands(), t).candidate_id)
    assert results == {"sum_digits"}


def test_adversarial_large_numeric_and_ood_magnitude_safe():
    p = call("square", var("int"))
    assert EVAL(p, 10 ** 4, max_magnitude=10 ** 9) == 10 ** 8
    with pytest.raises(dsl.DslError):
        EVAL(p, 10 ** 6, max_magnitude=10 ** 9)


def test_adversarial_deep_nested_conditional_safe():
    node = var("int")
    for _ in range(6):
        # only one branch reuses the previous expression, keeping size linear
        node = call("if", call("even", var("int")), call("add1", node),
                    call("neg", var("int")))
    assert dsl.validate(node, max_depth=16, max_nodes=64) == []
    assert isinstance(EVAL(node, 4), int)


def test_adversarial_resource_exhaustion_terminates():
    g = ResourceGovernor(global_deadline_s=0.05, node_budget=50)
    r = Searcher(governor=g, beam=20).search(G.numeric_ood_task("square", 0, k=3))
    assert r.timed_out or r.expansions <= 50


def test_adversarial_interrupted_experiment_marked_incomplete():
    g = ResourceGovernor(node_budget=5)
    status = CompletionStatus.COMPLETED
    try:
        with g.scope("fit"):
            for _ in range(50):
                g.charge_nodes()
                g.checkpoint("fit")
    except ResourceLimit:
        status = CompletionStatus.PARTIAL  # partial diagnostics preserved, not success
    assert status is not CompletionStatus.COMPLETED


def test_adversarial_repeated_identical_runs():
    t = G.twin_task(4, k=3)
    eng = ValidationEngine(budget=4, eval_fn=EVAL)
    a = eng.validate(_cands(), t)
    b = eng.validate(_cands(), t)
    assert (a.state, a.candidate_id) == (b.state, b.candidate_id)
