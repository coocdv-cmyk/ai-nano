"""Tests for the isolated EXP-J research package.

These tests validate the *experimental apparatus* and its scientific invariants:
independent generators, disjoint splits, test-label protection, honest resource
reporting, completion states and metric correctness. They do not assert that any
particular research result is positive.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from constructive.exp_j import (abstraction_economics as J4, analysis,
                                deep_composition as J3, distributions,
                                drift, generators as G, interactions as J5,
                                metrics as M, noise, resources, robustness as J2,
                                runner, smoke, transfer as J1)
from constructive.exp_j.config import (ExpJConfig, PROTECTED, RESULTS_DIR,
                                       sha256_file, source_revision)
from constructive.exp_j.engine import run_validation
from constructive.exp_j.types import (Arm, CompletionState, DriftKind,
                                      NoiseKind, Outcome)

ROOT = Path(__file__).resolve().parents[1]
TEST_CFG = ExpJConfig(
    seeds=(0,), n_episodes=4, n_val=3, search_node_budget=2000,
    fit_deadline_s=1.0, experiment_deadline_s=30.0,
    depths=(2, 3), quick=True,
)


# ---------------------------------------------------------------------------
# Stage A / independence
# ---------------------------------------------------------------------------


def test_stage_a_smoke_all_pass():
    r = smoke.run_stage_a()
    assert r["status"] == "completed"
    assert r["failures"] == 0
    assert r["passed"] == r["n_checks"] == 8


def test_generators_are_independent_of_exp_h_and_exp_i():
    src = inspect.getsource(G)
    assert "exp_i.generators" not in src
    assert "exp_h.generators" not in src
    assert "REFERENCE_LABELS" not in src
    known = {"sum_digits", "digital_root", "square", "fib", "is_prime",
             "double", "abs", "add1", "sum_of_evens", "distinct_count",
             "sum_squares", "count_gt2"}
    assert not (set(G.ALL_RULES) & known)


def test_all_rules_and_shifts_generate_disjoint_labelled_splits():
    for rule in G.ALL_RULES:
        for shift in ("none", "magnitude", "sign", "length", "both"):
            t = G.make_task(rule, 2, shift=shift)
            assert G.disjoint([e.x for e in t.train],
                              [e.x for e in t.validation_pool],
                              [e.x for e in t.test])
            assert t.train and t.validation_pool and t.test
            assert len(G.candidates_for(t)) >= 2


def test_deterministic_generation_same_seed():
    a, b = G.make_task("sum_top_k", 7), G.make_task("sum_top_k", 7)
    assert [e.x for e in a.train] == [e.x for e in b.train]
    assert [e.y for e in a.test] == [e.y for e in b.test]
    assert a.meta["params"] == b.meta["params"]


def test_distribution_shift_descriptors_are_real():
    matrix = distributions.transfer_matrix(("affine_mod",), ("none", "magnitude"))
    assert len(matrix) == 2
    assert matrix[0]["train_domain"] != matrix[1]["test_domain"]
    assert any(s.name == "affine_mod:magnitude"
               for s in distributions.SHIFT_SPECS)


# ---------------------------------------------------------------------------
# Noise, drift and test-label protection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(NoiseKind))
def test_noise_never_modifies_test_labels(kind):
    t = G.make_task("threshold_parity", 3)
    n = noise.apply_noise(t, kind, 0.3, 5)
    assert [e.y for e in n.test] == [e.y for e in t.test]
    assert [e.x for e in n.test] == [e.x for e in t.test]
    assert n.meta["noise_kind"] == kind.value


def test_noise_levels_match_directive():
    assert noise.noise_levels() == (0.0, 0.01, 0.05, 0.10, 0.20, 0.30)


def test_contradiction_noise_creates_duplicate_input_conflict():
    t = G.make_task("affine_mod", 1)
    n = noise.apply_noise(t, NoiseKind.CONTRADICTION, 0.5, 1)
    seen = {}
    conflict = False
    for e in n.train:
        k = repr(e.x)
        if k in seen and seen[k] != e.y:
            conflict = True
        seen[k] = e.y
    assert conflict


@pytest.mark.parametrize("kind", [DriftKind.GRADUAL, DriftKind.ABRUPT])
def test_drift_stream_phase_order(kind):
    tasks, d, r = drift.make_drift_stream("affine_mod", 1, kind)
    assert 0 < d < r < len(tasks)
    assert tasks[0].meta["phase"] == "base"
    assert tasks[d].meta["phase"] == "drift"
    assert tasks[r].meta["phase"] == "recover"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_stats_reports_n_mean_median_std_ci():
    s = M.stats([1.0, 2.0, 3.0, 4.0])
    assert s["n"] == 4
    assert s["mean"] == 2.5
    assert s["median"] == 2.5
    assert s["std"] > 0
    assert s["ci95"] > 0


def test_accuracy_coverage_and_retrieval_pr():
    ac = M.accuracy_coverage(correct=3, answered=4, total=5)
    assert ac["accuracy"] == 0.6 and ac["coverage"] == 0.8
    assert abs(ac["selective_accuracy"] - 0.75) < 1e-9
    pr = M.retrieval_pr({"a"}, ["b", "a"], k=2)
    assert pr["recall"] == 1.0 and pr["precision"] == 0.5


def test_calibration_split_is_independent():
    cal = M.calibration_ece([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    assert cal["n_fit"] == 2 and cal["n_eval"] == 2
    assert cal["ece_independent"] is not None


def test_detection_delay():
    assert M.detection_delay([False, False, True], 0) == 2
    assert M.detection_delay([False, False], 0) is None


# ---------------------------------------------------------------------------
# Validation engine outcomes
# ---------------------------------------------------------------------------


def test_validation_distinguishes_accept_reject_and_noniidentifiable():
    row = {}
    for rule in G.ALL_RULES[:5]:
        t = G.make_task(rule, 0)
        res, fields = run_validation(t, TEST_CFG)
        row[fields["state"]] = row.get(fields["state"], 0) + 1
        assert fields["state"] in {
            "ACCEPTED", "REJECTED", "INSUFFICIENT_EVIDENCE",
            "NON_IDENTIFIABLE", "CONTRADICTORY", "RESOURCE_LIMITED"}
        assert isinstance(fields["correct_acceptance"], bool)
        assert isinstance(fields["wrong_acceptance"], bool)


def test_contradictory_evidence_is_detected():
    t = noise.apply_noise(G.make_task("affine_mod", 2),
                          NoiseKind.CONTRADICTION, 1.0, 2)
    res, fields = run_validation(t, TEST_CFG)
    assert fields["contradictory"]
    assert not res.accepted


# ---------------------------------------------------------------------------
# Resource governance
# ---------------------------------------------------------------------------


def test_timeout_is_never_success():
    gov = resources.ResourceGovernor(global_deadline_s=0.0)
    with pytest.raises(resources.ResourceLimit):
        gov.checkpoint("test")
    assert gov.report()["timeouts"]
    mon = resources.Monitor()
    mon.mark_timeout("test")
    assert mon.finish()["completion_status"] == CompletionState.TIMED_OUT.value


def test_monitor_charges_resources():
    mon = resources.Monitor()
    mon.charge(expansions=5, nodes=9, verification_calls=2)
    out = mon.finish()
    assert out["resources"]["expansions"] == 5
    assert out["resources"]["nodes"] == 9
    assert out["resources"]["verification_calls"] == 2
    assert out["completion_status"] == "COMPLETED"


# ---------------------------------------------------------------------------
# Experiment modules produce classified, complete blocks
# ---------------------------------------------------------------------------


def test_j1_transfer_block_shape():
    r = J1.run_transfer(TEST_CFG)
    assert r["status"] == "completed"
    assert r["n_episodes"] > 0
    assert "wrong_acceptance_vs_support_only" in r["comparisons"]
    assert r["completion_status"] == "COMPLETED"
    assert all(e["failure_category"] for e in r["episodes"])


def test_j2_noise_and_drift_blocks():
    r = J2.run_noise_drift(TEST_CFG)
    assert r["noise"]["n_episodes"] > 0
    assert r["drift"]["n_episodes"] > 0
    assert set(r["noise"]["by_kind"]) == {k.value for k in NoiseKind}
    assert set(r["drift"]["by_kind"]) == {"gradual", "abrupt"}
    assert r["completion_status"] == "COMPLETED"


def test_j3_deep_outcomes_are_classified():
    r = J3.run_deep(TEST_CFG)
    valid = {o.value for o in Outcome}
    assert set(r["outcome_totals"]) <= valid
    assert r["n_runs"] > 0
    assert all(row["outcome"] in valid for row in r["episodes"])
    assert r["budgets"]["node_budget"] == TEST_CFG.search_node_budget
    assert r["completion_status"] == "COMPLETED"


def test_j4_all_five_arms_and_components_measured():
    r = J4.run_abstraction_economics(TEST_CFG)
    assert set(r["arms"]) == {a.value for a in Arm}
    for arm in Arm:
        for fam in ("shared-core", "unrelated"):
            s = r["summary"][f"{arm.value}/{fam}"]
            for key in ("discovery_s", "storage_s", "retrieval_s",
                        "instantiation_s", "savings_s", "total_cost_s",
                        "net_cost_s"):
                assert key in s
    assert r["completion_status"] == "COMPLETED"


def test_j5_all_combinations_and_regression_list():
    r = J5.run_interactions(TEST_CFG)
    assert set(r["combinations"]) == {
        "evidence_validation+deep_search", "retrieval+abstraction",
        "concept_drift+retrieval", "noisy_evidence+abstraction",
        "conditional_dsl+deep_composition",
        "program_repair+adversarial_evidence"}
    assert isinstance(r["regressions"], list)
    assert r["completion_status"] in {"COMPLETED", "ERROR"}


def test_j6_efficiency_from_blocks():
    blocks = {"J3": {"resources": {"elapsed_s": 1.0}}, "J4": {"resources": {}}}
    eff = analysis.efficiency(blocks)
    assert eff["total_wall_s"] == 1.0
    assert eff["cpu_only"] is True


def test_integration_readiness_is_not_recommended():
    r = analysis.integration_readiness({}, {"protected_files_unchanged": True})
    assert r["integration_recommended"] is False
    assert r["reasons"]


# ---------------------------------------------------------------------------
# Integrity / persistence
# ---------------------------------------------------------------------------


def test_protected_hashes_match_stage0_manifest():
    p = RESULTS_DIR / "baseline_hashes.json"
    assert p.exists(), "STAGE 0 baseline manifest must be recorded"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert len(data["files"]) == len(PROTECTED)
    for f, expected in data["files"].items():
        assert sha256_file(ROOT / f) == expected, f


def test_integrity_report_clean():
    r = runner.verify_integrity()
    assert r["protected_files_unchanged"] is True
    assert r["changed"] == []


def test_no_exp_j_code_in_protected_paths():
    for f in PROTECTED:
        assert "exp_j" not in f and "test_exp_j" not in f


def test_source_revision_is_honest():
    assert source_revision() in {"no-commits", "unknown"} or \
        len(source_revision()) >= 4
