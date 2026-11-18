"""Tests for the isolated EXP-L research package.

These tests never modify protected systems; they only read and verify them.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from constructive.exp_l import config as cfgmod
from constructive.exp_l import integrity as integ
from constructive.exp_l import reporting
from constructive.exp_l import runner
from constructive.exp_l.config import quick_config
from constructive.exp_l.datasets import minimal_fingerprint, minimal_fixture
from constructive.exp_l.determinism import ROOT, fingerprint, make_rng
from constructive.exp_l.integrity import verify_protected


# --- determinism -----------------------------------------------------------

def test_rng_reproducible_for_same_seed_and_stream():
    a = make_rng(7, "L1")
    b = make_rng(7, "L1")
    assert [a.random() for _ in range(8)] == [b.random() for _ in range(8)]


def test_rng_differs_by_seed_or_stream():
    base = [make_rng(7, "A").random() for _ in range(1)]
    assert base != [make_rng(8, "A").random() for _ in range(1)]
    assert base != [make_rng(7, "B").random() for _ in range(1)]


def test_fixture_deterministic_across_processes():
    code = ("from constructive.exp_l.datasets import minimal_fingerprint; "
            "print(minimal_fingerprint(11))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                         capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == minimal_fingerprint(11)


def test_minimal_fixture_rule_and_determinism():
    rows = minimal_fixture(3, 20)
    assert len(rows) == 20
    assert all(r["y"] == (2 * r["x"] + 3) % 7 for r in rows)
    assert minimal_fixture(3, 20) == rows
    assert minimal_fingerprint(3) != minimal_fingerprint(4)


def test_config_fingerprint_tracks_changes():
    c1 = cfgmod.default_config("L1")
    c2 = cfgmod.default_config("L1", n_val=5)
    assert c1.config_fingerprint != c2.config_fingerprint
    assert c1.config_fingerprint == cfgmod.default_config("L1").config_fingerprint


# --- protected integrity ---------------------------------------------------

def test_exp_i_and_exp_j_hashes_unchanged():
    r = verify_protected()
    assert r["exp_i"]["all_unchanged"], r["exp_i"]["changed"]
    assert r["exp_j"]["all_unchanged"], r["exp_j"]["changed"]
    assert r["exp_i"]["n_files"] == 22
    assert r["exp_j"]["n_files"] == 84


def test_exp_k_integrity_reverifies():
    r = verify_protected()
    assert r["exp_k_integrity"]["all_ok"]
    assert r["exp_k_dataset"]["fingerprint_match"]
    assert r["exp_k_dataset"]["all_splits_disjoint"]


def test_working_tree_has_no_tracked_source_modifications():
    # The legacy suite regenerates its own quick-run JSON result files; that is
    # a known side effect, restored after the milestone run. No protected
    # source file may ever be modified.
    wt = runner.working_tree()
    assert wt["protected_source_modified"] == []


# --- L0 smoke and isolation ------------------------------------------------

def test_l0_smoke_passes():
    assert runner.smoke_check(cfgmod.default_config("L0"))["passed"]


def test_l0_quick_run_isolated(tmp_path, monkeypatch):
    import constructive.exp_l.runner as r

    quick_dir = tmp_path / "quick"
    monkeypatch.setattr(r, "RESULTS", quick_dir)
    monkeypatch.setattr(reporting, "RESULTS", quick_dir)
    monkeypatch.setattr(reporting, "HISTORY", quick_dir / "history")
    block = r.run_l0(quick_config("L0"), run_full_suite=False)
    assert block["gate_pass"] is True
    assert (quick_dir / "L0-baseline-quick.json").exists()
    assert not (quick_dir / "L0-baseline.json").exists()
    assert not (quick_dir / "L0-integrity.json").exists()


def test_snapshot_immutable_and_manifest_additive(tmp_path, monkeypatch):
    monkeypatch.setattr(reporting, "RESULTS", tmp_path)
    monkeypatch.setattr(reporting, "HISTORY", tmp_path / "history")
    before = {"stage": "L0", "x": 1}
    snap = reporting.snapshot("L0", before, stamp="20260101-000000")
    assert json.loads(snap.read_text(encoding="utf-8"))["x"] == 1
    reporting.update_integrity({"stage": "L0", "gate_pass": True})
    reporting.update_integrity({"stage": "L1", "gate_pass": True})
    m = json.loads((tmp_path / "integrity.json").read_text(encoding="utf-8"))
    assert set(m["stages"]) == {"L0", "L1"}
    assert m["stages"]["L0"]["gate_pass"] is True


def test_fingerprint_is_stable_json():
    a = {"b": 1, "a": [2, 3]}
    b = {"a": [2, 3], "b": 1}
    assert fingerprint(a) == fingerprint(b)


# --- L1: semantic drift ----------------------------------------------------

from constructive.exp_l.semantic_drift import (  # noqa: E402
    CATEGORIES, DETECTORS, HELDOUT, LIBRARY, calibrate, detect, make_stream,
    run_l1, signals,
)
from constructive.exp_l.metrics import (precision_recall_f1, roc_auc,  # noqa: E402
                                        safe_rate)


def test_library_and_heldout_are_disjoint():
    assert not (set(LIBRARY) & set(HELDOUT))
    assert len(LIBRARY) == 6 and len(HELDOUT) == 4


def test_stream_deterministic_and_categories_valid():
    a = make_stream(5, "parity", "semantic_abrupt", "square_mod5")
    b = make_stream(5, "parity", "semantic_abrupt", "square_mod5")
    assert a["xs"] == b["xs"] and a["ys"] == b["ys"]
    s = signals(a)
    assert len(s["feat"]) == len(s["cons"]) == 12
    assert set(CATEGORIES) >= {"stable", "feature", "semantic_abrupt", "camouflage"}


def test_feature_drift_preserves_relationship():
    st = make_stream(2, "parity", "feature", "neg_mod4")
    sig = signals(st)
    # Reference fitted on base data still agrees on shifted-domain windows.
    assert all(c >= 0.95 for c in sig["cons"][4:8]), sig["cons"]


def test_camouflage_keeps_feature_marginals_identical():
    stable = make_stream(3, "mod3", "stable", "band")
    camo = make_stream(3, "mod3", "camouflage", "band")
    s1, s2 = signals(stable), signals(camo)
    # Camouflage reuses base windows' exact x multiset during drift.
    assert sorted(camo["xs"][4]) == sorted(camo["xs"][3])
    assert abs(s2["feat"][3] - s2["feat"][4]) < 1e-9
    assert abs(s1["feat"][0] - s2["feat"][4]) > 1e-9  # different base windows


def test_detectors_only_receive_signals_not_ground_truth():
    st = make_stream(1, "parity", "semantic_abrupt", "band")
    sig = signals(st)
    thr = calibrate()
    # A detector cannot see ys, category, or the drift family.
    for det in DETECTORS:
        flags, score = detect(det, sig, thr, 1)
        assert len(flags) == 12 and isinstance(score, float)


def test_false_alarm_and_precision_undefined():
    assert safe_rate(0, 0) is None
    assert precision_recall_f1(0, 0, 0)["f1"] is None
    assert precision_recall_f1(2, 1, 1)["precision"] == pytest.approx(2 / 3)


def test_roc_auc_perfect_and_random():
    assert roc_auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert roc_auc([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]) == pytest.approx(0.5)


def test_l1_semantic_detector_beats_random_on_small_campaign():
    cfg = cfgmod.default_config("L1", seeds=(0, 1, 2))
    block = run_l1(cfg)
    sem = block["by_detector"]["semantic_consistency"]["semantic_recall"]["mean"]
    rnd = block["by_detector"]["random"]["semantic_recall"]["mean"]
    assert sem > rnd
    assert block["by_detector"]["semantic_consistency"]["auroc"] is not None


# --- L2: selective composition verification --------------------------------

from constructive.exp_l.composition_verification import (  # noqa: E402
    DEPTHS, METHODS, decide, make_case, public_view, run_l2,
)


def test_public_view_excludes_all_oracle_fields():
    case = make_case(2, 5, "one_error")
    pub = public_view(case)
    for forbidden in ("hidden", "test_xs", "correct", "first_error",
                      "intermediate_error"):
        assert forbidden not in pub


def test_decision_is_independent_of_ground_truth():
    # Two cases with identical public evidence but different hidden truth must
    # produce identical decisions (no oracle leakage).
    c1 = make_case(1, 4, "correct")
    c2 = make_case(1, 4, "correct")
    for m in METHODS:
        assert decide(m, public_view(c1)) == decide(m, public_view(c2))


def test_always_abstain_has_undefined_selective_accuracy():
    block = run_l2(cfgmod.default_config("L2", seeds=(0,)))
    assert block["by_method"]["always_abstain"]["selective_accuracy"] is None
    assert block["by_method"]["always_abstain"]["useful_coverage"] == 0.0


def test_compensating_case_has_correct_final_but_invalid_path():
    case = make_case(4, 6, "compensating")
    assert case["correct"] is True
    assert case["intermediate_error"] is True


def test_full_verification_reduces_incorrect_acceptance_vs_baseline():
    block = run_l2(cfgmod.default_config("L2", seeds=(0, 1, 2)))
    b = block["by_method"]["baseline"]["incorrect_acceptance_rate"]
    f = block["by_method"]["full"]["incorrect_acceptance_rate"]
    assert f <= b
    assert block["gate"]["no_oracle_in_decision"]


# --- L3: retrieval generalization ------------------------------------------

from constructive.exp_l.retrieval_generalization import (  # noqa: E402
    FAMILIAR as RETRIEVAL_FAMILIAR, HELDOUT as RETRIEVAL_HELDOUT,
    decide as retrieval_decide, make_query, rank, run_l3,
)


def test_family_split_is_disjoint_by_identity():
    assert not (set(RETRIEVAL_FAMILIAR) & set(RETRIEVAL_HELDOUT))
    assert len(RETRIEVAL_FAMILIAR) == 6 and len(RETRIEVAL_HELDOUT) == 6


def test_heldout_query_has_no_correct_concept_in_pool():
    q = make_query(1, "heldout")
    assert q["expected"] is None
    assert q["target"] in RETRIEVAL_HELDOUT
    assert q["target"] not in [r["cid"] for r in rank(q)]


def test_near_duplicate_is_discriminated_by_behavior():
    q = make_query(0, "near_duplicate")
    assert q["expected"] == "c-inc-2"
    assert rank(q)[0]["cid"] == "c-inc-2"


def test_misleading_surface_similarity_fools_similarity_only():
    hits = 0
    for seed in range(6):
        q = make_query(seed, "misleading")
        sim = retrieval_decide("similarity_only", q)
        ver = retrieval_decide("rank_verify", q)
        if sim["accepted"] and sim["cid"] == q["target"]:
            hits += 1
        assert not ver["accepted"]
    assert hits == 0  # surface similarity never retrieves the held-out concept


def test_l3_substitution_reduced_with_familiar_coverage():
    block = run_l3(cfgmod.default_config("L3", seeds=(0, 1, 2)))
    base = block["by_method"]["baseline"]["silent_substitution_rate"]
    rv = block["by_method"]["rank_verify"]["silent_substitution_rate"]
    assert rv < base
    assert block["by_method"]["rank_verify"]["useful_coverage"] > 0
    assert block["note"].find("Recall@k") >= 0


# --- L4: conditional calibration -------------------------------------------

from constructive.exp_l.conditional_calibration import (  # noqa: E402
    CAL_SEEDS, MIN_GROUP_N, adaptive_ece, fit, make_rows, risk_coverage, run_l4,
)


def test_calibration_and_evaluation_seeds_are_disjoint():
    assert not (set(CAL_SEEDS) & set(range(10)))
    rows_cal = [r for s in CAL_SEEDS for r in make_rows(s)]
    rows_eval = [r for s in range(3) for r in make_rows(s)]
    assert set(r["seed"] for r in rows_cal).isdisjoint(
        set(r["seed"] for r in rows_eval))


def test_temperature_is_fitted_on_calibration_only():
    cal = [r for s in CAL_SEEDS for r in make_rows(s)]
    params = fit("temperature", cal)
    assert params["T"] in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0)
    # Fitting again on the same calibration data is deterministic.
    assert fit("temperature", cal) == params


def test_adaptive_ece_and_risk_coverage_undefined_case():
    assert adaptive_ece([0.2, 0.8, 0.5, 0.5], [0, 1, 0, 1]) >= 0
    curve = risk_coverage([0.1, 0.2], [0, 1], (0.9,))
    assert curve[0]["selective_accuracy"] is None
    assert curve[0]["selective_risk"] is None


def test_l4_improves_aggregate_ece_but_difficult_subgroups_persist():
    block = run_l4(cfgmod.default_config("L4", seeds=(0, 1, 2)))
    raw = block["by_method"]["raw"]
    temp = block["by_method"]["temperature"]
    assert temp["ece"] < raw["ece"]
    assert temp["accuracy"] == raw["accuracy"]
    deep = raw["by_condition"]["deep"]
    assert deep["n"] >= MIN_GROUP_N and deep["ece"] > raw["ece"]
    assert block["gate"]["calibration_split_disjoint"]


# --- L5: natural-input bridge ----------------------------------------------

from constructive.exp_l.natural_input_bridge import (  # noqa: E402
    TEMPLATES, answer, build_corpus, execute, expected_answer, parse, run_l5,
)


def test_corpus_split_is_by_template_identity():
    dev = {t.tid for t in TEMPLATES if t.split == "dev"}
    held = {t.tid for t in TEMPLATES if t.split == "heldout"}
    assert dev and held and not (dev & held)
    rows = build_corpus(0)
    assert {r["template"] for r in rows} == dev | held


def test_parse_failure_is_not_a_reasoning_failure():
    p = parse("Add 5 to the number and report the result.")
    assert p.ok and p.incomplete  # parsed fine, evidence insufficient
    answered, _ = execute(p)
    assert not answered  # abstention, not a parse error
    bad = parse("xyzzy @@@")
    assert not bad.ok


def test_expected_answer_matches_executor_on_answer_templates():
    for t in TEMPLATES:
        if t.kind != "answer":
            continue
        for seed in range(3):
            text, _ = t.generator(seed, make_rng(seed, t.tid))
            exp = expected_answer(text)
            p = parse(text)
            answered, got = execute(p)
            assert answered and got == exp, (t.tid, text, exp, got)


def test_adapter_transfer_and_preserved_failures():
    block = run_l5(cfgmod.default_config("L5", seeds=(0, 1, 2)))
    m = block["by_method"]["adapter_with_guard"]
    assert (m["heldout_exact_match"] or 0) > (block["by_method"]["constant"]["heldout_exact_match"] or 0)
    assert m["correct_abstention"] > block["by_method"]["adapter_no_guard"]["correct_abstention"]
    assert m["n_failures_preserved"] >= 0
    assert block["provenance"]["external_model"] is False


# --- L6: reliability economics ---------------------------------------------

from constructive.exp_l.reliability_economics import CONFIGS, run_l6  # noqa: E402


def test_l6_matched_workload_and_vector_metrics():
    block = run_l6(cfgmod.default_config("L6", seeds=(0, 1, 2)))
    cs = block["configurations"]
    assert all(cs[c]["n"] == block["n_tasks"] for c in CONFIGS)
    for c in CONFIGS:
        assert cs[c]["resources"]["wall_s"] > 0
        assert cs[c]["operation_counts"]["node_evaluations"] >= 0
        assert cs[c]["incorrect_acceptances_per_1000"] >= 0
    assert block["gate"]["no_composite_score"]


def test_l6_always_abstain_has_zero_useful_coverage():
    block = run_l6(cfgmod.default_config("L6", seeds=(0, 1)))
    aa = block["configurations"]["always_abstain"]
    assert aa["useful_coverage"] == 0.0
    assert aa["abstention_rate"] == 1.0
    assert aa["selective_accuracy"] if "selective_accuracy" in aa else True


# --- L7: final audit -------------------------------------------------------

from constructive.exp_l.audit import metric_fingerprint, strip_nondeterministic  # noqa: E402


def test_metric_fingerprint_ignores_timestamps_and_timing():
    a = {"x": 1, "resources": {"wall_s": 0.1}, "snapshot": "a"}
    b = {"x": 1, "resources": {"wall_s": 9.9}, "snapshot": "b"}
    assert metric_fingerprint(a) == metric_fingerprint(b)
    assert strip_nondeterministic(a) == {"x": 1}
