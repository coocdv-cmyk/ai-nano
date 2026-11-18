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
