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
    assert [p for p in wt["tracked_modified"] if p.endswith(".py")] == []


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
