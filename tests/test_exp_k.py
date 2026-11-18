"""EXP-K apparatus and stage tests.

Tests validate the experimental apparatus, scientific invariants and stage
artifacts. They never assert that a particular research result is positive.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from constructive.exp_k import (baseline, constructors as C, distributions,
                                generators as G, metrics as M, runner, smoke)
from constructive.exp_k.config import ExpKConfig, RESULTS_DIR, quick_config
from constructive.exp_k.engine import run_validation

ROOT = Path(__file__).resolve().parents[1]
CFG = quick_config()


# --- K0: generators, splits, reproducibility -------------------------------


def test_stage_a_smoke():
    r = smoke.run_stage_a()
    assert r["status"] == "completed" and r["failures"] == 0
    assert r["passed"] == r["n_checks"] == 9


def test_generators_independent_of_prior_campaigns():
    src = inspect.getsource(G)
    assert all(x not in src for x in ("exp_j.generators", "exp_i.generators",
                                      "exp_h.generators"))
    known = {"affine_mod", "sum_top_k", "weighted_pair", "sum_digits", "fib"}
    assert not (set(G.ALL_RULES) & known)


@pytest.mark.parametrize("rule", G.ALL_RULES)
def test_generator_labels_and_disjoint_splits(rule):
    for shift in ("none", "magnitude", "sign", "length", "both"):
        t = G.make_task(rule, 2, shift=shift)
        assert G.disjoint([e.x for e in t.train],
                          [e.x for e in t.validation_pool],
                          [e.x for e in t.test])
        prog = dict((c.id, c.program) for c in t.meta["candidates"])["true"]
        assert all(__import__("constructive.exp_i.dsl", fromlist=["evaluate"]).evaluate(
            prog, e.x) == e.y for e in t.train + t.validation_pool + t.test)


def test_deep_chain_trace_and_depth():
    for d in (2, 3, 5, 12):
        t, node = C.deep_chain_task(4, d)
        assert node.depth() == d + 1  # var leaf is depth 1
        assert [s["step"] for s in t.meta["steps"]] == list(range(1, d + 1))


@pytest.mark.parametrize("kind", ["gradual", "abrupt"])
def test_drift_phase_integrity(kind):
    tasks, d, r = C.drift_stream("mod_double", 1, kind)
    assert 0 < d < r < len(tasks)
    assert tasks[0].meta["phase"] == "base"
    assert tasks[d].meta["phase"] == "drift"
    assert tasks[r].meta["phase"] == "recover"


def test_metrics_correctness():
    s = M.stats([1.0, 2.0, 3.0, 4.0])
    assert s["mean"] == 2.5 and s["median"] == 2.5 and s["std"] > 0
    assert M.ece([1.0, 0.0], [1, 0]) == 0.0
    assert M.ece([0.8, 0.2], [1, 0]) == 0.2
    assert M.brier([1.0, 0.0], [1, 0]) == 0.0
    assert M.nll([1.0, 0.0], [1, 0]) is not None


def test_distribution_matrix():
    m = distributions.shift_matrix(("mod_double",), ("none", "magnitude"))
    assert len(m) == 2
    assert m[0]["train_domain"] != m[1]["test_domain"]


def test_baseline_manifest_splits_clean():
    m = baseline.dataset_manifest(quick_config().as_dict(), (0, 1))
    assert m["all_splits_disjoint"]
    assert m["n_datasets"] == len(G.ALL_RULES) * 2 * len(distributions.SHIFTS)


def test_integrity_snapshot_clean():
    integ = baseline.integrity_snapshot()
    assert all(v["all_unchanged"] for v in integ["manifests"].values())
    assert sum(v["n_files"] for v in integ["manifests"].values()) == 106


def test_validation_produces_distinct_states():
    t = G.make_task("gt_mod", 0)
    _r, fields = run_validation(t, CFG)
    assert fields["state"] in {"ACCEPTED", "REJECTED", "INSUFFICIENT_EVIDENCE",
                               "NON_IDENTIFIABLE", "CONTRADICTORY",
                               "RESOURCE_LIMITED"}
    assert isinstance(fields["correct"], bool)
    assert isinstance(fields["wrong"], bool)


def test_k0_baseline_block_shape_and_completion():
    r = runner.run(stage="K0", quick=True)
    k0 = r["experiments"]["K0"]
    assert k0["status"] == "completed"
    assert k0["completion_status"] == "COMPLETED"
    assert set(k0["baselines"]) == {"deep", "retrieval", "drift",
                                    "calibration", "misleading"}
    assert k0["baselines"]["deep"]["n"] > 0
    assert k0["baselines"]["retrieval"]["n"] > 0
    assert k0["baselines"]["drift"]["n"] > 0
    assert k0["baselines"]["calibration"]["n"] > 0
    assert k0["baselines"]["misleading"]["n"] > 0
    assert r["protected_files_unchanged"] is True


def test_k0_results_persisted():
    latest = RESULTS_DIR / "latest.json"
    assert latest.exists()
    data = json.loads(latest.read_text(encoding="utf-8"))
    assert "K0" in data["experiments"]
    assert (RESULTS_DIR / "manifests" / "k0_dataset_manifest.json").exists()
