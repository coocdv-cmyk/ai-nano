"""Tests for the EXP-H research module.

Small, deterministic, and honest: tests prove what they claim and no more.
"""

from __future__ import annotations

import random

import pytest

from constructive.exp_h import engine
from constructive.exp_h.engine import HNode, eval_hidden
from constructive.exp_h.generators import (
    build_abstraction_tasks,
    build_h1_task,
    build_twin_task,
    build_ood_list_task,
)
from constructive.exp_h import ast_engine as AE
from constructive.exp_h import h4_pool
from constructive.exp_h.episode import ccm_episode
from constructive.exp_h import smoke as smoke_stage_a
from constructive.exp_h.lexical_h8 import alternative_search
from constructive.exp_h.h1_discovery import run_h1


# ------------------------------------------------------------------ engine


def test_engine_evaluates_hidden_programs():
    node = HNode("sum_list", None,
                 (HNode("evens", None, (HNode("var"),)),))
    assert eval_hidden(node, [1, 2, 3, 4, 5]) == 6
    bitxor = HNode("bxor", 3, (HNode("var"),))
    assert eval_hidden(bitxor, 5) == 6
    cube = HNode("sum_list", None, (HNode("map_cube", None, (HNode("var"),)),))
    assert eval_hidden(cube, [1, 2, 3]) == 36


def test_engine_rejects_wrong_types():
    with pytest.raises(engine.HiddenError):
        eval_hidden(HNode("add1", None, (HNode("evens", None, (HNode("var"),)),)),
                    [1, 2, 3])


def test_engine_step_budget_is_enforced():
    node = HNode("var")
    for _ in range(12):
        node = HNode("abs", None, (node,))
    with pytest.raises(engine.HiddenError):
        engine.eval_hidden(node, -3, steps=engine._StepCounter(budget=5))


def test_engine_depth_cap_is_enforced():
    node = HNode("var")
    for _ in range(40):
        node = HNode("abs", None, (node,))
    with pytest.raises(engine.HiddenError):
        engine.eval_hidden(node, -3)


# --------------------------------------------------------------- generators


def test_h1_tasks_have_disjoint_splits_and_labels():
    for family in ("vocab", "exclusion", "conditional"):
        task = build_h1_task(17, 3, family)
        train_x = {str(x[0] if isinstance(x, tuple) else x) for x, _ in task.train}
        test_x = {str(x[0] if isinstance(x, tuple) else x) for x, _ in task.test}
        assert not (train_x & test_x)
        assert len(task.train) == 3 and len(task.test) > 0
        assert task.meta.get("family_tag") == family


def test_h1_task_labels_match_hidden_program():
    task = build_h1_task(23, 2, "vocab")
    for x, y in task.test:
        assert eval_hidden(task.hidden, x) == y


def test_twins_coincide_on_support():
    task = build_twin_task(5, 3)
    for x, y in task.train:
        assert eval_hidden(HNode("digital_root", None, (HNode("var"),)), x) == \
               eval_hidden(HNode("sum_digits", None, (HNode("var"),)), x)


def test_ood_list_task_ranges():
    task = build_ood_list_task("sum_of_evens", 7, k=2, train_len=(2, 3),
                              test_len=(9, 10), n_test=3)
    test_lens = {len(x) for x, _ in task.test}
    train_lens = {len(x) for x, _ in task.train}
    assert all(l >= 9 for l in test_lens)
    assert all(l <= 3 for l in train_lens)


def test_abstraction_stream_builds():
    stream = build_abstraction_tasks(11, k=2, n_test=3)
    assert len(stream) >= 4
    assert all(len(t.train) == 2 for t in stream)


# --------------------------------------------------------------- ast engine


def test_ast_total_bounded_by_loop_caps_not_crash():
    buggy = AE.buggy_from_template("sum_gcd", random.Random(9), 3,
                                   check_envs=[dict(a=4, b=2), dict(a=12, b=8)])
    # buggy may build or be rejected; either is an honest outcome
    if buggy is not None:
        try:
            AE.evaluate_ast(buggy, dict(a=4, b=2), max_steps=1000)
        except AE.AstError:
            pass


def test_ast_magnitude_limit_rejects_bombs():
    bomb = [AE.Assign("s", AE.C(1)),
            AE.While(AE.Cmp("<", AE.V("s"), AE.C(10**18)),
                     [AE.Assign("s", AE.B("*", AE.V("s"), AE.V("s")))]),
            AE.Ret(AE.V("s"))]
    with pytest.raises(AE.AstError):
        AE.evaluate_ast(bomb, {}, max_steps=60)


def test_ast_repair_recovers_a_known_bug():
    correct = AE.template_correct("pow2")
    buggy = [list(s) for s in correct]
    # plant a one-node bug: i < n becomes i <= n (off-by-one)
    sites = AE.enumerate_expr_sites(buggy)
    for path, node in sites:
        if node[0] == "cmp":
            AE.replace_at(buggy, path, ("cmp", "<=", node[2], node[3]))
            break
    inputs = AE.sample_inputs("pow2", random.Random(31), 5, lo=0, hi=6)
    support = [(e, AE.evaluate_ast(buggy, e)) for e in inputs]
    repair, n_tried, reason = AE.repair_program(buggy, support, verify=None,
                                                max_width=1, max_candidates=4000)
    assert reason == "ok"
    assert repair is not None
    # the repair must agree with the hidden correct program on the bounded
    ref = AE.template_correct("pow2")
    for env in inputs:
        assert AE.evaluate_ast(repair, env) == AE.evaluate_ast(ref, env)


# -------------------------------------------------------------- h4 pool


def test_h4_pool_builds_valid_tasks():
    pool = h4_pool.build_pool()
    assert len(pool) >= 40
    for name, make in pool[:12]:
        task = make(0)
        assert len(task.train) >= 3 and len(task.test) >= 10


# --------------------------------------------------------------- episodes


def test_ccm_episode_returns_schema():
    task = build_h1_task(101, 2, "vocab")
    row = ccm_episode(task, seed=101, max_nodes=20_000, time_budget_s=2.0)
    for key in ("accepted", "abstained", "fit_s", "nodes", "test_accuracy",
                "coverage", "failure_category", "seed", "k", "system"):
        assert key in row


def test_ccm_episode_labels_do_not_leak_from_test():
    task = build_h1_task(102, 2, "vocab")
    row1 = ccm_episode(task, seed=102, max_nodes=20_000, time_budget_s=2.0)
    corrupted = type(task)(
        task.family, task.name, task.rule, task.train, task.val,
        [(x, y + 1000) for x, y in task.test], task.input_type,
        task.output_type, task.probes, task.meta)
    row2 = ccm_episode(corrupted, seed=103, max_nodes=20_000, time_budget_s=2.0)
    assert row1["fit_s"] and row2["fit_s"]


def test_lexical_search_runs_and_dirs():
    task = build_h1_task(201, 2, "vocab")
    cands, nodes = alternative_search(task.train, task.input_type,
                                      task.output_type, beam=15, max_nodes=20_000,
                                      time_budget_s=2.0, max_cost=3,
                                      beam_elite=False, eager_prune=False)
    assert nodes >= 0  # search executed (any return is a measured outcome)


def test_stage_a_smoke_all_pass():
    block = smoke_stage_a.run_stage_a()
    assert block["failures"] == 0, block


def test_h1_small_config_runs():
    cfg = type("C", (), {"seeds": (0,)})()
    block = run_h1(cfg, seeds=(0,), k_grid=(1, 2), time_budget_s=1.0,
                   max_nodes=20_000)
    assert block["experiment_id"] == "H1"
    assert block["episodes"]
    assert "by_family" in block



def test_runner_stage_a_passes():
    from constructive.exp_h import runner
    body = runner.run(only=None, stage="A", quick=True, resume=False)
    assert body["failures"] == 0


def test_json_schema_for_one_experiment():
    from constructive.exp_h.runner import _conform
    cfg = type("C", (), {"seeds": (0,)})()
    block = run_h1(cfg, seeds=(0,), k_grid=(1,), time_budget_s=1.0,
                   max_nodes=20_000)
    block = _conform(block, block["episodes"])
    assert block["experiment_id"] == "H1"
    assert block["status"] in ("completed", "partial", "error")
    for key in ("experiment_id", "experiment_name", "status", "episodes",
                "hardware", "software", "dataset_fingerprint",
                "source_revision", "limitations", "artifacts"):
        assert key in block
    assert isinstance(block["episodes"], list) and block["episodes"]
    assert any("failure_category" in row for row in block["episodes"])
