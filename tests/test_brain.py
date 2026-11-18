"""Tests for omega_nano.brain: zero-hallucination verifiers, skill compilation, blind vault, and self-play."""

import json
from pathlib import Path

from omega_nano.brain import (
    BlindProblem,
    BlindVault,
    CodeVerifier,
    CurriculumProposer,
    ExecutionGuidedSynthesizer,
    MathVerifier,
    NeuralPolicyRanker,
    NeuroSymbolicSolver,
    SelfPlayEngine,
    SkillLibrary,
    SymbolicTacticSolver,
    index_repository,
)
from omega_nano.memory import MemoryFabric
from omega_nano.semantic import SemanticDriftMonitor


def test_math_verifier_proves_identity_and_refutes_with_counterexample():
    ok = MathVerifier.verify_identity("(x + y)^2", "x^2 + 2*x*y + y^2", var_names=("x", "y"))
    assert ok.passed and ok.status == "PROVEN_EXACT" and ok.confidence == 1.0

    bad = MathVerifier.verify_identity("(x + y)^2", "x^2 + y^2", var_names=("x", "y"))
    assert not bad.passed and bad.status == "REFUTED" and "counterexample" in bad.evidence

    inj = MathVerifier.verify_identity("__import__('os').system('echo x')", "0")
    assert not inj.passed and inj.status == "SYNTAX_ERROR"


def test_math_verifier_checks_equation_solution():
    res = MathVerifier.verify_equation_solution("x^2 - 13*y^2", "1", {"x": "649", "y": "180"})
    assert res.passed and res.status == "PROVEN_EXACT"

    wrong = MathVerifier.verify_equation_solution("x^2 - 13*y^2", "1", {"x": "5", "y": "1"})
    assert not wrong.passed and wrong.status == "REFUTED"


def test_code_verifier_checks_syntax_security_timeout_and_tests():
    syn = CodeVerifier.verify_python("def f(:", "assert f() == 1")
    assert not syn.passed and syn.status == "SYNTAX_ERROR"

    sec = CodeVerifier.verify_python("import socket\ndef f(): return 1", "assert f() == 1")
    assert not sec.passed and sec.status == "SECURITY_BLOCKED"

    fail = CodeVerifier.verify_python("def add(a, b): return a - b", "assert add(2, 3) == 5")
    assert not fail.passed and fail.status == "REFUTED" and "AssertionError" in fail.feedback

    ok = CodeVerifier.verify_python("def add(a, b): return a + b", "assert add(2, 3) == 5")
    assert ok.passed and ok.status == "TESTS_PASSED"

    to = CodeVerifier.verify_python("def f():\n    while True: pass", "f()", timeout_s=0.5)
    assert not to.passed and to.status == "TIMEOUT"


def test_blind_vault_never_leaks_sealed_reference_and_solves_independently(tmp_path):
    mem = MemoryFabric(tmp_path / "mem.db")
    skills = SkillLibrary(tmp_path / "brain.db", memory=mem)
    sp = SelfPlayEngine(skills)
    vault = BlindVault()
    solver = SymbolicTacticSolver()

    # Check that BlindProblem exposes no sealed reference attributes.
    for p in vault.problems():
        assert not hasattr(p, "reference_solution")

    report = vault.evaluate(lambda prob: sp.solve(prob, solver, max_attempts=3))
    assert report["total"] == 10
    # All 5 non-trivial math problems should pass with formal verification, including Pell D=13 (649, 180)
    # on attempt 2 after attempt 1 fails with counterexample!
    math_res = {r["id"]: r for r in report["results"] if r["domain"] == "math"}
    assert all(r["passed"] for r in math_res.values()), math_res
    assert math_res["math-pell-13"]["attempts"] == 2
    assert math_res["math-pell-13"]["post_hoc_reference"] == "649, 180"
    skills.close()
    mem.close()


def test_self_play_compiles_skills_and_improves_solve_speed_across_cycles(tmp_path):
    mem = MemoryFabric(tmp_path / "mem.db")
    skills = SkillLibrary(tmp_path / "brain.db", memory=mem)
    sp = SelfPlayEngine(skills)
    solver = SymbolicTacticSolver()

    c1 = sp.run_cycle(cycle=1, solver=solver, n_tasks=8, level=2, max_attempts=3)
    assert c1["solve_rate"] == 1.0
    assert c1["library_stats"]["skills"] >= 2
    assert c1["library_stats"]["replay_traces"] == 8

    # Cycle 2 hits compiled generalized skills on attempt 1 (reducing attempts vs Cycle 1).
    c2 = sp.run_cycle(cycle=2, solver=solver, n_tasks=8, level=2, max_attempts=3)
    assert c2["solve_rate"] == 1.0
    assert c2["avg_attempts"] <= c1["avg_attempts"]

    # Procedural skills were synced into MemoryFabric and replay JSONL exports cleanly.
    assert mem.recall("pell_continued_fraction", kind="procedural")
    out_jsonl = tmp_path / "replay.jsonl"
    n_exported = skills.export_replay_jsonl(out_jsonl)
    assert n_exported == 16
    first = json.loads(out_jsonl.read_text(encoding="utf-8").splitlines()[0])
    assert "verified_solution" in first and "proof_status" in first

    skills.close()
    mem.close()


def test_curriculum_proposer_creates_distinct_valid_tasks():
    tasks = CurriculumProposer(seed=9).generate(n=6, level=2)
    assert len(tasks) == 6
    assert len({t.statement for t in tasks}) >= 5


def test_neuro_symbolic_solves_blind_vault_heldout_and_ast_io(tmp_path):
    mem = MemoryFabric(tmp_path / "mem.db")
    skills = SkillLibrary(tmp_path / "brain.db", memory=mem)
    sp = SelfPlayEngine(skills)
    solver = NeuroSymbolicSolver()

    # Cold BlindVault evaluation (5 math + 5 code): must solve all 10/10
    report = BlindVault().evaluate(lambda p: sp.solve(p, solver, max_attempts=5))
    assert report["total"] == 10
    assert report["passed"] == 10, report

    # Independent held-out coding suite (10 distinct algorithmic tasks not in BlindVault)
    heldout = BlindVault.heldout().evaluate(lambda p: sp.solve(p, solver, max_attempts=5))
    assert heldout["total"] == 10
    assert heldout["passed"] == 10, heldout

    # Bottom-up I/O expression enumerator without domain keywords
    io_prob = BlindProblem(
        id="io-sum-pos", domain="code", title="IO Positive Sum", difficulty=1,
        statement="Write Python function calc_vals(arr: list[int]) -> int such that [1, -2, 3, -4] -> 4 and [-1, -5] -> 0.",
        verify=lambda code: CodeVerifier.verify_python(
            code,
            "assert calc_vals([1, -2, 3, -4]) == 4\n"
            "assert calc_vals([-1, -5]) == 0\n"
            "assert calc_vals([10, 20]) == 30\n",
        ),
    )
    io_res = sp.solve(io_prob, solver, max_attempts=4)
    assert io_res["passed"] and io_res["attempts"] == 1, io_res

    skills.close()
    mem.close()


def test_neural_policy_learns_and_persists_across_restart(tmp_path):
    db_path = tmp_path / "brain.db"
    skills1 = SkillLibrary(db_path)
    sp1 = SelfPlayEngine(skills1)
    solver1 = NeuroSymbolicSolver()

    # Problem deliberately omits division rounding keyword so policy/CEGIS disambiguates
    rpn_v1 = BlindProblem(
        id="pol-rpn-1", domain="code", title="Stack RPN V1", difficulty=1,
        statement="Write Python function run_rpn_v1(tokens: list[str]) -> int for Reverse Polish Notation (+ - * /).",
        verify=lambda code: CodeVerifier.verify_python(
            code,
            "assert run_rpn_v1(['-7', '2', '/']) == -3\n"
            "assert run_rpn_v1(['2', '1', '+', '3', '*']) == 9\n",
        ),
    )
    res1 = sp1.solve(rpn_v1, solver1, max_attempts=5)
    assert res1["passed"] and res1["attempts"] >= 2
    saved = skills1.load_policy()
    assert saved and (saved.get("bias") or saved.get("pair_w"))
    skills1.close()

    # Reopen on fresh process instance; query uses a brand-new function name so SkillLibrary regex won't hit
    skills2 = SkillLibrary(db_path)
    sp2 = SelfPlayEngine(skills2)
    solver2 = NeuroSymbolicSolver()
    rpn_v2 = BlindProblem(
        id="pol-rpn-2", domain="code", title="Stack RPN Unseen Name", difficulty=1,
        statement="Write Python function eval_postfix_unseen(tokens: list[str]) -> int for Reverse Polish Notation (+ - * /).",
        verify=lambda code: CodeVerifier.verify_python(
            code,
            "assert eval_postfix_unseen(['-7', '2', '/']) == -3\n"
            "assert eval_postfix_unseen(['4', '13', '5', '/', '+']) == 6\n",
        ),
    )
    res2 = sp2.solve(rpn_v2, solver2, max_attempts=5)
    assert res2["passed"] and res2["source"] == "search"
    assert res2["attempts"] == 1, f"Expected policy transfer to solve in 1 attempt, got {res2}"
    skills2.close()


def test_l3_guard_and_l1_drift_monitor_catch_spec_shift(tmp_path):
    skills = SkillLibrary(tmp_path / "brain.db")
    monitor = SemanticDriftMonitor(window=6, baseline_rate=0.9, threshold=1.4)
    sp = SelfPlayEngine(skills, drift_monitor=monitor)
    solver = NeuroSymbolicSolver()

    # Register stale strict-< interval skills; specifications then shift to touching intervals (<=).
    drifted_seen = False
    for idx in range(3):
        fn = f"merge_shift_{idx}"
        stale_code = (
            f"def {fn}(intervals):\n"
            "    if not intervals: return []\n"
            "    pts = sorted(intervals)\n"
            "    out = [list(pts[0])]\n"
            "    for s, e in pts[1:]:\n"
            "        if s < out[-1][1]: out[-1][1] = max(out[-1][1], e)\n"
            "        else: out.append([s, e])\n"
            "    return out\n"
        )
        skills.register_skill(f"code_{fn}", "code", rf"\b{fn}\b", {"kind": "code_template", "code": stale_code}, "PROVEN")
        shifted = BlindProblem(
            id=f"drift-{idx}", domain="code", title="Shifted Interval Spec", difficulty=1,
            statement=f"Write Python function {fn}(intervals: list[list[int]]) -> list[list[int]] merging overlapping/touching intervals.",
            verify=lambda code, F=fn: CodeVerifier.verify_python(
                code,
                f"assert {F}([[1, 3], [3, 5]]) == [[1, 5]]\n"
                f"assert {F}([[1, 2], [4, 6]]) == [[1, 2], [4, 6]]\n",
            ),
        )
        res = sp.solve(shifted, solver, max_attempts=4)
        assert res["passed"] and res["l3_rejected_skill"] is True, res
        if res["drift"]:
            drifted_seen = True

    assert drifted_seen, "SemanticDriftMonitor should detect repeated skill-verification mismatch drift"
    # Self-healing check: repaired skill for merge_shift_0 now succeeds in 1 attempt via skill recall
    healed = BlindProblem(
        id="drift-healed-0", domain="code", title="Repaired Skill Verification", difficulty=1,
        statement="Write Python function merge_shift_0(intervals: list[list[int]]) -> list[list[int]] merging overlapping/touching intervals.",
        verify=lambda code: CodeVerifier.verify_python(
            code,
            "assert merge_shift_0([[1, 3], [3, 5]]) == [[1, 5]]\n",
        ),
    )
    res_healed = sp.solve(healed, solver, max_attempts=4)
    assert res_healed["passed"] and res_healed["source"] == "skill:code_merge_shift_0" and res_healed["attempts"] == 1
    skills.close()


def test_repo_graph_indexes_symbols_callers_and_memory(tmp_path):
    mem = MemoryFabric(tmp_path / "mem.db")
    graph = index_repository(Path("omega_nano/brain"), memory=mem)
    summary = graph.summarize()
    assert summary["files_indexed"] >= 6
    assert summary["total_symbols"] >= 20
    assert graph.find_symbol("ExecutionGuidedSynthesizer")
    assert graph.find_symbol("SelfPlayEngine")
    callers = graph.callers_of("verify_python")
    assert callers, "Expected verify_python call sites inside omega_nano/brain"

    recalled = mem.recall("ExecutionGuidedSynthesizer", kind="project", k=5)
    assert recalled and any("synthesizer.py" in r.text for r in recalled)
    mem.close()


def test_diverse_curriculum_self_play_across_cycles(tmp_path):
    mem = MemoryFabric(tmp_path / "mem.db")
    skills = SkillLibrary(tmp_path / "brain.db", memory=mem)
    sp = SelfPlayEngine(skills)
    solver = NeuroSymbolicSolver()

    c1 = sp.run_cycle(cycle=1, solver=solver, n_tasks=12, level=2, max_attempts=4)
    c2 = sp.run_cycle(cycle=2, solver=solver, n_tasks=12, level=2, max_attempts=4)
    assert c1["solve_rate"] == 1.0, c1
    assert c2["solve_rate"] == 1.0, c2
    assert c2["avg_attempts"] <= c1["avg_attempts"]
    skills.close()
    mem.close()
