"""Tests for omega_nano.brain: zero-hallucination verifiers, skill compilation, blind vault, and self-play."""

import json

from omega_nano.brain import (
    BlindVault,
    CodeVerifier,
    CurriculumProposer,
    MathVerifier,
    SelfPlayEngine,
    SkillLibrary,
)
from omega_nano.brain.evolver import SymbolicTacticSolver
from omega_nano.memory import MemoryFabric


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
