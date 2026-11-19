"""Tests for the EXP-N N1 execution-verified AST repair engine.

The engine is a capability, not an instrument, so these tests check three things:

* the parser/operator primitives behave on synthetic input;
* a fresh workspace (not one of the frozen N0 tasks) can be repaired, which is a
  small transfer probe rather than a memorised fix;
* the engine refuses a workspace no operator can justify, without writing a file.
"""

import json
from pathlib import Path

from omega_nano.brain.repair import (
    OPERATORS,
    RepairEngine,
    parse_pytest_output,
    run_pytest,
)

EXP_DIR = Path(__file__).resolve().parents[1] / "EXPERIMENTS" / "EXP-010-n1-ast-repair"
EXP011_DIR = Path(__file__).resolve().parents[1] / "EXPERIMENTS" / "EXP-011-n1-heldout"


def _write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_operator_registry_is_frozen():
    assert OPERATORS == (
        "mutable_default",
        "exception_scope",
        "cache_key",
        "closure_binding",
        "boundary_guard",
        "off_by_one",
        "integer_distribution",
        "mapping_fallback",
        "comparison_flip",
    )


def test_parse_pytest_output_extracts_failures_and_counts():
    stdout = """
FFF.                                                                     [100%]
=================================== FAILURES ===================================
_________________________________ test_second _________________________________

    def test_second():
>       assert page(2) == 2
E       assert 1 == 2

tests/test_page.py:7: AssertionError
=========================== short test summary info ===========================
FAILED tests/test_page.py::test_second - assert 1 == 2
1 failed, 3 passed in 0.05s
"""
    passed, failed, failures, summary = parse_pytest_output(stdout, "", 1)
    assert passed is False
    assert failed == 1
    assert len(failures) == 1
    assert failures[0].file == "tests/test_page.py"
    assert failures[0].function == "test_second"
    assert "1 failed, 3 passed" in summary

    passed, failed, failures, _ = parse_pytest_output("4 passed in 0.02s", "", 0)
    assert passed is True and failed == 0 and failures == ()


def test_engine_repairs_a_fresh_remainder_workspace(tmp_path):
    # Deliberately different names and domain from every frozen N0 task.
    _write(tmp_path, "allocator.py", (
        '"""Split a budget across workers."""\n\n\n'
        "def allocate(total, workers):\n"
        "    share = total // workers\n"
        "    return [share] * workers\n"
    ))
    _write(tmp_path, "tests/test_allocator.py", (
        "from allocator import allocate\n\n\n"
        "def test_sum_is_preserved():\n"
        "    assert sum(allocate(100, 3)) == 100\n"
        "    assert sum(allocate(11, 4)) == 11\n\n\n"
        "def test_remainder_is_distributed():\n"
        "    assert allocate(100, 3) == [34, 33, 33]\n"
        "    assert allocate(11, 4) == [3, 3, 3, 2]\n"
    ))

    report = RepairEngine(max_verifier_calls=20).repair(tmp_path)

    assert report.solved is True
    assert report.final_failed == 0
    assert report.accepted_without_execution == 0
    assert report.verifier_calls >= 2
    assert report.applied is True
    assert run_pytest(tmp_path).passed is True
    assert "divmod" in (tmp_path / "allocator.py").read_text(encoding="utf-8")


def test_engine_refuses_a_workspace_no_operator_justifies(tmp_path):
    original = "def add(a, b):\n    return a + b\n"
    _write(tmp_path, "calc.py", original)
    _write(tmp_path, "tests/test_calc.py", (
        "from calc import add\n\n\n"
        "def test_wrong_on_purpose():\n"
        "    assert add(2, 3) == 6\n"
    ))

    report = RepairEngine(max_verifier_calls=12).repair(tmp_path)

    assert report.solved is False
    assert report.applied is False
    assert report.final_failed == 1
    assert (tmp_path / "calc.py").read_text(encoding="utf-8") == original


def test_engine_reports_no_change_for_an_already_green_workspace(tmp_path):
    _write(tmp_path, "ok.py", "def double(value):\n    return value * 2\n")
    _write(tmp_path, "tests/test_ok.py", (
        "from ok import double\n\n\n"
        "def test_double():\n"
        "    assert double(4) == 8\n"
    ))

    report = RepairEngine(max_verifier_calls=5).repair(tmp_path)

    assert report.solved is True
    assert report.initial_failed == report.final_failed == 0
    assert report.verifier_calls == 1
    assert report.edits == []
    assert report.diffs == []
    assert report.applied is False


def test_recorded_exp010_results_pass_the_n1_gate():
    payload = json.loads((EXP_DIR / "results.json").read_text(encoding="utf-8"))
    assert payload["exp_id"] == "EXP-010"
    assert payload["suite"]["ok"] is True
    assert payload["baseline_reference"]["code_passed"] == 0
    assert payload["baseline_reference"]["code_total"] == 6
    assert payload["oracle_code"]["passed"] == payload["oracle_code"]["total"] == 6
    assert payload["summary"]["solved"] == 6
    assert payload["summary"]["accepted_without_execution"] == 0
    assert payload["summary"]["independent_scorer_agrees"] is True
    assert payload["gate"]["passed"] is True
    solved = [row for row in payload["tasks"] if row["scorer_passed"]]
    assert len(solved) == 6
    assert all(row["verifier_calls"] >= 1 for row in solved)


def test_recorded_exp011_results_pass_the_transfer_gate():
    payload = json.loads((EXP011_DIR / "results.json").read_text(encoding="utf-8"))
    assert payload["exp_id"] == "EXP-011"
    assert payload["suite"]["ok"] is True
    assert payload["suite"]["manifest_sha256"] == (
        "4cf5c03cb5a615f5d1bd06a3d462d2cfc820a49af4fc3adf43c5c3ba685875f1"
    )
    assert payload["oracle_code"]["passed"] == payload["oracle_code"]["total"] == 7
    assert payload["summary"]["solved"] == 7
    assert payload["summary"]["verifier_calls_total"] == 26
    assert payload["summary"]["accepted_without_execution"] == 0
    assert payload["summary"]["independent_scorer_agrees"] is True
    assert payload["gate"]["passed"] is True
    assert all(row["scorer_passed"] for row in payload["tasks"])

    # Every registered operator has an executed, accepted edit across the two runs.
    exp010 = json.loads((EXP_DIR / "results.json").read_text(encoding="utf-8"))
    used = {
        edit["operator"]
        for report in (exp010, payload)
        for row in report["tasks"]
        for edit in row.get("edits", [])
        if edit.get("verified")
    }
    assert used == set(OPERATORS)
    assert any(
        edit["operator"] == "comparison_flip"
        for row in payload["tasks"]
        for edit in row.get("edits", [])
    )
