"""Tests for EXPERIMENTS/EXP-009-n0-benchmark: the frozen N0 suite and its harness.

The suite is an evaluation instrument, so these tests check the instrument, not a
model: integrity, oracle satisfiability, answer digests, label resolution, and the
sealed-artefact guard.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

EXP_DIR = Path(__file__).resolve().parents[1] / "EXPERIMENTS" / "EXP-009-n0-benchmark"

_spec = importlib.util.spec_from_file_location("exp009_harness", EXP_DIR / "harness.py")
harness = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = harness  # dataclasses resolve their module by name
_spec.loader.exec_module(harness)


def test_suite_integrity_passes():
    result = harness.verify_integrity()
    assert result["ok"] is True
    assert result["mismatch_count"] == 0
    assert result["suite_version"] == "v1"
    assert result["checked"] >= 30


def test_canonicalisation_forms():
    assert harness.canonical(" 007 ", "int") == "7"
    assert harness.canonical("10/32", "frac") == "5/16"
    assert harness.canonical("  Hello   World ", "lower") == "hello world"
    with pytest.raises(ValueError):
        harness.canonical("1", "unknown")


def test_every_code_task_fails_before_the_fix_and_passes_after():
    tasks = harness.load_code_tasks()
    assert len(tasks) == 6
    for task in tasks:
        buggy = harness.prepare_workspace(task, oracle=False)
        try:
            result = harness.run_code_checks(buggy, timeout_s=task.timeout_s)
            assert result["passed"] is False, f"{task.id} unexpectedly passes before the fix"
        finally:
            import shutil

            shutil.rmtree(buggy, ignore_errors=True)

        fixed = harness.prepare_workspace(task, oracle=True)
        try:
            result = harness.run_code_checks(fixed, timeout_s=task.timeout_s)
            assert result["passed"] is True, f"{task.id} fails with the sealed fix: {result['summary']}"
        finally:
            import shutil

            shutil.rmtree(fixed, ignore_errors=True)


def test_reasoning_digests_match_sealed_answers_and_reject_wrong_answers():
    tasks = harness.load_reasoning_tasks()
    answers = harness.sealed_answers()
    assert len(tasks) == 12
    for task in tasks:
        assert task.id in answers, f"missing sealed answer for {task.id}"
        assert harness.check_exact_answer(task, answers[task.id]) is True
        assert harness.check_exact_answer(task, "definitely-not-the-answer") is False


def test_retrieval_labels_resolve_to_documents_and_topic_view_is_consistent():
    docs, queries = harness.load_retrieval()
    doc_ids = {doc["id"] for doc in docs}
    labels = harness.sealed_labels()["retrieval"]
    assert len(queries) == 18
    assert len(doc_ids) == 36
    assert set(labels) == {query.id for query in queries}
    for ids in labels.values():
        assert ids and set(ids) <= doc_ids

    # The English and Arabic renderings of the same fact share a topic id.
    assert harness.topic_id("d01") == harness.topic_id("d19")
    assert harness.topic_id("d18") == harness.topic_id("d36")
    exact = {qid: list(ids) for qid, ids in labels.items()}
    assert harness.score_retrieval(exact)["recall_at_5"] == 1.0
    assert harness.score_retrieval_topics(exact)["topic_recall_at_5"] == 1.0


def test_sealed_artefacts_are_guarded_from_solver_paths():
    with pytest.raises(harness.SealedAccessError):
        harness.assert_not_sealed(harness.SEALED_DIR / "labels.json")
    for task in harness.load_code_tasks():
        workspace = harness.prepare_workspace(task, oracle=False)
        try:
            copied = {path.name for path in harness.iter_files(workspace)}
            assert "task.json" not in copied
            assert not any("sealed" in path.as_posix() for path in harness.iter_files(workspace))
        finally:
            import shutil

            shutil.rmtree(workspace, ignore_errors=True)


def test_recorded_results_match_the_frozen_suite():
    import json

    payload = json.loads((EXP_DIR / "results.json").read_text(encoding="utf-8"))
    assert payload["exp_id"] == "EXP-009"
    assert payload["suite"]["ok"] is True
    assert payload["suite"]["manifest_sha256"] == harness.verify_integrity()["manifest_sha256"]
    assert payload["oracle"]["ok"] is True
    assert payload["oracle"]["code"]["passed"] == payload["oracle"]["code"]["total"] == 6
    assert payload["oracle"]["reasoning"]["passed"] == payload["oracle"]["reasoning"]["total"] == 12
    assert payload["baseline"]["code"]["passed"] == 0
    assert payload["baseline"]["reasoning"]["passed"] == 0
