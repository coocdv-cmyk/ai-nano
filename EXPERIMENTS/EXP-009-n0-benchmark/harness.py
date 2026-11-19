"""EXP-009 N0 — sealed benchmark harness (suite `v1`).

The harness is the *evaluator*. It may read `.sealed/`; a solver must never see it.
Everything a solver receives is a prompt, a workspace copy, or a verifier.

Design rules
------------
* The suite is frozen: `manifest.json` pins the SHA-256 of every file under
  `task_suite/v1`, including the sealed solutions. `verify_integrity()` must pass
  before any score is recorded.
* Code tasks are scored by executing the task's own ``pytest`` suite in an
  isolated temporary copy. A patch is accepted only when the tests actually ran
  and passed.
* Reasoning answers are compared by SHA-256 of a canonical form, so the answer
  need not be stored in plaintext for scoring.
* Retrieval is scored against sealed labels by recall@k and MRR.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

EXP_DIR = Path(__file__).resolve().parent
SUITE_DIR = EXP_DIR / "task_suite" / "v1"
SEALED_DIR = SUITE_DIR / ".sealed"

# Deterministic environment for verifier subprocesses: no inherited secrets.
_SAFE_ENV_KEYS = ("PATH", "SYSTEMROOT", "SystemRoot", "WINDIR", "TEMP", "TMP", "PATHEXT", "COMSPEC")


class SuiteIntegrityError(RuntimeError):
    """Raised when the frozen suite does not match its manifest."""


class SealedAccessError(RuntimeError):
    """Raised when a solve-time code path tries to read sealed artefacts."""


# --------------------------------------------------------------------------- #
# Hashing and canonicalisation
# --------------------------------------------------------------------------- #

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value: str | int, mode: str) -> str:
    """Canonical text form used for exact-answer hashing.

    Modes are deliberately small and explicit so a solver can reproduce them
    from the task statement: ``int``, ``frac``, ``lower``.
    """
    text = str(value).strip()
    if mode == "int":
        return str(int(text.replace(",", "").replace("_", "")))
    if mode == "frac":
        from fractions import Fraction

        frac = Fraction(text.replace(" ", ""))
        return f"{frac.numerator}/{frac.denominator}"
    if mode == "lower":
        return " ".join(text.casefold().split())
    raise ValueError(f"unknown canonicalisation mode: {mode!r}")


def answer_digest(value: str | int, mode: str) -> str:
    return sha256_bytes(canonical(value, mode).encode("utf-8"))


# --------------------------------------------------------------------------- #
# Suite loading
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CodeTask:
    id: str
    title: str
    statement: str
    path: Path
    timeout_s: float

    @property
    def repo_dir(self) -> Path:
        return self.path / "repo"

    @property
    def tests_dir(self) -> Path:
        return self.path / "tests"

    @property
    def sealed_dir(self) -> Path:
        return SEALED_DIR / "solutions" / self.id


@dataclass(frozen=True)
class ReasoningTask:
    id: str
    statement: str
    canon: str
    answer_sha256: str


@dataclass(frozen=True)
class RetrievalQuery:
    id: str
    query: str
    lang: str


def load_manifest() -> dict[str, Any]:
    path = SUITE_DIR / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing suite manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def iter_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def verify_integrity(strict: bool = True) -> dict[str, Any]:
    """Compare every file under the suite against ``manifest.json``."""
    manifest = load_manifest()
    recorded: dict[str, str] = manifest["files"]
    mismatches: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in iter_files(SUITE_DIR):
        rel = path.relative_to(SUITE_DIR).as_posix()
        if rel == "manifest.json":
            continue
        seen.add(rel)
        expected = recorded.get(rel)
        if expected is None:
            mismatches.append({"path": rel, "problem": "unlisted file"})
            continue
        actual = sha256_file(path)
        if actual != expected:
            mismatches.append({"path": rel, "problem": "hash mismatch", "expected": expected, "actual": actual})
    for rel in sorted(set(recorded) - seen):
        mismatches.append({"path": rel, "problem": "missing file"})
    result = {
        "ok": not mismatches,
        "checked": len(seen),
        "suite_version": manifest.get("suite_version"),
        "manifest_sha256": sha256_file(SUITE_DIR / "manifest.json"),
        "mismatches": mismatches[:20],
        "mismatch_count": len(mismatches),
    }
    if strict and mismatches:
        raise SuiteIntegrityError(f"suite integrity failed: {len(mismatches)} mismatch(es)")
    return result


def load_code_tasks() -> list[CodeTask]:
    tasks: list[CodeTask] = []
    for task_json in sorted((SUITE_DIR / "code").glob("*/task.json")):
        meta = json.loads(task_json.read_text(encoding="utf-8"))
        tasks.append(
            CodeTask(
                id=meta["id"],
                title=meta["title"],
                statement=meta["statement"],
                path=task_json.parent,
                timeout_s=float(meta.get("timeout_s", 60.0)),
            )
        )
    return tasks


def load_reasoning_tasks() -> list[ReasoningTask]:
    data = json.loads((SUITE_DIR / "reasoning.json").read_text(encoding="utf-8"))
    return [
        ReasoningTask(
            id=item["id"],
            statement=item["statement"],
            canon=item["canon"],
            answer_sha256=item["answer_sha256"],
        )
        for item in data["tasks"]
    ]


def load_retrieval() -> tuple[list[dict[str, Any]], list[RetrievalQuery]]:
    docs = json.loads((SUITE_DIR / "retrieval" / "docs.json").read_text(encoding="utf-8"))["documents"]
    queries = json.loads((SUITE_DIR / "retrieval" / "queries.json").read_text(encoding="utf-8"))["queries"]
    return docs, [RetrievalQuery(id=q["id"], query=q["query"], lang=q["lang"]) for q in queries]


def sealed_labels() -> dict[str, Any]:
    """Evaluator-only labels. Never call from a solver path."""
    return json.loads((SEALED_DIR / "labels.json").read_text(encoding="utf-8"))


def sealed_answers() -> dict[str, str]:
    return json.loads((SEALED_DIR / "reasoning_answers.json").read_text(encoding="utf-8"))


def assert_not_sealed(path: Path) -> None:
    """Guard for solve-time code paths."""
    resolved = Path(path).resolve()
    if SEALED_DIR.resolve() == resolved or SEALED_DIR.resolve() in resolved.parents:
        raise SealedAccessError(f"solver attempted to read sealed artefact: {path}")


# --------------------------------------------------------------------------- #
# Code track
# --------------------------------------------------------------------------- #

def _safe_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {k: os.environ[k] for k in _SAFE_ENV_KEYS if k in os.environ}
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if extra:
        env.update(extra)
    return env


def prepare_workspace(task: CodeTask, *, oracle: bool, root: Path | None = None) -> Path:
    """Materialise a flat, isolated copy of the task (repo + tests).

    The layout is flat so task modules import each other by plain module name,
    exactly as the sealed tests expect.
    """
    dest = Path(root or tempfile.mkdtemp(prefix=f"exp009_{task.id}_"))
    dest.mkdir(parents=True, exist_ok=True)
    for src in iter_files(task.repo_dir):
        assert_not_sealed(src)  # repo/ must never contain sealed material
        target = dest / src.relative_to(task.repo_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    if oracle:
        if not task.sealed_dir.is_dir():
            raise FileNotFoundError(f"missing sealed solution for {task.id}")
        for src in iter_files(task.sealed_dir):
            target = dest / src.relative_to(task.sealed_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
    for src in iter_files(task.tests_dir):
        target = dest / "tests" / src.relative_to(task.tests_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    return dest


def run_code_checks(workspace: Path, timeout_s: float = 120.0) -> dict[str, Any]:
    """Run the task's own pytest suite inside ``workspace`` and record the result."""
    cmd = [sys.executable, "-m", "pytest", "-q", "tests", "-p", "no:cacheprovider"]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(workspace),
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=max(5.0, float(timeout_s)),
            env=_safe_env({"PYTHONPATH": str(workspace)}),
        )
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "status": "TIMEOUT",
            "returncode": None,
            "elapsed_s": round(time.perf_counter() - t0, 3),
            "summary": f"TIMEOUT after {timeout_s}s",
            "output_tail": "",
        }
    stdout = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace")
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    summary = lines[-1] if lines else f"exit {proc.returncode}"
    return {
        "passed": proc.returncode == 0,
        "status": "TESTS_PASSED" if proc.returncode == 0 else "REFUTED",
        "returncode": proc.returncode,
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "summary": summary,
        "output_tail": (stderr or stdout)[-1500:],
    }


def score_code_tasks(tasks: Iterable[CodeTask], *, oracle: bool) -> dict[str, Any]:
    results = []
    for task in tasks:
        workspace = prepare_workspace(task, oracle=oracle)
        outcome = run_code_checks(workspace, timeout_s=task.timeout_s)
        outcome.update({"id": task.id, "title": task.title})
        results.append(outcome)
        shutil.rmtree(workspace, ignore_errors=True)
    passed = sum(1 for r in results if r["passed"])
    return {
        "passed": passed,
        "total": len(results),
        "pass_rate": round(passed / len(results), 4) if results else 0.0,
        "elapsed_s": round(sum(r["elapsed_s"] for r in results), 3),
        "tasks": results,
    }


# --------------------------------------------------------------------------- #
# Reasoning track
# --------------------------------------------------------------------------- #

def check_exact_answer(task: ReasoningTask, candidate: str) -> bool:
    if not isinstance(candidate, str) or not candidate.strip():
        return False
    try:
        return answer_digest(candidate, task.canon) == task.answer_sha256
    except (ValueError, ZeroDivisionError):
        return False


# --------------------------------------------------------------------------- #
# Retrieval track
# --------------------------------------------------------------------------- #

def score_retrieval(ranked: dict[str, list[str]], k: int = 5) -> dict[str, Any]:
    labels = sealed_labels()["retrieval"]
    holds = []
    rr = []
    for qid, relevant in labels.items():
        expected = set(relevant)
        got = [doc_id for doc_id in ranked.get(qid, [])[:k]]
        hit = next((i for i, doc_id in enumerate(got, 1) if doc_id in expected), None)
        holds.append(1 if hit else 0)
        rr.append(1.0 / hit if hit else 0.0)
    total = len(labels)
    return {
        "queries": total,
        "k": k,
        f"recall_at_{k}": round(sum(holds) / total, 4) if total else 0.0,
        "mrr": round(sum(rr) / total, 4) if total else 0.0,
        "hits": sum(holds),
    }


def topic_id(doc_id: str) -> str:
    """Map a document to its topic.

    Documents `d01`–`d18` (English) and `d19`–`d36` (Arabic) are two renderings of
    the same fact, so a query that names a fact is satisfied by either language.
    Strict scoring ignores this; topic scoring does not. Both are reported because
    they answer different questions ("did it find the exact document?" vs "did it
    find the right fact?").
    """
    index = int(doc_id[1:])
    if 19 <= index <= 36:
        index -= 18
    return f"t{index:02d}"


def score_retrieval_topics(ranked: dict[str, list[str]], k: int = 5) -> dict[str, Any]:
    labels = sealed_labels()["retrieval"]
    holds = []
    rr = []
    for qid, relevant in labels.items():
        expected = {topic_id(doc_id) for doc_id in relevant}
        got = [topic_id(doc_id) for doc_id in ranked.get(qid, [])[:k] if doc_id]
        hit = next((i for i, topic in enumerate(got, 1) if topic in expected), None)
        holds.append(1 if hit else 0)
        rr.append(1.0 / hit if hit else 0.0)
    total = len(labels)
    return {
        "queries": total,
        "k": k,
        f"topic_recall_at_{k}": round(sum(holds) / total, 4) if total else 0.0,
        "topic_mrr": round(sum(rr) / total, 4) if total else 0.0,
        "topic_hits": sum(holds),
    }
