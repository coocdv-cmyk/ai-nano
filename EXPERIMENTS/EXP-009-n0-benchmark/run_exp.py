"""EXP-009 N0 — sealed EXP-N benchmark v1: oracle + product baseline.

Usage:
    python EXPERIMENTS/EXP-009-n0-benchmark/run_exp.py                # oracle + baseline
    python EXPERIMENTS/EXP-009-n0-benchmark/run_exp.py --oracle-only
    python EXPERIMENTS/EXP-009-n0-benchmark/run_exp.py --baseline-only

The oracle run proves the suite is satisfiable; the baseline run measures what
the product can do today. Both are written to ``results.json`` together with the
suite manifest hash, hardware and resource cost.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]
for path in (str(REPO_ROOT), str(EXP_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import harness  # noqa: E402

RESULTS = EXP_DIR / "results.json"
HISTORY_DIR = EXP_DIR / "results" / "history"


def _rss_mb() -> float:
    try:
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _PMC()
        counters.cb = ctypes.sizeof(counters)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
        k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        if k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return round(counters.WorkingSetSize / (1024 * 1024), 2)
    except Exception:
        pass
    try:
        import resource

        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2)
    except Exception:
        return 0.0


def _hardware() -> dict[str, Any]:
    try:
        from omega_nano.hw import detect

        return detect()
    except Exception as exc:  # pragma: no cover - informational only
        return {"error": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------- #
# Oracle
# --------------------------------------------------------------------------- #

def run_oracle(code_tasks: list[harness.CodeTask]) -> dict[str, Any]:
    t0 = time.perf_counter()
    code = harness.score_code_tasks(code_tasks, oracle=True)

    reasoning_tasks = harness.load_reasoning_tasks()
    answers = harness.sealed_answers()
    reasoning_rows = []
    for task in reasoning_tasks:
        answer = answers.get(task.id, "")
        ok = harness.check_exact_answer(task, answer)
        reasoning_rows.append({"id": task.id, "passed": ok})
    reasoning_passed = sum(1 for row in reasoning_rows if row["passed"])

    docs, queries = harness.load_retrieval()
    doc_ids = {doc["id"] for doc in docs}
    labels = harness.sealed_labels()["retrieval"]
    label_ok = all(set(ids) <= doc_ids and ids for ids in labels.values())
    ranked = {qid: list(ids) for qid, ids in labels.items()}
    retrieval = harness.score_retrieval(ranked, k=5)
    retrieval.update({key: value for key, value in harness.score_retrieval_topics(ranked, k=5).items()
                      if key not in ("queries", "k")})
    retrieval["labels_resolve_to_documents"] = label_ok
    retrieval["passed"] = label_ok and retrieval["hits"] == retrieval["queries"]

    return {
        "code": {k: code[k] for k in ("passed", "total", "pass_rate", "elapsed_s")},
        "reasoning": {"passed": reasoning_passed, "total": len(reasoning_tasks), "tasks": reasoning_rows},
        "retrieval": retrieval,
        "wall_s": round(time.perf_counter() - t0, 3),
        "ok": code["passed"] == code["total"] == len(code_tasks)
        and reasoning_passed == len(reasoning_tasks)
        and retrieval["passed"],
    }


# --------------------------------------------------------------------------- #
# Baseline — the product as it exists today
# --------------------------------------------------------------------------- #

def _answer_candidates(result: dict[str, Any]) -> list[str]:
    """Generous extraction: any numeric token the product surfaced, best-first."""
    chunks: list[str] = []
    for key in ("solution", "summary", "output"):
        value = result.get(key)
        if isinstance(value, str):
            chunks.append(value)
    for step in result.get("steps") or []:
        if isinstance(step, dict) and isinstance(step.get("output"), str):
            chunks.append(step["output"])
    text = "\n".join(chunks)
    found = re.findall(r"-?\d+(?:\s*/\s*\d+)?", text)
    ordered: list[str] = []
    for token in reversed(found):  # later output is usually the final answer
        token = token.replace(" ", "")
        if token not in ordered:
            ordered.append(token)
    return ordered


def _baseline_code(agent: Any, code_tasks: list[harness.CodeTask]) -> dict[str, Any]:
    rows = []
    for task in code_tasks:
        workspace = harness.prepare_workspace(task, oracle=False)
        t0 = time.perf_counter()
        record: dict[str, Any] = {"id": task.id, "title": task.title}
        try:
            before = {p.relative_to(workspace).as_posix(): harness.sha256_file(p)
                      for p in harness.iter_files(workspace)}
            result = agent.handle(f"{task.statement}\n\nWorkspace: {workspace}")
            after = {p.relative_to(workspace).as_posix(): harness.sha256_file(p)
                     for p in harness.iter_files(workspace)}
            record["route"] = result.get("domain")
            record["status"] = result.get("status")
            record["tier"] = result.get("tier")
            record["files_changed"] = sorted(k for k, v in after.items() if before.get(k) != v)
            record["agent_elapsed_ms"] = result.get("elapsed_ms")
        except Exception as exc:  # recorded, never hidden
            record["error"] = f"{type(exc).__name__}: {exc}"
        checks = harness.run_code_checks(workspace, timeout_s=task.timeout_s)
        record["passed"] = checks["passed"]
        record["tests"] = checks["summary"]
        record["elapsed_s"] = round(time.perf_counter() - t0, 3)
        rows.append(record)
        shutil.rmtree(workspace, ignore_errors=True)
    passed = sum(1 for row in rows if row["passed"])
    return {
        "passed": passed,
        "total": len(rows),
        "pass_rate": round(passed / len(rows), 4) if rows else 0.0,
        "elapsed_s": round(sum(row["elapsed_s"] for row in rows), 3),
        "tasks": rows,
    }


def _baseline_reasoning(agent: Any, tasks: list[harness.ReasoningTask]) -> dict[str, Any]:
    rows = []
    for task in tasks:
        record: dict[str, Any] = {"id": task.id, "passed": False, "candidates_tried": 0}
        try:
            result = agent.handle(task.statement)
            record["route"] = result.get("domain")
            record["status"] = result.get("status")
            candidates = _answer_candidates(result)
            record["candidates_tried"] = len(candidates)
            for candidate in candidates:
                if harness.check_exact_answer(task, candidate):
                    record["passed"] = True
                    record["answer"] = candidate
                    break
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(record)
    passed = sum(1 for row in rows if row["passed"])
    return {
        "passed": passed,
        "total": len(rows),
        "pass_rate": round(passed / len(rows), 4) if rows else 0.0,
        "tasks": rows,
    }


def _baseline_retrieval(k: int = 5) -> dict[str, Any]:
    from omega_nano.memory import MemoryFabric

    docs, queries = harness.load_retrieval()
    tmp = Path(tempfile.mkdtemp(prefix="exp009_mem_"))
    t0 = time.perf_counter()
    memory = MemoryFabric(tmp / "memory.db")
    try:
        for doc in docs:
            memory.remember(doc["text"], kind="semantic", source="exp009", key=doc["id"])
        ranked: dict[str, list[str]] = {}
        rows = []
        for query in queries:
            hits = memory.recall(query.query, k=k)
            ids = [hit.key for hit in hits if hit.key]
            ranked[query.id] = ids
            rows.append({"id": query.id, "top": ids[:k]})
        score = harness.score_retrieval(ranked, k=k)
        score.update({key: value for key, value in harness.score_retrieval_topics(ranked, k=k).items()
                      if key not in ("queries", "k")})
        score["elapsed_s"] = round(time.perf_counter() - t0, 3)
        score["queries"] = rows
        return score
    finally:
        memory.close()
        shutil.rmtree(tmp, ignore_errors=True)


def run_baseline(code_tasks: list[harness.CodeTask], k: int = 5) -> dict[str, Any]:
    from omega_nano.agent import OmegaAgent

    home = Path(tempfile.mkdtemp(prefix="exp009_agent_"))
    t0 = time.perf_counter()
    agent = OmegaAgent(home=home)
    try:
        code = _baseline_code(agent, code_tasks)
        reasoning = _baseline_reasoning(agent, harness.load_reasoning_tasks())
    finally:
        agent.close()
        shutil.rmtree(home, ignore_errors=True)
    retrieval = _baseline_retrieval(k=k)
    return {
        "code": code,
        "reasoning": reasoning,
        "retrieval": retrieval,
        "wall_s": round(time.perf_counter() - t0, 3),
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP-009 N0 sealed benchmark")
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--k", type=int, default=5, help="retrieval cut-off (default 5)")
    args = parser.parse_args(argv)
    if args.oracle_only and args.baseline_only:
        parser.error("--oracle-only and --baseline-only are mutually exclusive")

    started = time.perf_counter()
    integrity = harness.verify_integrity(strict=True)
    code_tasks = harness.load_code_tasks()
    payload: dict[str, Any] = {
        "exp_id": "EXP-009",
        "title": "N0 — sealed EXP-N benchmark v1 (oracle + product baseline)",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "suite": integrity,
        "hardware": _hardware(),
        "counts": {
            "code": len(code_tasks),
            "reasoning": len(harness.load_reasoning_tasks()),
            "retrieval_queries": len(harness.load_retrieval()[1]),
        },
    }

    if not args.baseline_only:
        payload["oracle"] = run_oracle(code_tasks)
    if not args.oracle_only:
        payload["baseline"] = run_baseline(code_tasks, k=args.k)

    payload["efficiency"] = {
        "peak_rss_mb": _rss_mb(),
        "total_wall_seconds": round(time.perf_counter() - started, 3),
    }
    payload["gaps"] = _gaps(payload)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS.is_file():
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        shutil.copy2(RESULTS, HISTORY_DIR / f"EXP-009-{stamp}.json")
    RESULTS.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("counts", "efficiency")}, indent=2, ensure_ascii=False))
    if "oracle" in payload:
        print(f"oracle ok={payload['oracle']['ok']} code={payload['oracle']['code']['passed']}"
              f"/{payload['oracle']['code']['total']}")
    if "baseline" in payload:
        base = payload["baseline"]
        print(f"baseline code={base['code']['passed']}/{base['code']['total']} "
              f"reasoning={base['reasoning']['passed']}/{base['reasoning']['total']} "
              f"recall@5={base['retrieval']['recall_at_5']} mrr={base['retrieval']['mrr']} "
              f"topic_recall@5={base['retrieval']['topic_recall_at_5']}")
    print(f"wrote {RESULTS}")
    return 0


def _gaps(payload: dict[str, Any]) -> list[str]:
    gaps = []
    oracle = payload.get("oracle")
    if oracle is not None and not oracle["ok"]:
        gaps.append("SUITE INVALID: the oracle did not reach 100%; no baseline may be interpreted until it does.")
    baseline = payload.get("baseline")
    if baseline is not None:
        if baseline["code"]["passed"] == 0:
            gaps.append("Code track at 0: the product has no patch operator or traceback-driven localization (EXP-N N1).")
        if baseline["reasoning"]["passed"] == 0:
            gaps.append("Reasoning track at 0: no general arithmetic/word-problem path exists outside the symbolic families.")
        if baseline["retrieval"]["recall_at_5"] < 1.0:
            gaps.append("Retrieval is below the sealed label ceiling; EXP-N N3 must raise it without raising false accepts.")
    return gaps


if __name__ == "__main__":
    raise SystemExit(main())
