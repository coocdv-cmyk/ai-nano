"""EXP-011 N1-heldout — same-family transfer probe for the frozen repair operators.

The operator set was frozen while `EXP-010` was measured. These seven tasks were
authored **afterwards**, with different names, domains and call shapes, and are
run **once** without tuning. A failure here is a real coverage gap, recorded as
such; it is not quietly repaired by editing the operators before the run.

Usage:
    python EXPERIMENTS/EXP-011-n1-heldout/run_exp.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]
N0_DIR = REPO_ROOT / "EXPERIMENTS" / "EXP-009-n0-benchmark"
SUITE = EXP_DIR / "task_suite" / "v1"
for _path in (str(REPO_ROOT), str(N0_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import harness  # noqa: E402

from omega_nano.brain.repair import OPERATORS, RepairEngine  # noqa: E402

# Reuse the frozen EXP-009 harness code with a different suite root.
harness.SUITE_DIR = SUITE
harness.SEALED_DIR = SUITE / ".sealed"

RESULTS = EXP_DIR / "results.json"
HISTORY_DIR = EXP_DIR / "results" / "history"


def _hardware() -> dict[str, Any]:
    try:
        from omega_nano.hw import detect

        return detect()
    except Exception as exc:  # pragma: no cover - informational only
        return {"error": f"{type(exc).__name__}: {exc}"}


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
    return 0.0


def run(tasks: list[harness.CodeTask], max_calls: int, max_depth: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        workspace = harness.prepare_workspace(task, oracle=False)
        t0 = time.perf_counter()
        try:
            report = RepairEngine(max_verifier_calls=max_calls, max_depth=max_depth).repair(workspace)
            scored = harness.run_code_checks(workspace, timeout_s=task.timeout_s)
            rows.append({
                "id": task.id,
                "title": task.title,
                "engine_solved": report.solved,
                "scorer_passed": scored["passed"],
                "initial_failed": report.initial_failed,
                "final_failed": report.final_failed,
                "verifier_calls": report.verifier_calls,
                "attempted_candidates": len(report.attempts),
                "accepted_without_execution": report.accepted_without_execution,
                "budget_exhausted": report.budget_exhausted,
                "edits": report.edits,
                "attempts": report.attempts,
                "targets": report.targets[:4],
                "diffs": report.diffs,
                "elapsed_s": round(time.perf_counter() - t0, 3),
            })
        except Exception as exc:
            rows.append({"id": task.id, "title": task.title, "engine_solved": False,
                         "scorer_passed": False, "error": f"{type(exc).__name__}: {exc}",
                         "elapsed_s": round(time.perf_counter() - t0, 3)})
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    solved = [row for row in rows if row.get("engine_solved")]
    calls = [row["verifier_calls"] for row in solved if "verifier_calls" in row]
    return {
        "tasks": rows,
        "summary": {
            "solved": len(solved),
            "total": len(rows),
            "pass_rate": round(len(solved) / len(rows), 4) if rows else 0.0,
            "verifier_calls_total": sum(row.get("verifier_calls", 0) for row in rows),
            "verifier_calls_per_solved_mean": round(sum(calls) / len(calls), 2) if calls else 0.0,
            "accepted_without_execution": sum(row.get("accepted_without_execution", 0) for row in rows),
            "independent_scorer_agrees": all(row.get("scorer_passed") == row.get("engine_solved")
                                             for row in rows),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP-011 N1 held-out transfer probe")
    parser.add_argument("--max-calls", type=int, default=80)
    parser.add_argument("--max-depth", type=int, default=3)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    integrity = harness.verify_integrity(strict=True)
    tasks = harness.load_code_tasks()
    oracle = harness.score_code_tasks(tasks, oracle=True)
    payload: dict[str, Any] = {
        "exp_id": "EXP-011",
        "title": "N1-heldout — same-family transfer probe (operators frozen on EXP-010)",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "suite": integrity,
        "hardware": _hardware(),
        "operators": list(OPERATORS),
        "configuration": {"max_verifier_calls": args.max_calls, "max_depth": args.max_depth},
        "oracle_code": {key: oracle[key] for key in ("passed", "total", "pass_rate", "elapsed_s")},
    }
    payload.update(run(tasks, args.max_calls, args.max_depth))

    summary = payload["summary"]
    gate = {
        "oracle_still_100": oracle["passed"] == oracle["total"],
        "zero_unexecuted_patches": summary["accepted_without_execution"] == 0,
        "independent_scorer_agrees": summary["independent_scorer_agrees"],
        "transfer_rate_at_least_half": summary["pass_rate"] >= 0.5,
    }
    gate["passed"] = all(gate.values())
    payload["gate"] = gate
    payload["efficiency"] = {"peak_rss_mb": _rss_mb(),
                             "total_wall_seconds": round(time.perf_counter() - started, 3)}

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS.is_file():
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        shutil.copy2(RESULTS, HISTORY_DIR / f"EXP-011-{stamp}.json")
    RESULTS.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({"summary": summary, "gate": gate, "oracle": payload["oracle_code"],
                      "efficiency": payload["efficiency"]}, indent=2, ensure_ascii=False))
    for row in payload["tasks"]:
        state = "SOLVED" if row.get("engine_solved") else "OPEN"
        print(f"  {state:6s} {row['id']:28s} calls={row.get('verifier_calls', '-')} "
              f"failed {row.get('initial_failed', '-')}->{row.get('final_failed', '-')}")
    print(f"wrote {RESULTS}")
    return 0 if gate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
