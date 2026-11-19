"""EXP-010 N1 — execution-verified AST repair on the frozen N0 code track.

Usage:
    python EXPERIMENTS/EXP-010-n1-ast-repair/run_exp.py
    python EXPERIMENTS/EXP-010-n1-ast-repair/run_exp.py --max-calls 40
    python EXPERIMENTS/EXP-010-n1-ast-repair/run_exp.py --task n0-code-04-cache-key

The suite is read from ``EXP-009-n0-benchmark`` and is **not** modified. Every
solved task is re-scored by the frozen harness after the repair engine returns,
so the engine's own report can never be the only evidence of success.
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
for _path in (str(REPO_ROOT), str(N0_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import harness  # noqa: E402

from omega_nano.brain.repair import OPERATORS, RepairEngine  # noqa: E402

RESULTS = EXP_DIR / "results.json"
HISTORY_DIR = EXP_DIR / "results" / "history"
N0_RESULTS = N0_DIR / "results.json"


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


def _baseline_reference() -> dict[str, Any]:
    payload = json.loads(N0_RESULTS.read_text(encoding="utf-8"))
    code = payload["baseline"]["code"]
    return {
        "experiment": payload.get("exp_id"),
        "suite_manifest_sha256": payload["suite"]["manifest_sha256"],
        "code_passed": code["passed"],
        "code_total": code["total"],
        "code_pass_rate": code["pass_rate"],
        "source": str(N0_RESULTS.relative_to(REPO_ROOT)).replace("\\", "/"),
    }


def run(tasks: list[harness.CodeTask], max_calls: int, max_depth: int) -> dict[str, Any]:
    started = time.perf_counter()
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
                "scorer_summary": scored["summary"],
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
        except Exception as exc:  # recorded, never hidden
            rows.append({
                "id": task.id,
                "title": task.title,
                "engine_solved": False,
                "scorer_passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_s": round(time.perf_counter() - t0, 3),
            })
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    solved = [row for row in rows if row.get("engine_solved")]
    unexecuted = sum(row.get("accepted_without_execution", 0) for row in rows)
    scorer_agrees = all(row.get("scorer_passed") == row.get("engine_solved") for row in rows)
    calls = [row["verifier_calls"] for row in solved if "verifier_calls" in row]

    return {
        "tasks": rows,
        "summary": {
            "solved": len(solved),
            "total": len(rows),
            "pass_rate": round(len(solved) / len(rows), 4) if rows else 0.0,
            "verifier_calls_total": sum(row.get("verifier_calls", 0) for row in rows),
            "verifier_calls_per_solved_mean": round(sum(calls) / len(calls), 2) if calls else 0.0,
            "verifier_calls_per_solved_max": max(calls) if calls else 0,
            "accepted_without_execution": unexecuted,
            "independent_scorer_agrees": scorer_agrees,
        },
        "wall_s": round(time.perf_counter() - started, 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP-010 N1 AST repair")
    parser.add_argument("--max-calls", type=int, default=80)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--task", action="append", default=[], help="limit to a task id (repeatable)")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    integrity = harness.verify_integrity(strict=True)
    baseline = _baseline_reference()
    tasks = harness.load_code_tasks()
    if args.task:
        wanted = set(args.task)
        tasks = [task for task in tasks if task.id in wanted]
        if not tasks:
            parser.error(f"no such task: {sorted(wanted)}")

    oracle = harness.score_code_tasks(harness.load_code_tasks(), oracle=True)
    payload: dict[str, Any] = {
        "exp_id": "EXP-010",
        "title": "N1 — execution-verified AST repair on the frozen N0 code track",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "suite": integrity,
        "hardware": _hardware(),
        "operators": list(OPERATORS),
        "configuration": {"max_verifier_calls": args.max_calls, "max_depth": args.max_depth},
        "baseline_reference": baseline,
        "oracle_code": {key: oracle[key] for key in ("passed", "total", "pass_rate", "elapsed_s")},
    }
    payload.update(run(tasks, args.max_calls, args.max_depth))

    summary = payload["summary"]
    gate = {
        "code_pass_rate_above_baseline": summary["solved"] > baseline["code_passed"],
        "zero_unexecuted_patches": summary["accepted_without_execution"] == 0,
        "independent_scorer_agrees": summary["independent_scorer_agrees"],
        "oracle_still_100": payload["oracle_code"]["passed"] == payload["oracle_code"]["total"],
    }
    gate["passed"] = all(gate.values())
    payload["gate"] = gate
    payload["efficiency"] = {
        "peak_rss_mb": _rss_mb(),
        "total_wall_seconds": round(time.perf_counter() - started, 3),
    }
    payload["gaps"] = _gaps(payload)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS.is_file():
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        shutil.copy2(RESULTS, HISTORY_DIR / f"EXP-010-{stamp}.json")
    RESULTS.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "summary": summary,
        "gate": gate,
        "baseline": {"passed": baseline["code_passed"], "total": baseline["code_total"]},
        "oracle": payload["oracle_code"],
        "efficiency": payload["efficiency"],
    }, indent=2, ensure_ascii=False))
    for row in payload["tasks"]:
        state = "SOLVED" if row.get("engine_solved") else "OPEN"
        print(f"  {state:6s} {row['id']:28s} calls={row.get('verifier_calls', '-')} "
              f"failed {row.get('initial_failed', '-')}->{row.get('final_failed', '-')}")
    print(f"wrote {RESULTS}")
    return 0 if gate["passed"] else 2


def _gaps(payload: dict[str, Any]) -> list[str]:
    gaps: list[str] = []
    summary = payload["summary"]
    baseline = payload["baseline_reference"]
    if summary["solved"] <= baseline["code_passed"]:
        gaps.append("N1 did not move the code track above the frozen N0 baseline.")
    if summary["accepted_without_execution"]:
        gaps.append("An accepted patch had no executed verification run; the result is invalid.")
    if not summary["independent_scorer_agrees"]:
        gaps.append("The frozen scorer disagreed with the engine report on at least one task.")
    for row in payload["tasks"]:
        if not row.get("engine_solved"):
            gaps.append(f"{row['id']}: no operator produced a passing patch inside the verifier budget.")
    if not gaps:
        gaps.append("N1 covers eight operator families on six synthetic multi-file tasks; this is not "
                    "open-world repository repair. Held-out transfer must be measured before generalising.")
    return gaps


if __name__ == "__main__":
    raise SystemExit(main())
