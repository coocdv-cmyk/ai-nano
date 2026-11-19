"""Unified Offline OMEGA-NANO V1 Agent (M0..M4 Integration).

Brings every subsystem under one cohesive, CPU-first, zero-hallucination API:
  * Checkpointed tool execution & security (`Engine`, `Store`, `RuleReasoner`)
  * Bilingual semantic memory + AST repo indexer (`MemoryFabric`, `index_repository`)
  * Execution-guided neuro-symbolic search + skill compiler + L1/L3 guards
    (`SelfPlayEngine`, `NeuroSymbolicSolver`, `SkillLibrary`)
  * Native 2-bit packed / ACT / CopyGate neural core + Wake-Sleep distillation
    (`NeuralCodeProposer`, `sleep_distill`)
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import time
from typing import Any

from .brain import BlindProblem, BlindVault, CodeVerifier, NeuroSymbolicSolver, SelfPlayEngine, SkillLibrary, index_repository
from .config import Config, load_config
from .core.distill import NeuralCodeProposer, sleep_distill
from .engine import Engine


def _rss_mb() -> float:
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
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

            counters = _PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.K32GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
                wintypes.DWORD,
            ]
            k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
            h = k32.GetCurrentProcess()
            if k32.K32GetProcessMemoryInfo(h, ctypes.byref(counters), counters.cb):
                return round(counters.WorkingSetSize / (1024 * 1024), 2)
        else:
            import resource

            return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2)
    except Exception:
        pass
    return 0.0


_CODE_SPEC_RE = re.compile(
    r"\b(?:write|implement)\s+(?:a\s+)?(?:python\s+)?(?:function|class)\s+([A-Za-z_]\w*)"
    r"|\bdef\s+([A-Za-z_]\w*)\s*\(",
    re.IGNORECASE,
)


def _extract_inline_tests(goal: str) -> tuple[str, str]:
    """Extract optional explicit assertions (`tests:` / `assert ...`) supplied in a goal."""
    if m := re.search(r"(?:tests|verify|asserts)\s*:\s*(.+)$", goal, re.IGNORECASE | re.DOTALL):
        return goal[: m.start()].strip(), m.group(1).strip()
    asserts = [ln.strip() for ln in goal.splitlines() if ln.strip().startswith("assert ")]
    if asserts:
        stmt_lines = [ln for ln in goal.splitlines() if not ln.strip().startswith("assert ")]
        return "\n".join(stmt_lines).strip(), "\n".join(asserts)
    return goal.strip(), ""


class OmegaAgent:
    """Integrated offline OMEGA-NANO V1 agent runtime."""

    def __init__(
        self,
        home: str | Path | None = None,
        *,
        cfg: Config | None = None,
        core_checkpoint: str | Path | None = None,
    ):
        self.cfg = cfg or load_config(home)
        self.engine = Engine(self.cfg)
        self.skills = SkillLibrary(self.cfg.home / "brain.db", memory=self.engine.memory)
        self.selfplay = SelfPlayEngine(self.skills)

        ckpt = Path(core_checkpoint) if core_checkpoint else (self.cfg.home / "core_v2.pt")
        self.core_checkpoint = ckpt
        self.proposer = NeuralCodeProposer(ckpt)
        self.solver = NeuroSymbolicSolver(neural_proposer=self.proposer)
        self._repo_indexed = False

    def close(self) -> None:
        self.skills.close()
        self.engine.close()

    def index_codebase(self, root: str | Path | None = None) -> dict[str, Any]:
        target = Path(root) if root is not None else self.cfg.workspace
        graph = index_repository(target, memory=self.engine.memory)
        self._repo_indexed = True
        return graph.summarize()

    def solve_problem(self, problem: BlindProblem, max_attempts: int = 5) -> dict[str, Any]:
        """Solve a structured BlindProblem (math or code) via SelfPlayEngine + Verifier."""
        return self.selfplay.solve(problem, self.solver, max_attempts=max_attempts)

    def repair_workspace(
        self,
        root: str | Path | None = None,
        *,
        tests_dir: str = "tests",
        max_verifier_calls: int = 80,
        apply: bool = True,
    ) -> dict[str, Any]:
        """Repair a failing multi-file Python workspace through verified AST mutation.

        The engine runs the workspace's own suite in an isolated copy. Nothing is
        written to ``root`` unless the suite passes, and every kept edit is
        recorded with the execution that accepted it.
        """
        from .brain.repair import RepairEngine

        target = Path(root) if root is not None else self.cfg.workspace
        report = RepairEngine(tests_dir=tests_dir, max_verifier_calls=max_verifier_calls).repair(
            target, apply=apply
        )
        return report.to_dict()

    def handle(self, goal: str, mode: str = "workspace", max_attempts: int = 5) -> dict[str, Any]:
        """Execute either a formal/coding problem or a durable workspace/memory task."""
        t0 = time.perf_counter()
        stmt, inline_tests = _extract_inline_tests(goal)

        # Route 1: verifiable mathematical / Python synthesis request
        if _CODE_SPEC_RE.search(stmt) and inline_tests:
            prob = BlindProblem(
                id=f"live-{int(t0 * 1000) % 1000000}",
                domain="code",
                title="Live Verification Task",
                difficulty=2,
                statement=stmt,
                verify=lambda code, T=inline_tests: CodeVerifier.verify_python(code, T),
            )
            res = self.selfplay.solve(prob, self.solver, max_attempts=max_attempts)
            return {
                "ok": bool(res.get("passed")),
                "domain": "code_synthesis",
                "tier": res.get("source", "search"),
                "verified": bool(res.get("passed")),
                "status": res.get("status", "UNVERIFIED"),
                "attempts": res.get("attempts", 0),
                "solution": res.get("solution", ""),
                "compiled_skill": res.get("compiled_skill"),
                "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "rss_mb": _rss_mb(),
            }

        # Route 2: codebase query triggers lazy workspace indexing on first ask
        if not self._repo_indexed and any(k in goal.casefold() for k in ("codebase", "symbol", "class ", "def ", "دالة", "كلاس")):
            if any(self.cfg.workspace.rglob("*.py")):
                self.index_codebase(self.cfg.workspace)

        # Route 3: durable checkpointed Engine task
        task = self.engine.submit(goal, mode=mode)
        if task.status.value == "planned":
            task = self.engine.run(task.id)
        return {
            "ok": task.status.value == "done",
            "domain": "workspace_agent",
            "task_id": task.id,
            "tier": task.tier,
            "status": task.status.value,
            "verified": task.status.value == "done",
            "summary": task.summary,
            "steps": [
                {"idx": s.idx, "tool": s.tool, "status": s.status.value, "output": s.output}
                for s in task.steps
            ],
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "rss_mb": _rss_mb(),
        }

    def sleep(self, budget_s: float = 6.0, steps_cap: int = 80, extra_frames: list[bytes] | None = None) -> dict[str, Any]:
        """Run Wake-Sleep replay distillation into this agent's compact v2 core."""
        ckpt_in = self.core_checkpoint if self.core_checkpoint.is_file() else None
        stats = sleep_distill(
            self.core_checkpoint,
            checkpoint_in=ckpt_in,
            skills=self.skills,
            memory=self.engine.memory,
            extra_frames=extra_frames,
            budget_s=budget_s,
            steps_cap=steps_cap,
        )
        # Refresh proposer so future calls pick up newly distilled weights cleanly
        self.proposer = NeuralCodeProposer(self.core_checkpoint)
        self.solver.neural_proposer = self.proposer
        return stats

    def status(self) -> dict[str, Any]:
        return {
            "home": str(self.cfg.home),
            "workspace": str(self.cfg.workspace),
            "memory": self.engine.memory.stats(),
            "brain": self.skills.stats(),
            "core_checkpoint": str(self.core_checkpoint),
            "core_ready": self.core_checkpoint.is_file(),
            "rss_mb": _rss_mb(),
        }
