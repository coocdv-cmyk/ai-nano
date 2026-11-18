"""The OMEGA-NANO control loop.

receive -> understand -> plan (adaptive tier) -> for each step:
    permission check -> write-ahead checkpoint -> bounded execution ->
    observe -> independent verification -> retry / revise / escalate ->
    checkpoint
-> report, learn (procedural plan cache + episodic memory).

Crash safety: a step is marked STARTED and checkpointed *before* it runs. On
resume, a STARTED step whose postconditions can be checked from the world
alone is re-verified first and marked done without re-execution if the
effect already happened; a non-idempotent STARTED step is escalated to a
human instead of being blindly repeated.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from typing import Any

from .config import Config
from .memory import MemoryFabric, normalize
from .model import Step, StepSpec, StepStatus, Task, TaskStatus
from .reasoner import LocalLLMReasoner, Reasoner, RuleReasoner
from .security import Decision, PathEscape, decide
from .store import Store
from .tools.base import ToolContext, ToolError, ToolRegistry, default_registry
from .verify import verify, world_checks

TEMPLATE = re.compile(r"\$\{(prev|s(\d+))\.([\w.]+)\}")


class Unresolved(ToolError):
    pass


def pid_alive(pid: int) -> bool:
    """Liveness probe that never signals the process (os.kill on Windows would terminate it)."""
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not h:
            return False
        try:
            return k32.WaitForSingleObject(h, 0) == 0x102  # WAIT_TIMEOUT: still running
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _fault(point: str, idx: int) -> None:
    """Deterministic crash injection for recovery tests: OMEGA_FAULT=crash_after_start:1"""
    spec = os.environ.get("OMEGA_FAULT")
    if spec and spec == f"{point}:{idx}":
        os._exit(77)


class Engine:
    def __init__(self, cfg: Config, tools: ToolRegistry | None = None, reasoner: Reasoner | None = None,
                 llm: Reasoner | None = None):
        self.cfg = cfg
        self.store = Store(cfg.db_path)
        self.memory = MemoryFabric(cfg.home / "memory.db")
        self.tools = tools or default_registry()
        self.rules = reasoner or RuleReasoner()
        self.llm = llm or (LocalLLMReasoner(cfg.llm_url, cfg.llm_model) if cfg.llm_url else None)
        self.ctx = ToolContext(workspace=cfg.workspace.resolve(), home=cfg.home, memory=self.memory, config=cfg)

    def close(self) -> None:
        self.store.close()
        self.memory.close()

    # -- intake ----------------------------------------------------------------

    def submit(self, goal: str, mode: str = "workspace", idem_key: str | None = None,
               plan: list[dict] | None = None) -> Task:
        if idem_key:
            existing = self.store.find_by_key(idem_key)
            if existing:
                self.store.log(existing, "duplicate_delivery", {"idem_key": idem_key})
                return self.store.load(existing)[0]
        from .security import MODES
        if mode not in MODES:
            raise ValueError(f"mode must be one of {sorted(MODES)}")
        task = Task(id=uuid.uuid4().hex[:12], goal=goal, mode=mode, idem_key=idem_key)
        self.store.save(task, "received", {"goal": goal, "mode": mode})
        self._plan(task, explicit=plan)
        return task

    def _plan(self, task: Task, explicit: list[dict] | None = None) -> None:
        t0 = time.perf_counter()
        sig = hashlib.sha256(normalize(task.goal).encode("utf-8")).hexdigest()
        specs: list[StepSpec] = []
        und: dict[str, Any] = {}
        if explicit is not None:
            specs = [StepSpec(**s) for s in explicit]
            und = {"objective": task.goal, "reasoner": "explicit-plan", "confidence": 1.0, "unparsed": []}
            task.tier = "explicit"
        elif (cached := self.store.cache_get(sig)) is not None:
            specs = [StepSpec(**s) for s in cached]
            und = {"objective": task.goal, "reasoner": "plan-cache", "confidence": 1.0, "unparsed": []}
            task.tier = "t0:plan-cache"
        else:
            und, specs = self.rules.plan(task.goal, self.tools)
            task.tier = "t1:rules"
            if (und.get("unparsed") or not specs) and self.llm is not None:
                try:
                    und, specs = self.llm.plan(task.goal, self.tools)
                    task.tier = "t2:llm"
                except (OSError, ValueError, KeyError) as e:
                    und["llm_error"] = str(e)
        und["signature"] = sig
        und["plan_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        task.understanding = und
        problems = self._validate_plan(specs)
        if und.get("unparsed") or not specs or problems:
            task.transition(TaskStatus.BLOCKED)
            why = problems or [f"could not understand: {c!r}" for c in und.get("unparsed", [])] or ["empty plan"]
            task.summary = "Blocked at planning: " + "; ".join(why) + (
                "" if self.llm else " (no LLM backend configured; set OMEGA_LLM_URL for open-ended goals)")
            self.store.save(task, "blocked", {"reasons": why})
            return
        task.steps = [Step.from_spec(i, s) for i, s in enumerate(specs)]
        task.transition(TaskStatus.PLANNED)
        self.store.save(task, "planned", {"tier": task.tier, "steps": len(specs), "plan_ms": und["plan_ms"]})

    def _validate_plan(self, specs: list[StepSpec]) -> list[str]:
        problems = []
        for i, s in enumerate(specs):
            if s.tool not in self.tools:
                problems.append(f"step {i}: unknown tool {s.tool!r}")
                continue
            try:
                self.tools.get(s.tool).validate(s.args)
            except ToolError as e:
                problems.append(f"step {i}: {e}")
        return problems

    # -- control ---------------------------------------------------------------

    def stop(self) -> None:
        self.cfg.stop_file.write_text(str(time.time()), encoding="utf-8")

    def clear_stop(self) -> None:
        self.cfg.stop_file.unlink(missing_ok=True)

    def approve(self, task_id: str, deny: bool = False) -> Task:
        task, _ = self.store.load(task_id)
        waiting = [s for s in task.steps if s.status in (StepStatus.AWAITING_APPROVAL, StepStatus.STARTED)]
        if task.status not in (TaskStatus.AWAITING_APPROVAL, TaskStatus.BLOCKED) or not waiting:
            raise ValueError(f"task {task.id} is not waiting for approval (status {task.status.value})")
        step = waiting[0]
        if deny:
            step.status = StepStatus.DENIED
            task.transition(TaskStatus.FAILED)
            task.summary = f"Step {step.idx} ({step.tool}) was denied by the user."
            self.store.save(task, "denied", {"step": step.idx})
            return task
        step.approved = True
        step.status = StepStatus.PENDING
        task.transition(TaskStatus.RUNNING)
        self.store.save(task, "approved", {"step": step.idx, "tool": step.tool})
        return self.run(task.id)

    def resume_all(self) -> list[Task]:
        """Resume every task interrupted by a crash or restart (expired lease) or planned but not started."""
        out = []
        now = time.time()
        for row in self.store.list_tasks({TaskStatus.RUNNING, TaskStatus.PLANNED, TaskStatus.PAUSED}):
            task, _ = self.store.load(row["id"])
            if task.status == TaskStatus.RUNNING and task.lease_until > now and \
                    task.lease_pid != os.getpid() and pid_alive(task.lease_pid):
                continue  # another live worker holds it
            if task.status == TaskStatus.PAUSED and self.cfg.stop_file.exists():
                continue
            out.append(self.run(task.id))
        return out

    # -- execution -------------------------------------------------------------

    def run(self, task_id: str) -> Task:
        task, recovered = self.store.load(task_id)
        if task.status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.BLOCKED, TaskStatus.AWAITING_APPROVAL):
            if recovered:
                self.store.save(task, "restored", {})  # heal the corrupted current state
            return task  # terminal, or waiting on a human: only approve() moves these forward
        if recovered:
            # State came from an older checkpoint: don't trust its DONE marks, re-verify them.
            for s in task.steps:
                if s.status == StepStatus.DONE:
                    s.status = StepStatus.STARTED
        if task.status != TaskStatus.RUNNING:
            task.transition(TaskStatus.RUNNING)
        self._lease(task)
        self.store.save(task, "running", {"cursor": task.cursor})

        i = 0
        while i < len(task.steps):
            step = task.steps[i]
            if step.status == StepStatus.DONE:
                i += 1
                continue
            if self.cfg.stop_file.exists():
                task.transition(TaskStatus.PAUSED)
                self.store.save(task, "paused", {"reason": "emergency stop", "step": step.idx})
                return task
            outcome = self._step(task, step)
            if outcome == "done":
                i += 1
            elif outcome == "revised":
                continue  # step list changed in place; re-evaluate index i
            else:
                return task
        task.transition(TaskStatus.DONE)
        task.summary = self._summary(task)
        task.lease_until = 0.0
        self.store.save(task, "done", {"steps": len(task.steps)})
        self._learn(task)
        return task

    def _lease(self, task: Task) -> None:
        task.lease_until = time.time() + self.cfg.lease_s
        task.lease_pid = os.getpid()

    def _step(self, task: Task, step: Step) -> str:
        tool = self.tools.get(step.tool)
        try:
            args = tool.validate(self._resolve(step.args, task, step.idx))
            expect = self._resolve_expect(step.expect, task, step.idx, args)
            scopes = tool.needed_scopes(self.ctx, args)
        except (ToolError, PathEscape) as e:
            return self._fail(task, step, f"invalid step: {e}")

        if step.status == StepStatus.STARTED:
            # We crashed (or were restored) after the write-ahead marker.
            checks = world_checks(expect)
            if checks:
                ok, ver = verify(self.ctx, checks, None)
                if ok:
                    step.status, step.recovered, step.verification = StepStatus.DONE, True, ver
                    step.ended_at = time.time()
                    self.store.save(task, "recovered_without_reexecution", {"step": step.idx})
                    return "done"
            if not tool.is_idempotent(self.ctx, args) and not step.approved:
                task.transition(TaskStatus.BLOCKED)
                task.summary = (f"Step {step.idx} ({step.tool}) may have run before an interruption and is not "
                                f"safe to repeat automatically. Approve to re-run it.")
                self.store.save(task, "uncertain_side_effect", {"step": step.idx})
                return "stop"
            self.store.log(task.id, "reexecute_after_restart", {"step": step.idx})

        decision = decide(task.mode, scopes)
        if decision is Decision.DENY:
            return self._fail(task, step, f"permission denied: mode '{task.mode}' forbids "
                                          f"{sorted(s.value for s in scopes)}")
        if decision is Decision.ASK and not step.approved:
            step.status = StepStatus.AWAITING_APPROVAL
            task.transition(TaskStatus.AWAITING_APPROVAL)
            task.summary = f"Waiting for approval: step {step.idx} {step.tool} {json.dumps(args, ensure_ascii=False)[:200]}"
            self.store.save(task, "approval_required", {"step": step.idx, "scopes": sorted(s.value for s in scopes)})
            return "stop"

        while True:
            step.status = StepStatus.STARTED
            step.attempts += 1
            step.started_at = time.time()
            self._lease(task)
            self.store.save(task, "step_started", {"step": step.idx, "tool": step.tool, "attempt": step.attempts})
            _fault("crash_after_start", step.idx)

            output, error, permanent = self._execute(tool, args)
            _fault("crash_after_exec", step.idx)
            ok, ver = verify(self.ctx, expect, output)
            step.output, step.error, step.verification = _trim(output), error, ver
            step.ended_at = time.time()
            if ok:
                step.status = StepStatus.DONE
                self.store.save(task, "step_verified", {"step": step.idx, "attempt": step.attempts})
                return "done"
            self.store.save(task, "step_failed_verification",
                            {"step": step.idx, "attempt": step.attempts, "error": error,
                             "failed": [v["detail"] for v in ver if not v["passed"]]})
            if permanent or not tool.is_idempotent(self.ctx, args) or step.attempts > self.cfg.max_retries:
                break
            time.sleep(min(self.cfg.backoff_max_s, self.cfg.backoff_base_s * 2 ** (step.attempts - 1)))

        if task.revisions < self.cfg.max_revisions:
            specs = self.rules.revise(task.goal, task.steps, step, self.tools, self.ctx.workspace)
            if specs is None and self.llm is not None:
                specs = self.llm.revise(task.goal, task.steps, step, self.tools, self.ctx.workspace)
            if specs and not self._validate_plan(specs):
                task.revisions += 1
                new = [Step.from_spec(0, s) for s in specs]
                task.steps[step.idx:step.idx + 1] = new
                for n, s in enumerate(task.steps):
                    s.idx = n
                self.store.save(task, "revised", {"replaced_step": step.idx, "with": [s.tool for s in new],
                                                  "why": [s.why for s in new]})
                return "revised"
        return self._fail(task, step, error or "postcondition not met: " +
                          "; ".join(v["detail"] for v in ver if not v["passed"]))

    def _execute(self, tool, args) -> tuple[dict | None, str | None, bool]:
        """Run one tool call with a wall-clock bound. Returns (output, error, permanent_failure)."""
        box: dict[str, Any] = {}

        def target() -> None:
            try:
                box["out"] = tool.run(self.ctx, args)
            except (ToolError, PathEscape) as e:
                box["err"], box["perm"] = str(e), True
            except Exception as e:  # tool bugs are observations, never agent crashes
                box["err"], box["perm"] = f"{type(e).__name__}: {e}", False

        th = threading.Thread(target=target, daemon=True)
        th.start()
        th.join(self.cfg.step_timeout_s)
        if th.is_alive():
            return {"ok": False, "error": "timeout", "timed_out": True}, f"timeout after {self.cfg.step_timeout_s}s", False
        if "err" in box:
            return {"ok": False, "error": box["err"]}, box["err"], box["perm"]
        out = box["out"]
        return out, (None if out.get("ok") else str(out.get("error"))), False

    def _fail(self, task: Task, step: Step, reason: str) -> str:
        step.status = StepStatus.FAILED
        step.error = reason
        task.transition(TaskStatus.FAILED)
        task.summary = f"Failed at step {step.idx} ({step.tool}): {reason}"
        task.lease_until = 0.0
        self.store.save(task, "failed", {"step": step.idx, "reason": reason})
        if task.tier == "t0:plan-cache":
            self.store.cache_drop(task.understanding.get("signature", ""))
        return "stop"

    # -- templates ---------------------------------------------------------------

    def _ref(self, task: Task, idx: int, m: re.Match) -> str:
        src = idx - 1 if m.group(1) == "prev" else int(m.group(2))
        if not 0 <= src < idx:
            raise Unresolved(f"reference {m.group(0)} does not point to an earlier step")
        out = task.steps[src].output or {}
        cur: Any = out
        for part in m.group(3).split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                raise Unresolved(f"reference {m.group(0)}: field not found in step {src} output")
        return cur if isinstance(cur, str) else json.dumps(cur, ensure_ascii=False)

    def _resolve(self, value: Any, task: Task, idx: int) -> Any:
        if isinstance(value, str):
            return TEMPLATE.sub(lambda m: self._ref(task, idx, m), value)
        if isinstance(value, dict):
            return {k: self._resolve(v, task, idx) for k, v in value.items()}
        if isinstance(value, list):
            return [self._resolve(v, task, idx) for v in value]
        return value

    def _resolve_expect(self, expect: list[dict], task: Task, idx: int, args: dict) -> list[dict]:
        out = []
        content = args.get("content")
        for e in self._resolve(expect, task, idx):
            e = dict(e)
            for k, v in e.items():
                if v == "@content_sha256" and isinstance(content, str):
                    e[k] = hashlib.sha256(content.encode("utf-8")).hexdigest()
                elif v == "@content" and isinstance(content, str):
                    e[k] = content
            out.append(e)
        return out

    # -- learning & reporting --------------------------------------------------------

    def _learn(self, task: Task) -> None:
        sig = task.understanding.get("signature")
        if sig and task.tier in ("t1:rules", "t2:llm"):
            self.store.cache_put(sig, [{"tool": s.tool, "args": s.args, "expect": s.expect, "why": s.why}
                                       for s in task.steps])
        try:
            self.memory.remember(f"Completed task: {task.goal}", kind="episodic", source=f"task:{task.id}",
                                 confidence=1.0, meta={"steps": [s.tool for s in task.steps], "tier": task.tier})
        except ValueError:
            pass

    @staticmethod
    def _summary(task: Task) -> str:
        parts = []
        for s in task.steps:
            o = s.output or {}
            shown = next(
                (o[k] for k in ("value", "answer", "completion", "path", "count") if o.get(k) is not None),
                None,
            )
            parts.append(f"{s.tool}" + (f" -> {str(shown)[:80]}" if shown is not None else ""))
        rec = sum(1 for s in task.steps if s.recovered)
        return f"All {len(task.steps)} step(s) verified: " + "; ".join(parts) + (
            f" ({rec} recovered after restart)" if rec else "")


def _trim(output: dict | None, limit: int = 8000) -> dict | None:
    if output is None:
        return None
    out = {}
    for k, v in output.items():
        if isinstance(v, str) and len(v) > limit:
            out[k] = v[:limit]
            out[k + "_truncated"] = True
        else:
            out[k] = v
    return out
