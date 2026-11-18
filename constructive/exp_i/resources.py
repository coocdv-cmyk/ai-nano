"""Central resource governance for EXP-I (EXP-I-H).

Every long-running operation checks a governor checkpoint. Exceeding any budget
raises :class:`ResourceLimit`, which is captured by the runner and recorded with
an explicit non-success :class:`CompletionStatus`. A timeout can never silently
become a valid partial *success*.
"""

from __future__ import annotations

import gc
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable

from .types import CompletionStatus


class ResourceLimit(Exception):
    """Raised when a deadline, node budget or cancellation is hit."""

    def __init__(self, kind: str, detail: str):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail


@dataclass
class TimeoutReport:
    scope: str
    kind: str
    elapsed_s: float
    deadline_s: float | None
    nodes_used: int
    node_budget: int | None
    partial_diagnostics: dict[str, Any] = field(default_factory=dict)
    status: CompletionStatus = CompletionStatus.TIMED_OUT


class ResourceGovernor:
    """In-process deadlines, node budgets, cancellation and cleanup.

    Parameters may be ``None`` to disable a limit. Nested scopes are supported:
    each scope gets its own deadline derived from the minimum of its explicit
    deadline and the remaining parent deadline.
    """

    def __init__(
        self,
        global_deadline_s: float | None = None,
        node_budget: int | None = None,
        memory_guard_mb: float | None = None,
    ) -> None:
        self._start = time.monotonic()
        self.global_deadline_s = global_deadline_s
        self.node_budget = node_budget
        self.memory_guard_mb = memory_guard_mb
        self.nodes_used = 0
        self._cancelled = False
        self._scopes: list[dict[str, Any]] = []
        self.timeouts: list[TimeoutReport] = []
        self.diagnostics: dict[str, Any] = {}

    # ---- clock / deadlines -------------------------------------------------
    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self._start

    def remaining_s(self) -> float | None:
        if self.global_deadline_s is None:
            return None
        return max(0.0, self.global_deadline_s - self.elapsed_s)

    def _effective_deadline(self, local: float | None) -> float | None:
        rem = self.remaining_s()
        candidates = [d for d in (local, rem) if d is not None]
        return min(candidates) if candidates else None

    # ---- control -----------------------------------------------------------
    def cancel(self, reason: str = "explicit cancellation") -> None:
        self._cancelled = True
        self.diagnostics["cancel_reason"] = reason

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def charge_nodes(self, n: int = 1) -> None:
        self.nodes_used += n

    def set_diagnostics(self, **kw: Any) -> None:
        self.diagnostics.update(kw)

    # ---- checkpoint --------------------------------------------------------
    def checkpoint(self, scope: str = "operation") -> None:
        if self._cancelled:
            raise ResourceLimit("cancellation", self.diagnostics.get("cancel_reason", "cancelled"))
        rem = self.remaining_s()
        if rem is not None and rem <= 0.0:
            self._record_timeout(scope, "global_deadline", self.global_deadline_s)
            raise ResourceLimit("global_deadline", f"{scope} exceeded global deadline")
        if self._scopes:
            active = self._scopes[-1]
            local = active.get("deadline_s")
            if local is not None:
                left = local - (time.monotonic() - active["start"])
                if left <= 0.0:
                    self._record_timeout(scope, "scope_deadline", local)
                    raise ResourceLimit("scope_deadline", f"{scope} exceeded local deadline")
            budget = active.get("node_budget")
            if budget is not None and self.nodes_used >= budget:
                self._record_timeout(scope, "node_budget", float(budget))
                raise ResourceLimit("node_budget", f"{scope} exhausted node budget")
        if self.node_budget is not None and self.nodes_used >= self.node_budget:
            self._record_timeout(scope, "node_budget", float(self.node_budget))
            raise ResourceLimit("node_budget", f"{scope} exhausted global node budget")
        if self.memory_guard_mb is not None:
            used = _rss_mb()
            if used is not None and used > self.memory_guard_mb:
                self._record_timeout(scope, "memory_guard", self.memory_guard_mb,
                                     {"rss_mb": used})
                raise ResourceLimit("memory_guard", f"{scope} exceeded memory guard")
        # last scope (innermost) sees the checkpoint
        if self._scopes:
            self._scopes[-1]["last_scope"] = scope

    def _record_timeout(self, scope: str, kind: str, deadline: float | None,
                        extra: dict[str, Any] | None = None) -> None:
        diag = dict(self.diagnostics)
        if extra:
            diag.update(extra)
        self.timeouts.append(TimeoutReport(
            scope=scope, kind=kind, elapsed_s=self.elapsed_s,
            deadline_s=deadline, nodes_used=self.nodes_used,
            node_budget=self.node_budget, partial_diagnostics=diag,
            status=CompletionStatus.TIMED_OUT,
        ))

    # ---- scoped execution --------------------------------------------------
    @contextmanager
    def scope(self, name: str, *, deadline_s: float | None = None,
              node_budget: int | None = None):
        """Enter a governed scope; deterministic cleanup on exit."""
        effective = self._effective_deadline(deadline_s)
        frame = {
            "name": name, "deadline_s": effective,
            "node_budget": node_budget, "start": time.monotonic(),
            "node_start": self.nodes_used, "last_scope": name,
        }
        self._scopes.append(frame)
        status = CompletionStatus.COMPLETED
        try:
            yield self
        except ResourceLimit as exc:
            status = CompletionStatus.TIMED_OUT
            self.set_diagnostics(last_timeout=exc.kind, last_scope=name)
            raise
        except BaseException as exc:  # failure isolation: record, never fabricate success
            status = CompletionStatus.ERROR
            self.set_diagnostics(last_error=f"{type(exc).__name__}: {exc}", last_scope=name)
            raise
        finally:
            self._scopes.pop()
            self.set_diagnostics(
                last_completed_scope=name,
                last_scope_status=status.value,
                scope_elapsed_s=time.monotonic() - frame["start"],
                scope_nodes=self.nodes_used - frame["node_start"],
            )
            gc.collect()

    def run(self, name: str, fn: Callable[[], Any], *, deadline_s: float | None = None,
            node_budget: int | None = None) -> Any:
        with self.scope(name, deadline_s=deadline_s, node_budget=node_budget):
            self.checkpoint(name)
            return fn()

    def report(self) -> dict[str, Any]:
        return {
            "elapsed_s": self.elapsed_s,
            "nodes_used": self.nodes_used,
            "global_deadline_s": self.global_deadline_s,
            "node_budget": self.node_budget,
            "cancelled": self._cancelled,
            "timeouts": [t.__dict__ for t in self.timeouts],
            "diagnostics": dict(self.diagnostics),
            "memory_guard_mb": self.memory_guard_mb,
            "peak_rss_mb": _peak_rss_mb(),
        }


def _memory_mb() -> tuple[float | None, float | None]:
    """Return (current_rss_mb, peak_rss_mb); None where unavailable."""
    try:
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("Psapi", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(),
                                          ctypes.byref(pmc), pmc.cb):
            return None, None
        return (round(pmc.WorkingSetSize / 1e6, 2),
                round(pmc.PeakWorkingSetSize / 1e6, 2))
    except Exception:
        try:
            import resource

            peak = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2)
            return peak, peak
        except Exception:
            return None, None


def _rss_mb() -> float | None:
    return _memory_mb()[0]


def _peak_rss_mb() -> float | None:
    return _memory_mb()[1]


def run_isolated(fn: Callable[[], Any], default: Any = None, *,
                 log: list[dict[str, Any]] | None = None) -> Any:
    """Run ``fn``, returning ``default`` and preserving a diagnostic on failure.

    Failure isolation: one broken experiment never crashes the campaign, and the
    failure is recorded rather than silently converted to a (false) success.
    """
    try:
        return fn()
    except BaseException as exc:  # noqa: BLE001 - deliberate campaign-level isolation
        entry = {"error": f"{type(exc).__name__}: {exc}"}
        if log is not None:
            log.append(entry)
        return default
