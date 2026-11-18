"""EXP-J resource governance and measurement (CPU-only).

Reuses the audited EXP-I governor as a primitive but adds EXP-J-specific
measurement: elapsed time, search expansions, verification cost and peak RSS.
All budgets are hard caps; a timeout is always recorded as TIMED_OUT, never as
a successful completion.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from constructive.exp_i.resources import ResourceGovernor, ResourceLimit, run_isolated

from .types import CompletionState


@dataclass
class ResourceSample:
    elapsed_s: float = 0.0
    peak_rss_mb: float | None = None
    expansions: int = 0
    nodes: int = 0
    cache_hits: int = 0
    verification_calls: int = 0
    memory_per_concept_bytes: float | None = None
    timed_out: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "elapsed_s": round(self.elapsed_s, 6),
            "peak_rss_mb": self.peak_rss_mb,
            "expansions": self.expansions,
            "nodes": self.nodes,
            "cache_hits": self.cache_hits,
            "verification_calls": self.verification_calls,
            "memory_per_concept_bytes": self.memory_per_concept_bytes,
            "timed_out": self.timed_out,
            "notes": list(self.notes),
        }


class Monitor:
    """Lightweight cumulative resource monitor with structured completion."""

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self.sample = ResourceSample()
        self.completion = CompletionState.COMPLETED

    def charge(self, *, expansions: int = 0, nodes: int = 0, cache_hits: int = 0,
               verification_calls: int = 0) -> None:
        self.sample.expansions += expansions
        self.sample.nodes += nodes
        self.sample.cache_hits += cache_hits
        self.sample.verification_calls += verification_calls

    def mark_timeout(self, detail: str) -> None:
        self.sample.timed_out = True
        self.completion = CompletionState.TIMED_OUT
        self.sample.notes.append(f"timeout: {detail}")

    def mark_error(self, detail: str) -> None:
        self.completion = CompletionState.ERROR
        self.sample.notes.append(f"error: {detail}")

    def finish(self) -> dict[str, Any]:
        self.sample.elapsed_s = time.perf_counter() - self._start
        from constructive.exp_i.resources import _peak_rss_mb

        self.sample.peak_rss_mb = _peak_rss_mb()
        return {"resources": self.sample.as_dict(),
                "completion_status": self.completion.value}


def guarded(governor: ResourceGovernor, name: str, fn: Callable[[], Any], *,
            deadline_s: float | None = None) -> Any:
    """Run fn under a governor scope with deterministic error isolation."""
    with governor.scope(name, deadline_s=deadline_s):
        governor.checkpoint(name)
        return fn()


__all__ = ["ResourceGovernor", "ResourceLimit", "run_isolated",
           "Monitor", "ResourceSample", "guarded"]
