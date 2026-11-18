"""EXP-K resource governance and measurement (CPU-only)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from constructive.exp_i.resources import ResourceGovernor, ResourceLimit, run_isolated
from constructive.exp_i.resources import _peak_rss_mb

from .types import Completion


@dataclass
class ResourceSample:
    elapsed_s: float = 0.0
    peak_rss_mb: float | None = None
    expansions: int = 0
    nodes: int = 0
    validation_calls: int = 0
    candidates_evaluated: int = 0
    timed_out: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "elapsed_s": round(self.elapsed_s, 6),
            "peak_rss_mb": self.peak_rss_mb,
            "expansions": self.expansions, "nodes": self.nodes,
            "validation_calls": self.validation_calls,
            "candidates_evaluated": self.candidates_evaluated,
            "timed_out": self.timed_out, "notes": list(self.notes),
        }


class Monitor:
    def __init__(self) -> None:
        self._start = time.perf_counter()
        self.sample = ResourceSample()
        self.completion = Completion.COMPLETED

    def charge(self, *, expansions: int = 0, nodes: int = 0,
               validation_calls: int = 0, candidates: int = 0) -> None:
        self.sample.expansions += expansions
        self.sample.nodes += nodes
        self.sample.validation_calls += validation_calls
        self.sample.candidates_evaluated += candidates

    def mark_timeout(self, detail: str) -> None:
        self.sample.timed_out = True
        self.completion = Completion.TIMED_OUT
        self.sample.notes.append(f"timeout: {detail}")

    def mark_error(self, detail: str) -> None:
        self.completion = Completion.ERROR
        self.sample.notes.append(f"error: {detail}")

    def finish(self) -> dict[str, Any]:
        self.sample.elapsed_s = time.perf_counter() - self._start
        self.sample.peak_rss_mb = _peak_rss_mb()
        return {"resources": self.sample.as_dict(),
                "completion_status": self.completion.value}


__all__ = ["ResourceGovernor", "ResourceLimit", "run_isolated", "Monitor", "ResourceSample"]
