"""Resource profiling with explicit, honest measurement boundaries."""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

T = TypeVar("T")


@dataclass
class ProfileResult:
    value: Any
    wall_s: float
    traced_peak_bytes: int | None
    nodes: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "wall_s": round(self.wall_s, 6),
            "traced_peak_bytes": self.traced_peak_bytes,
            "nodes": self.nodes,
        }


def profile_region(fn: Callable[[], T], *, count_nodes: Callable[[Any], int] | None = None
                   ) -> ProfileResult:
    """Run ``fn`` and report wall time, traced peak heap and (optionally) nodes.

    ``tracemalloc`` measures *Python-allocated* bytes only; it under-reports
    native allocations and says nothing about OS RSS.  Where a process-wide
    peak matters, the runner additionally records OS counters.  Unavailable
    measurements are reported as ``None``, never as zero.
    """
    tracemalloc.start()
    try:
        t0 = time.perf_counter()
        value = fn()
        dt = time.perf_counter() - t0
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    nodes = count_nodes(value) if count_nodes else None
    return ProfileResult(value=value, wall_s=dt, traced_peak_bytes=int(peak), nodes=nodes)
