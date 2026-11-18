"""CPU-only resource accounting for matched comparisons."""

from __future__ import annotations

import os
import threading
import time
from typing import Any

try:  # optional; core experiments do not require it
    import psutil
except Exception:  # pragma: no cover
    psutil = None  # type: ignore


class ResourceMeter:
    """Sample wall time, process CPU time, and peak resident memory."""

    def __init__(self, sample_interval_s: float = 0.01) -> None:
        self.interval = sample_interval_s
        self._proc = psutil.Process(os.getpid()) if psutil else None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_rss_mb = 0.0

    def _rss_mb(self) -> float:
        if self._proc is None:
            return 0.0
        return self._proc.memory_info().rss / (1024 ** 2)

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak_rss_mb = max(self.peak_rss_mb, self._rss_mb())
            except Exception:  # pragma: no cover
                pass
            self._stop.wait(self.interval)

    def __enter__(self) -> "ResourceMeter":
        self.wall_start = time.perf_counter()
        self.cpu_start = time.process_time()
        self.peak_rss_mb = self._rss_mb()
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.peak_rss_mb = max(self.peak_rss_mb, self._rss_mb())
        self.wall_s = time.perf_counter() - self.wall_start
        self.cpu_s = time.process_time() - self.cpu_start

    def result(self) -> dict[str, float]:
        return {
            "wall_s": round(getattr(self, "wall_s", 0.0), 6),
            "cpu_s": round(getattr(self, "cpu_s", 0.0), 6),
            "peak_rss_mb": round(self.peak_rss_mb, 3),
        }
