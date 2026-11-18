"""Deterministic dataset fixtures.

Stage L0 uses only ``minimal_fixture`` as a smoke fixture.  Later stages add
their own generators here, all derived from ``make_rng(seed, stream)``.
"""

from __future__ import annotations

from typing import Any

from .determinism import fingerprint, make_rng


def minimal_fixture(seed: int, n: int = 12) -> list[dict[str, Any]]:
    """A tiny, fully deterministic (x, y) fixture: y = (2*x + 3) mod 7."""
    rng = make_rng(seed, "minimal-fixture")
    rows = []
    for i in range(n):
        x = rng.randint(0, 1000)
        rows.append({"i": i, "x": x, "y": (2 * x + 3) % 7})
    return rows


def minimal_fingerprint(seed: int, n: int = 12) -> str:
    return fingerprint({"generator": "minimal_fixture/v1", "seed": seed,
                        "n": n, "rows": minimal_fixture(seed, n)})
