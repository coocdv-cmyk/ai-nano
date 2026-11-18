"""Deterministic primitives for EXP-L.

Experiment randomness is never drawn from Python's randomized string hash, the
process environment, clocks, or unordered containers.  Every generator is
derived from an explicit (seed, stream) pair via SHA-256, so a fixture produced
in one process is byte-identical to one produced in another.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def stable_json(obj: Any) -> str:
    """Canonical JSON with sorted keys, compact separators, str fallback."""
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def full_hash(obj: Any) -> str:
    return hashlib.sha256(stable_json(obj).encode("utf-8")).hexdigest()


def fingerprint(obj: Any) -> str:
    """Short deterministic fingerprint (first 16 hex chars of SHA-256)."""
    return full_hash(obj)[:16]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def make_rng(seed: int, stream: str = "") -> random.Random:
    """A reproducible RNG keyed by (seed, stream), independent of hash()."""
    digest = hashlib.sha256(f"{int(seed)}|{stream}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def sample_seeds(n: int, base: int = 0) -> tuple[int, ...]:
    return tuple(base + i for i in range(n))
