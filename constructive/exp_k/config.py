"""EXP-K configuration, hardware disclosure, source fingerprints and hashing."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MANIFESTS_DIR = RESULTS_DIR / "manifests"
HISTORY_DIR = RESULTS_DIR / "history"

# Authoritative baseline manifest recorded by EXP-J at STAGE 0. EXP-K treats it
# as read-only and re-verifies it before and after every stage.
EXPJ_BASELINE = RESULTS_DIR.parent.parent / "exp_j" / "results" / "baseline_hashes.json"
EXPI_BASELINE = RESULTS_DIR.parent.parent / "exp_i" / "results" / "baseline_hashes.json"


@dataclass(frozen=True)
class ExpKConfig:
    """Bounded, fully disclosed execution policy."""

    seeds: tuple[int, ...] = tuple(range(10))
    n_val: int = 5
    n_train: int = 4
    n_test: int = 12
    abstain_threshold: float = 0.6
    search_beam: int = 24
    search_node_budget: int = 12_000
    max_cost: float = 4.0
    fit_deadline_s: float = 2.5
    experiment_deadline_s: float = 240.0
    campaign_deadline_s: float = 1800.0
    depths: tuple[int, ...] = (2, 3, 4, 5, 8, 12, 16)
    quick: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def quick_config() -> ExpKConfig:
    """Fast pilot: identical code paths, fewer seeds and episodes."""
    return ExpKConfig(
        seeds=(0, 1), n_val=3, n_train=3, n_test=8,
        search_node_budget=4_000, fit_deadline_s=1.5,
        experiment_deadline_s=90.0, campaign_deadline_s=600.0,
        quick=True,
    )


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_of(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=20)
        return (out.stdout or "").strip() or "no-commits"
    except Exception:  # pragma: no cover
        return "unknown"


def git_branch() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                             capture_output=True, text=True, timeout=20)
        return (out.stdout or "").strip() or "unknown"
    except Exception:  # pragma: no cover
        return "unknown"


def probe_hardware() -> dict[str, Any]:
    ram_gb = None
    try:
        import ctypes

        class _Mem(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        m = _Mem()
        m.dwLength = ctypes.sizeof(_Mem)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        ram_gb = round(m.ullTotalPhys / 1e9, 2)
    except Exception:
        pass
    cpu = platform.processor() or "unknown"
    return {
        "cpu": cpu, "logical_cores": os.cpu_count(), "ram_gb": ram_gb,
        "os": platform.platform(), "device": "cpu",
        "acceleration": "none (CPU-only; no external APIs)",
        "python": sys.version.split()[0],
    }


def source_fingerprints() -> dict[str, str]:
    """Hashes of the read-only systems under test and EXP-K baseline manifest."""
    root = RESULTS_DIR.resolve().parents[2]
    files = [
        "constructive/exp_j/generators.py", "constructive/exp_j/deep_composition.py",
        "constructive/exp_j/transfer.py", "constructive/exp_j/robustness.py",
        "constructive/exp_j/calibration.py" if (root / "constructive/exp_j/calibration.py").exists()
        else "constructive/exp_j/metrics.py",
        "constructive/exp_i/dsl.py", "constructive/exp_i/search.py",
        "constructive/exp_i/retrieval.py", "constructive/exp_i/validation.py",
    ]
    return {f: sha256_file(root / f) for f in files if (root / f).exists()}


def reproducibility(cfg: ExpKConfig) -> dict[str, Any]:
    return {
        "config_fingerprint": fingerprint_of(cfg.as_dict()),
        "configuration": cfg.as_dict(),
        "hardware": probe_hardware(),
        "git_commit": git_commit(),
        "git_branch": git_branch(),
        "source_fingerprints": source_fingerprints(),
        "note": "Systems under test (EXP-I/EXP-J) are imported read-only.",
    }
