"""EXP-I configuration, hardware disclosure and hashing helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class ExpIConfig:
    """Bounded execution policy. Deliberately explicit about information budgets."""

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    n_episodes: int = 30           # principal-family default
    n_val: int = 4                 # multi-sample validation budget
    abstain_threshold: float = 0.6  # chosen/checked against calibration, not arbitrary
    max_program_depth: int = 8
    max_program_nodes: int = 64
    max_int_magnitude: int = 10 ** 9
    max_steps: int = 10_000
    search_node_budget: int = 20_000
    fit_deadline_s: float = 3.0
    experiment_deadline_s: float = 120.0
    campaign_deadline_s: float = 900.0
    quick: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def probe_hardware() -> dict[str, Any]:
    """Honest hardware disclosure (CPU-only machine; no external acceleration)."""
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
    try:
        import subprocess

        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"],
            capture_output=True, text=True, timeout=20,
        )
        name = (out.stdout or "").strip()
        if name:
            cpu = name
    except Exception:
        pass
    device = "cpu"
    try:
        import torch  # noqa: F401  presence check only

        if bool(torch.cuda.is_available()):
            device = "cuda-present"
    except Exception:
        pass
    return {
        "hardware": {
            "cpu": cpu,
            "logical_cores": os.cpu_count(),
            "ram_gb": ram_gb,
            "os": platform.platform(),
            "cuda": device == "cuda-present",
            "device": "cpu",
            "acceleration": "none (CPU-only, no external APIs)",
        },
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "exp_i": "0.1.0",
            "numpy": _numpy_version(),
        },
    }


def _numpy_version() -> str | None:
    try:
        import numpy

        return numpy.__version__
    except Exception:
        return None


def fingerprint_of(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def source_revision() -> str:
    try:
        import subprocess

        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=20)
        rev = (out.stdout or "").strip()
        return rev or "no-commits"
    except Exception:
        return "unknown"


def reproducibility_metadata(cfg: ExpIConfig) -> dict[str, Any]:
    hw = probe_hardware()
    return {
        "config_fingerprint": fingerprint_of(cfg.as_dict()),
        "configuration": cfg.as_dict(),
        "hardware": hw["hardware"],
        "software": hw["software"],
        "source_revision": source_revision(),
        "note": "No commits in repository; source_revision='no-commits' is expected.",
    }
