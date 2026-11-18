"""EXP-H configuration and hardware disclosure."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class ExpHConfig:
    """Bounded execution policy for the whole EXP-H campaign."""

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    quick: bool = False
    # CCM search caps used everywhere (both baseline and budget-matched arms)
    max_cost: int = 3
    beam: int = 45
    max_nodes: int = 250_000
    time_budget_s: float = 4.0
    # individual experiments may widen these explicitly and must say so
    n_test: int = 20
    n_probe: int = 4
    # Stage-C safety caps
    h4_max_concepts: int = 100
    h2_max_depth: int = 16
    campaign_time_budget_s: float = 1800.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HardwareInfo:
    hardware: dict[str, Any]
    software: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"hardware": self.hardware, "software": self.software}


def probe_hardware() -> HardwareInfo:
    ram_gb = None
    try:
        import ctypes

        class _Mem(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        m = _Mem()
        m.dwLength = ctypes.sizeof(_Mem)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        ram_gb = round(m.ullTotalPhys / 1e9, 2)
    except Exception:
        try:
            import resource
            ram_gb = round(resource.getrlimit(resource.RLIMIT_AS) / 1e9, 2)
        except Exception:
            pass
    cuda = False
    device = "cpu"
    try:
        import torch  # noqa: F401  (presence check only; never used)
        cuda = bool(torch.cuda.is_available())
    except Exception:
        cuda = False
    cpu_name = platform.processor()
    try:
        import subprocess
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"],
            capture_output=True, text=True, timeout=20)
        name = (out.stdout or "").strip()
        if name:
            cpu_name = name
    except Exception:
        pass
    hardware = {
        "cpu": cpu_name,
        "logical_cores": os.cpu_count(),
        "ram_gb": ram_gb,
        "os": platform.platform(),
        "cuda": cuda,
        "c_compiler": "none on PATH",
        "device": device,
    }
    software = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "exp_h": "0.1.0",
    }
    try:
        import numpy
        software["numpy"] = numpy.__version__
    except Exception:
        software["numpy"] = None
    software["constructive_root"] = os.path.dirname(os.path.dirname(__file__))
    return HardwareInfo(hardware=hardware, software=software)


def fingerprint_of(obj: Any) -> str:
    """Stable 16-char fingerprint of configuration or dataset structures."""
    raw = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
