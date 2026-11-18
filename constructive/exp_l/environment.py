"""Environment capture for reproducibility (no experiment logic here)."""

from __future__ import annotations

import datetime as dt
import os
import platform
import sys
from typing import Any

from .determinism import ROOT
from .integrity import git_info


def _dep(name: str) -> dict[str, Any]:
    try:
        mod = __import__(name)
    except Exception as exc:  # pragma: no cover - defensive
        return {"installed": False, "error": type(exc).__name__}
    return {"installed": True, "version": getattr(mod, "__version__", "unknown")}


def _memory() -> dict[str, Any]:
    out: dict[str, Any] = {"total_gb": None, "available_gb": None}
    try:
        import psutil  # optional
        vm = psutil.virtual_memory()
        out["total_gb"] = round(vm.total / 1e9, 3)
        out["available_gb"] = round(vm.available / 1e9, 3)
        return out
    except Exception:
        pass
    try:  # Windows fallback, no third-party dependency
        import ctypes

        class Mem(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        m = Mem()
        m.dwLength = ctypes.sizeof(Mem)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        out["total_gb"] = round(m.ullTotalPhys / 1e9, 3)
        out["available_gb"] = round(m.ullAvailPhys / 1e9, 3)
    except Exception:  # pragma: no cover
        pass
    return out


def capture_environment() -> dict[str, Any]:
    return {
        "captured_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "os": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
        },
        "cpu": {"logical_cores": os.cpu_count(), "device": "cpu", "acceleration": "none"},
        "memory": _memory(),
        "dependencies": {n: _dep(n) for n in ("numpy", "scipy", "pytest", "psutil")},
        "git": git_info(),
        "repository_root": str(ROOT),
        "external_services_required": False,
        "gpu_required": False,
    }
