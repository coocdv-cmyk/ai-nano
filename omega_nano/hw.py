"""Hardware and environment detection, so the runtime can choose a path and benchmarks can state conditions."""

from __future__ import annotations

import os
import platform
import sqlite3
import subprocess
import sys


def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    if sys.platform.startswith("linux"):
        try:
            for line in open("/proc/cpuinfo", encoding="utf-8"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or "unknown"


def _ram_bytes() -> int | None:
    if sys.platform == "win32":
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        ms = MS()
        ms.dwLength = ctypes.sizeof(MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
            return int(ms.ullTotalPhys)
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def detect() -> dict:
    info = {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "cpu": _cpu_name(),
        "logical_cpus": os.cpu_count(),
        "ram_gb": round((_ram_bytes() or 0) / 2**30, 2) or None,
        "python": sys.version.split()[0],
        "sqlite": sqlite3.sqlite_version,
    }
    try:
        import numpy
        info["numpy"] = numpy.__version__
    except ImportError:
        info["numpy"] = None
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
    except ImportError:
        info["torch"] = None
    info["runtime_path"] = "cpu" if not info.get("cuda") else "cpu+cuda"
    return info
