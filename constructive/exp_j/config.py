"""EXP-J configuration, hardware disclosure and integrity helpers.

Every long-running stage is governed by explicit deadlines and node budgets.
The protected-file manifest is recorded at STAGE 0 (before any EXP-J code runs
against it) and verified at close-out. The manifest deliberately covers all of
``omega_nano``, the original constructive core, EXP-H, EXP-I and the pre-existing
tests — but never EXP-J itself.
"""

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


@dataclass(frozen=True)
class ExpJConfig:
    """Bounded execution policy. Every number here is disclosed in the results."""

    # Principal evaluations: 10 independent seeds; quick mode uses 2.
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    n_episodes: int = 20
    n_val: int = 5
    abstain_threshold: float = 0.6
    max_cost: float = 4.0
    search_beam: int = 24
    search_node_budget: int = 12_000
    fit_deadline_s: float = 2.5
    experiment_deadline_s: float = 180.0
    campaign_deadline_s: float = 1200.0
    depths: tuple[int, ...] = (2, 3, 4, 5, 8, 12, 16)
    noise_levels: tuple[float, ...] = (0.0, 0.01, 0.05, 0.10, 0.20, 0.30)
    quick: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def quick_config() -> ExpJConfig:
    """Fast pilot: fewer seeds/episodes but the same code paths."""
    return ExpJConfig(
        seeds=(0, 1), n_episodes=6, n_val=3, search_node_budget=4_000,
        fit_deadline_s=1.5, experiment_deadline_s=60.0,
        campaign_deadline_s=300.0, quick=True,
    )


# ---------------------------------------------------------------------------
# Protected files (STAGE 0). EXP-J never writes to any of these paths.
# ---------------------------------------------------------------------------

PROTECTED: tuple[str, ...] = (
    "constructive/__init__.py", "constructive/baselines.py",
    "constructive/bench.py", "constructive/core.py", "constructive/model.py",
    "constructive/worlds.py",
    "constructive/exp_h/__init__.py", "constructive/exp_h/analysis.py",
    "constructive/exp_h/ast_engine.py", "constructive/exp_h/config.py",
    "constructive/exp_h/engine.py", "constructive/exp_h/episode.py",
    "constructive/exp_h/generators.py", "constructive/exp_h/h1_discovery.py",
    "constructive/exp_h/h2_scaling.py", "constructive/exp_h/h3_ood.py",
    "constructive/exp_h/h4_continual.py", "constructive/exp_h/h4_pool.py",
    "constructive/exp_h/h5_ast.py", "constructive/exp_h/h6_abstraction.py",
    "constructive/exp_h/h7_adversarial.py", "constructive/exp_h/h8_efficiency.py",
    "constructive/exp_h/lexical_h8.py", "constructive/exp_h/metrics.py",
    "constructive/exp_h/profiling.py", "constructive/exp_h/runner.py",
    "constructive/exp_h/smoke.py",
    "constructive/exp_i/__init__.py", "constructive/exp_i/abstraction.py",
    "constructive/exp_i/analysis.py", "constructive/exp_i/config.py",
    "constructive/exp_i/dsl.py", "constructive/exp_i/evidence.py",
    "constructive/exp_i/experiments.py", "constructive/exp_i/generalization.py",
    "constructive/exp_i/generators.py", "constructive/exp_i/metrics.py",
    "constructive/exp_i/repair.py", "constructive/exp_i/resources.py",
    "constructive/exp_i/retrieval.py", "constructive/exp_i/runner.py",
    "constructive/exp_i/search.py", "constructive/exp_i/smoke.py",
    "constructive/exp_i/types.py", "constructive/exp_i/validation.py",
    "omega_nano/__init__.py", "omega_nano/__main__.py", "omega_nano/cli.py",
    "omega_nano/config.py", "omega_nano/core/__init__.py",
    "omega_nano/core/data.py", "omega_nano/core/engine.py",
    "omega_nano/core/model.py", "omega_nano/core/train.py",
    "omega_nano/engine.py", "omega_nano/hw.py", "omega_nano/kernels.py",
    "omega_nano/memory.py", "omega_nano/model.py", "omega_nano/reasoner.py",
    "omega_nano/report.py", "omega_nano/security.py", "omega_nano/store.py",
    "omega_nano/tools/__init__.py", "omega_nano/tools/base.py",
    "omega_nano/tools/coretools.py", "omega_nano/tools/fs.py",
    "omega_nano/tools/mathx.py", "omega_nano/tools/memtools.py",
    "omega_nano/tools/shell.py", "omega_nano/verify.py",
    "tests/conftest.py", "tests/test_constructive.py", "tests/test_core.py",
    "tests/test_engine.py", "tests/test_exp_h.py", "tests/test_exp_i.py",
    "tests/test_llm.py", "tests/test_memory.py", "tests/test_reasoner.py",
    "tests/test_recovery.py", "tests/test_security.py", "tests/test_state.py",
    "tests/test_tools.py",
)

_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_of(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def source_revision() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=20)
        return (out.stdout or "").strip() or "no-commits"
    except Exception:  # pragma: no cover - environment dependent
        return "unknown"


def probe_hardware() -> dict[str, Any]:
    ram_gb = None
    try:
        import ctypes

        class _Mem(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
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
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"],
            capture_output=True, text=True, timeout=20,
        )
        cpu = (out.stdout or "").strip() or cpu
    except Exception:
        pass
    return {
        "hardware": {
            "cpu": cpu, "logical_cores": os.cpu_count(), "ram_gb": ram_gb,
            "os": platform.platform(), "cuda": False, "device": "cpu",
            "acceleration": "none (CPU-only, no external APIs)",
        },
        "software": {
            "python": sys.version.split()[0], "exp_j": "0.1.0",
            "source_revision": source_revision(),
        },
    }


def reproducibility_metadata(cfg: ExpJConfig) -> dict[str, Any]:
    hw = probe_hardware()
    return {
        "config_fingerprint": fingerprint_of(cfg.as_dict()),
        "configuration": cfg.as_dict(),
        "hardware": hw["hardware"],
        "software": hw["software"],
        "source_revision": source_revision(),
        "exp_i_source_hashes": {
            f: sha256_file(_ROOT / f)
            for f in PROTECTED if f.startswith("constructive/exp_i/")
        },
        "note": "System under test is EXP-I (read-only); generators are EXP-J-original.",
    }
