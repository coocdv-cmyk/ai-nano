"""OMEGA EXP-H — open-world intelligence research module (isolated).

This package tests and, where a limitation is found, extends the CCM mechanism
without touching the protected ``constructive`` core.  It implements the eight
EXP-H experiments (H1..H8) with shared profiling, metrics and validation
infrastructure.

Absolute rules observed here:

* Nothing in this package modifies ``constructive/core.py``, ``model.py``,
  ``worlds.py``, ``baselines.py``, ``bench.py`` or ``omega_nano/``.
* No benchmark is solved by name: task identities never leak into the learner.
* Discovered results are never assumed to transfer outside this module's task
  distributions; the final report states exactly that.
"""

from .config import ExpHConfig, HardwareInfo, probe_hardware
from .metrics import (
    CI_95,
    aggregate,
    failure_counts,
    mean_ci,
    quantiles,
)
from .profiling import ProfileResult, profile_region

__all__ = [
    "ExpHConfig",
    "HardwareInfo",
    "probe_hardware",
    "ProfileResult",
    "profile_region",
    "CI_95",
    "aggregate",
    "failure_counts",
    "mean_ci",
    "quantiles",
]

__version__ = "0.1.0"
