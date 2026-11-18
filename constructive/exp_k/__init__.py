"""OMEGA EXP-K — root-cause isolation and reliability engineering.

Isolated research package. EXP-I/EXP-J are the read-only systems under test;
EXP-K supplies independent generators, diagnostics, safeguards and evaluation.
Nothing here is integrated into CCM or ``omega_nano``.
"""

from .config import ExpKConfig, quick_config

__all__ = ["ExpKConfig", "quick_config"]
