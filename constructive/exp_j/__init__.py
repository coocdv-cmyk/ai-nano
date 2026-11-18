"""OMEGA EXP-J — independent transfer validation and robustness campaign.

EXP-J is an isolated research package. It *uses* EXP-I mechanisms as the system
under test (read-only) but supplies entirely independent generators, splits,
noise models, cost accounting and resource governance. Nothing here is
integrated into CCM or ``omega_nano``.
"""

from .config import ExpJConfig

__all__ = ["ExpJConfig"]
