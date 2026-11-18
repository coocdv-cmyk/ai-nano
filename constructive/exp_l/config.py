"""Pre-registered configuration for EXP-L and its fingerprint.

A configuration is fixed before a stage runs and is recorded verbatim in every
result artifact.  The fingerprint is the canonical-hash of that configuration,
so a change in any parameter is detectable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from . import GENERATOR_VERSION, SEED_POLICY
from .determinism import fingerprint

STAGES = ("L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7")


@dataclass(frozen=True)
class ExpLConfig:
    stage: str = "L0"
    seed: int = 0
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    quick: bool = False
    no_resume: bool = False
    merge_only: bool = False
    generator_version: str = GENERATOR_VERSION
    seed_policy: str = SEED_POLICY
    # Shared data sizing (stages override as needed).
    n_train: int = 4
    n_val: int = 4
    n_test: int = 8
    # Resource budgets.  A stage exceeding these records a TIMEOUT/FAILURE,
    # never a successful abstention.
    time_limit_s: float = 120.0
    memory_limit_mb: float = 512.0
    full_test_timeout_s: float = 300.0

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["seeds"] = list(self.seeds)
        return d

    @property
    def config_fingerprint(self) -> str:
        return fingerprint(self.as_dict())


def default_config(stage: str = "L0", **overrides: Any) -> ExpLConfig:
    return replace(ExpLConfig(stage=stage), **overrides)


def quick_config(stage: str = "L0", **overrides: Any) -> ExpLConfig:
    base = ExpLConfig(
        stage=stage,
        seeds=(0, 1),
        quick=True,
        n_train=3,
        n_val=3,
        n_test=5,
        time_limit_s=30.0,
        memory_limit_mb=256.0,
        full_test_timeout_s=30.0,
    )
    return replace(base, **overrides)


def from_args(args: Any) -> ExpLConfig:
    """Build a config from argparse args, honouring --seed and --quick."""
    kw: dict[str, Any] = {
        "stage": args.stage,
        "no_resume": bool(args.no_resume),
        "merge_only": bool(args.merge_only),
    }
    if args.quick:
        cfg = quick_config(**kw)
    else:
        cfg = default_config(**kw)
    if getattr(args, "seed", None) is not None:
        cfg = replace(cfg, seed=int(args.seed), seeds=(int(args.seed),))
    return cfg
