"""EXP-J explicit distribution-shift descriptors and split matrices."""

from __future__ import annotations

from typing import Any

from . import generators as G
from .types import ShiftSpec

# Names are unique to EXP-J and describe the structural / numerical change.
SHIFT_SPECS: tuple[ShiftSpec, ...] = tuple(
    ShiftSpec(
        name=f"{rule}:{shift}",
        train_domain=G.default_domains(rule, shift)[0],
        test_domain=G.default_domains(rule, shift)[1],
        numerical=shift in ("magnitude", "sign", "wide"),
        structural=shift in ("length", "both"),
        unseen_ops=G.UNSEEN_OPS.get(rule, ()),
    )
    for rule in G.ALL_RULES
    for shift in ("none", "magnitude", "sign", "wide", "length", "both")
)


def spec_for(rule: str, shift: str) -> ShiftSpec:
    for s in SHIFT_SPECS:
        if s.name == f"{rule}:{shift}":
            return s
    raise KeyError(f"{rule}:{shift}")


def transfer_matrix(rules: tuple[str, ...], shifts: tuple[str, ...]) -> list[dict[str, Any]]:
    """Describe a train/test distribution matrix for reporting."""
    return [
        {
            "rule": rule, "shift": shift,
            "train_domain": G.default_domains(rule, shift)[0],
            "test_domain": G.default_domains(rule, shift)[1],
            "structural": shift in ("length", "both"),
            "numerical": shift in ("magnitude", "sign", "wide"),
            "unseen_ops": list(G.UNSEEN_OPS.get(rule, ())),
        }
        for rule in rules for shift in shifts
    ]
