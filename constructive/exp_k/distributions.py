"""EXP-K distribution descriptors and shift matrix."""

from __future__ import annotations

from typing import Any

from . import generators as G

SHIFTS = ("none", "magnitude", "sign", "length", "both")


def shift_matrix(rules: tuple[str, ...] | None = None,
                 shifts: tuple[str, ...] = SHIFTS) -> list[dict[str, Any]]:
    rules = rules or G.ALL_RULES
    out = []
    for rule in rules:
        tr, te = G.domains(rule, "none")
        for shift in shifts:
            _, te_s = G.domains(rule, shift)
            out.append({
                "rule": rule, "shift": shift,
                "input_type": G.INPUT_TYPE[rule],
                "train_domain": tr, "test_domain": te_s,
                "structurally_shifted": shift in ("length", "both"),
                "numerically_shifted": shift in ("magnitude", "sign", "both"),
            })
    return out
