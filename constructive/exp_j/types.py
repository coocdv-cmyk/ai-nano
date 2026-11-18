"""EXP-J data vocabulary.

Containers are intentionally small and serialisable. Complex epistemic states
(wrong / unknown / resource-limited) are always kept distinct.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NoiseKind(str, Enum):
    CLEAN = "clean"
    RANDOM_LABEL = "random_label"
    STRUCTURED_CORRUPTION = "structured_corruption"
    CONTRADICTION = "contradiction"
    MISSING = "missing"
    DUPLICATE = "duplicate"
    MISLEADING_SUPPORT = "misleading_support"


class DriftKind(str, Enum):
    NONE = "none"
    GRADUAL = "gradual"
    ABRUPT = "abrupt"


class Outcome(str, Enum):
    """Six-way classification required by J3."""

    CORRECT_DISCOVERED = "correct_discovered"
    INCORRECT_ACCEPTED = "incorrect_accepted"
    CORRECT_NOT_DISCOVERED = "correct_not_discovered"
    NON_IDENTIFIABLE = "non_identifiable"
    REPRESENTATION_LIMITATION = "representation_limitation"
    BUDGET_EXHAUSTION = "budget_exhaustion"


class Arm(str, Enum):
    """J4 abstraction-economy arms."""

    NONE = "A_no_abstraction"
    CURRENT = "B_current_cost_model"
    REUSE_CREDIT = "C_reuse_crediting"
    RETRIEVAL_RANKED = "D_retrieval_aware_ranking"
    SELECTIVE = "E_selective_retention"


class CompletionState(str, Enum):
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    TIMED_OUT = "TIMED_OUT"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


@dataclass
class ShiftSpec:
    """Explicit train / validation / test domains for a distribution shift."""

    name: str
    train_domain: dict[str, Any]
    test_domain: dict[str, Any]
    structural: bool = False
    numerical: bool = False
    unseen_ops: tuple[str, ...] = ()


@dataclass
class EpisodeRecord:
    """One measured episode; raw data is preserved, never collapsed."""

    seed: int
    family: str
    task: str
    state: str = ""
    accepted: bool = False
    correct_acceptance: bool = False
    wrong_acceptance: bool = False
    abstained: bool = False
    false_rejection: bool = False
    non_identifiable: bool = False
    contradictory: bool = False
    insufficient: bool = False
    confidence: float = 0.0
    test_accuracy: float | None = None
    coverage: float = 0.0
    calibrated_confidence: float | None = None
    evidence_informative: int = 0
    evidence_redundant: int = 0
    expansions: int = 0
    nodes: int = 0
    cache_hits: int = 0
    elapsed_s: float = 0.0
    timed_out: bool = False
    outcome: str = ""
    failure_category: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
