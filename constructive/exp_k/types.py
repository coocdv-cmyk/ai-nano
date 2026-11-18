"""EXP-K vocabulary: outcomes, stages, records and enums."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Stage(str, Enum):
    K0 = "K0"
    K1 = "K1"
    K2 = "K2"
    K3 = "K3"
    K4 = "K4"
    K5 = "K5"
    K6 = "K6"
    K7 = "K7"


class Completion(str, Enum):
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    TIMED_OUT = "TIMED_OUT"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


class DeepOutcome(str, Enum):
    CORRECT = "correct_acceptance"
    INCORRECT = "incorrect_acceptance"
    CORRECT_ABSTENTION = "correct_abstention"
    INCORRECT_ABSTENTION = "incorrect_abstention"
    NON_IDENTIFIABLE = "non_identifiable"
    REPRESENTATION = "representation_limitation"
    BUDGET = "budget_exhaustion"


class RetrievalOutcome(str, Enum):
    CORRECT = "correct_acceptance"
    WRONG = "wrong_acceptance"
    CORRECT_ABSTENTION = "correct_abstention"
    WRONG_ABSTENTION = "wrong_abstention"
    AMBIGUOUS = "ambiguous"


class DriftKind(str, Enum):
    STABLE = "stable"
    GRADUAL = "gradual"
    ABRUPT = "abrupt"
    RECURRING = "recurring"
    CYCLIC = "cyclic"
    CONCEPT = "concept"
    FEATURE = "feature"
    LABEL = "label"
    NOISE_LIKE = "noise_like"
    MISSING = "missing"
    DELAYED = "delayed"
    MISLEADING = "misleading"


class IntegrationDecision(str, Enum):
    NOT_RECOMMENDED = "NOT_RECOMMENDED"
    REQUIRES_MORE_EVIDENCE = "REQUIRES_MORE_EVIDENCE"
    PROMISING_BUT_UNVALIDATED = "PROMISING_BUT_UNVALIDATED"
    ELIGIBLE_FOR_SEPARATE_INTEGRATION_REVIEW = "ELIGIBLE_FOR_SEPARATE_INTEGRATION_REVIEW"


@dataclass
class Episode:
    seed: int
    family: str
    task: str
    condition: str = ""
    outcome: str = ""
    accepted: bool = False
    correct: bool = False
    wrong: bool = False
    abstained: bool = False
    coverage: float = 0.0
    confidence: float | None = None
    test_accuracy: float | None = None
    expansions: int = 0
    elapsed_s: float = 0.0
    timed_out: bool = False
    diagnostic: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed, "family": self.family, "task": self.task,
            "condition": self.condition, "outcome": self.outcome,
            "accepted": self.accepted, "correct": self.correct,
            "wrong": self.wrong, "abstained": self.abstained,
            "coverage": self.coverage, "confidence": self.confidence,
            "test_accuracy": self.test_accuracy, "expansions": self.expansions,
            "elapsed_s": self.elapsed_s, "timed_out": self.timed_out,
            "diagnostic": self.diagnostic, "meta": self.meta,
        }
