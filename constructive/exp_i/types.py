"""Immutable core data types for EXP-I.

No dependency on CCM or EXP-H. Every structure here is deliberately small,
serialisable, and carries the provenance/completion information needed to keep
"unknown" distinct from "wrong" and from "success".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ValidationState(str, Enum):
    """Required outcome vocabulary for the evidence engine."""

    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NON_IDENTIFIABLE = "NON_IDENTIFIABLE"
    CONTRADICTORY = "CONTRADICTORY"
    RESOURCE_LIMITED = "RESOURCE_LIMITED"


class CompletionStatus(str, Enum):
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    TIMED_OUT = "TIMED_OUT"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


class Expressiveness(str, Enum):
    """The four epistemic categories the DSL must keep distinct."""

    EXPRESSIBLE_DISCOVERED = "EXPRESSIBLE_DISCOVERED"
    EXPRESSIBLE_NOT_DISCOVERED = "EXPRESSIBLE_NOT_DISCOVERED"
    NOT_EXPRESSIBLE = "NOT_EXPRESSIBLE"
    NOT_IDENTIFIABLE = "NOT_IDENTIFIABLE"
    COMPUTE_LIMITED = "COMPUTE_LIMITED"


@dataclass(frozen=True)
class Example:
    x: Any
    y: Any

    def key(self) -> str:
        return _stable_key((self.x, self.y))


@dataclass
class Task:
    """A discovery task with disjoint support / validation / test splits.

    ``validation_pool`` is a separate labelled set available to the adaptive
    validator. It is never the test set and never used to train candidate
    generators: using it for candidate selection is legitimate, using test
    labels would not be.
    """

    name: str
    family: str
    train: list[Example]
    validation_pool: list[Example]
    test: list[Example]
    input_type: str
    output_type: str
    meta: dict[str, Any] = field(default_factory=dict)

    def all_inputs(self) -> list[Any]:
        return [e.x for e in (self.train + self.validation_pool + self.test)]


@dataclass(frozen=True)
class EvidenceRecord:
    """Provenance for a single validation observation."""

    x: Any
    y: Any
    source: str                 # "support" | "validation_pool" | "counterexample"
    informative: bool
    eliminated: tuple[str, ...] = ()   # candidate ids this example ruled out
    note: str = ""


@dataclass
class ValidationResult:
    state: ValidationState
    candidate_id: str | None
    confidence: float
    calibrated_confidence: float | None
    surviving_ids: tuple[str, ...]
    evidence: list[EvidenceRecord] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    coverage: float = 0.0
    accuracy: float = 0.0
    metainfo: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.state is ValidationState.ACCEPTED


def _stable_key(obj: Any) -> str:
    """Short deterministic key usable for ints, bools, lists, tuples, strings."""
    import hashlib
    import json

    raw = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def stable_key(obj: Any) -> str:
    return _stable_key(obj)


@dataclass
class Candidate:
    """A typed program hypothesis."""

    id: str
    program: Any
    cost: float = 1.0
    notes: dict[str, Any] = field(default_factory=dict)
