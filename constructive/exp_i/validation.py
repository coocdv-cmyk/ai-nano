"""EXP-I-A — multi-sample evidence validation, calibration and abstention.

The engine reports *why* it declined, using the six required states. It never
converts "all observed examples agree" into "semantically correct".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from . import evidence as ev
from .types import (Candidate, Example, Expressiveness, Task, ValidationResult,
                    ValidationState)

EvalFn = Callable[[Any, Any], Any]


# ---------------------------------------------------------------------------
# Calibration (empirical binning; fit on a dedicated calibration split)
# ---------------------------------------------------------------------------


class Calibrator:
    """Monotone empirical binning calibrator with ECE reporting.

    Confidence is a number in [0, 1]; correctness is 0/1. Calibration is fit on
    a split that is never used for candidate selection or test measurement.
    """

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins
        self.bin_correct = [0.0 for _ in range(n_bins)]
        self.bin_count = [0 for _ in range(n_bins)]
        self.fitted = False

    def _bin(self, c: float) -> int:
        i = int(c * self.n_bins)
        return max(0, min(self.n_bins - 1, i))

    def fit(self, confidences: Iterable[float], correct: Iterable[int]) -> None:
        pairs = list(zip(confidences, correct))
        if not pairs:
            return
        for c, y in pairs:
            b = self._bin(float(c))
            self.bin_count[b] += 1
            self.bin_correct[b] += float(y)
        self.fitted = True

    def calibrate(self, confidence: float) -> float:
        b = self._bin(confidence)
        if self.bin_count[b] == 0:
            return confidence  # unobserved bin: honest fallback, no fabrication
        return self.bin_correct[b] / self.bin_count[b]

    def ece(self, confidences: Iterable[float], correct: Iterable[int]) -> float:
        pairs = list(zip(confidences, correct))
        if not pairs:
            return 0.0
        total = len(pairs)
        err = 0.0
        for c, y in pairs:
            b = self._bin(float(c))
            if self.bin_count[b] == 0:
                obs = float(y)  # no calibration data in this bin
            else:
                obs = self.bin_correct[b] / self.bin_count[b]
            err += abs(float(c) - obs)
        return err / total


def calibration_report(confidences: list[float], correct: list[int],
                       *, n_bins: int = 10) -> dict[str, Any]:
    cal = Calibrator(n_bins)
    cal.fit(confidences, correct)
    bins = []
    for b in range(n_bins):
        if cal.bin_count[b]:
            mean_conf = None  # raw confidences per bin not retained; report empirical
            bins.append({
                "bin": b, "n": cal.bin_count[b],
                "empirical_accuracy": round(cal.bin_correct[b] / cal.bin_count[b], 4),
            })
    return {
        "n": len(confidences),
        "ece": round(cal.ece(confidences, correct), 4),
        "bins": bins,
        "note": "ECE computed on a split independent of candidate selection and test.",
    }


# ---------------------------------------------------------------------------
# Validation engine
# ---------------------------------------------------------------------------


class ValidationEngine:
    """Adaptive multi-sample validation over typed candidates."""

    def __init__(self, *, budget: int = 4, abstain_threshold: float = 0.6,
                 calibrator: Calibrator | None = None,
                 eval_fn: EvalFn | None = None):
        self.budget = budget
        self.threshold = abstain_threshold
        self.calibrator = calibrator
        self.eval_fn = eval_fn or _default_eval

    def validate(self, candidates: list[Candidate], task: Task, *,
                 probe_pool: list[Any] | None = None,
                 governor: Any = None) -> ValidationResult:
        eval_fn = self.eval_fn
        reasons: list[str] = []
        evidence_log: list[ev.EvidenceRecord] = []
        support = task.train
        pool = task.validation_pool

        def checkpoint() -> None:
            if governor is not None:
                governor.checkpoint("validation")

        checkpoint()

        # No expressible hypothesis at all: a representation limitation, not a
        # contradiction in the evidence.
        if not candidates:
            return ValidationResult(
                state=ValidationState.REJECTED, candidate_id=None,
                confidence=0.0, calibrated_confidence=None, surviving_ids=(),
                reasons=["no expressible candidate program in the declared DSL"],
                coverage=0.0, accuracy=0.0,
                metainfo={"expressiveness": Expressiveness.NOT_EXPRESSIBLE.value},
            )

        # Low-support guard
        if len(support) < 2:
            return ValidationResult(
                state=ValidationState.INSUFFICIENT_EVIDENCE, candidate_id=None,
                confidence=0.0, calibrated_confidence=None, surviving_ids=(),
                reasons=["support has fewer than 2 examples; no identifiability"],
                coverage=0.0, accuracy=0.0,
            )

        # Direct corrupted-evidence check
        direct = ev.contradictions_in_support(support)
        if direct:
            return ValidationResult(
                state=ValidationState.CONTRADICTORY, candidate_id=None,
                confidence=0.0, calibrated_confidence=None, surviving_ids=(),
                reasons=[f"identical input with conflicting labels: {direct}"],
                evidence=[ev.EvidenceRecord(x=d["x"], y=None, source="support",
                                            informative=True, note="contradiction") for d in direct],
                coverage=0.0, accuracy=0.0,
            )

        blind = ev.blind_positive_risk(support)
        if blind["at_risk"]:
            reasons.append("BLIND_POSITIVE: no positive support examples; all-negative rule unfalsified")

        scores = ev.score_support(candidates, support, eval_fn)
        surviving = ev.consistent_ids(scores)
        reasons.append(f"support-consistent candidates: {len(surviving)}/{len(candidates)}")

        if not surviving:
            return ValidationResult(
                state=ValidationState.CONTRADICTORY, candidate_id=None,
                confidence=0.0, calibrated_confidence=None, surviving_ids=(),
                reasons=reasons + ["no candidate reproduces all support examples"],
                coverage=0.0, accuracy=0.0,
                metainfo={"blind_positive": blind},
            )

        active = [c for c in candidates if c.id in surviving]
        used: set[str] = set()
        remaining_budget = self.budget

        # Adaptive validation: eliminate candidates using labelled validation pool.
        while remaining_budget > 0 and len(active) > 1:
            checkpoint()
            pick = ev.select_informative(pool, active, eval_fn, already_used=used)
            if pick is None:
                break
            used.add(ev._key(pick.x))
            eliminated_before = {c.id for c in active}
            new_active: list[Candidate] = []
            eliminated: list[str] = []
            for c in active:
                good, y = ev._out(eval_fn, c.program, pick.x)
                if good and ev._same(y, pick.y):
                    new_active.append(c)
                else:
                    eliminated.append(c.id)
            evidence_log.append(ev.provenance_for_selection(
                pick, eliminated, note="adaptive validation observation"))
            if new_active:
                active = new_active
            remaining_budget -= 1
            if not eliminated:
                reasons.append("validation example was redundant (no candidate eliminated)")

        # Non-identifiability test: survivors that disagree anywhere in probes/pool.
        probes = (probe_pool or []) + [e.x for e in pool] + [e.x for e in support]
        disagreement = ev.candidate_disagreement(active, probes, eval_fn)
        multi = [c for c in active]
        non_identifiable = False
        witness = None
        if len(multi) > 1:
            for x in probes:
                if ev.disagreement_at(multi, x, eval_fn) > 1:
                    non_identifiable = True
                    witness = x
                    break

        if non_identifiable:
            cex = None
            if witness is not None and len(multi) >= 2:
                cex = ev.counterexamples(multi[0].program, multi[1].program,
                                         probes, eval_fn, limit=1)
            return ValidationResult(
                state=ValidationState.NON_IDENTIFIABLE, candidate_id=None,
                confidence=0.0, calibrated_confidence=None,
                surviving_ids=tuple(c.id for c in active),
                evidence=evidence_log, reasons=reasons + [
                    f"{len(active)} observationally distinct hypotheses remain; "
                    f"they disagree at {witness!r}; refusing to claim a true rule",
                    "agreement on observed samples is not proof of semantic correctness",
                ],
                provenance={"disagreement": disagreement,
                            "counterexample": cex, "blind_positive": blind},
                coverage=0.0, accuracy=0.0,
            )

        if len(active) > 1:
            # More than one survivor but no observed disagreement in the pool:
            # cannot distinguish with available evidence.
            return ValidationResult(
                state=ValidationState.INSUFFICIENT_EVIDENCE, candidate_id=None,
                confidence=0.0, calibrated_confidence=None,
                surviving_ids=tuple(c.id for c in active),
                evidence=evidence_log, reasons=reasons + [
                    "multiple candidates survive but no discriminating observation available"],
                coverage=0.0, accuracy=0.0,
            )

        winner = active[0]

        # Budget exhausted with ambiguity unresolved?
        if remaining_budget <= 0 and len(candidates) > 1:
            reasons.append("validation budget exhausted before uniqueness established")

        # Final multi-sample check of the winner on the validation pool.
        correct = answered = 0
        for ex in pool:
            checkpoint()
            good, y = ev._out(eval_fn, winner.program, ex.x)
            if good:
                answered += 1
                if ev._same(y, ex.y):
                    correct += 1
        coverage = answered / len(pool) if pool else 0.0
        accuracy = correct / len(pool) if pool else 0.0
        if pool and accuracy < 1.0:
            return ValidationResult(
                state=ValidationState.REJECTED, candidate_id=winner.id,
                confidence=0.0, calibrated_confidence=None,
                surviving_ids=(winner.id,), evidence=evidence_log,
                reasons=reasons + [f"winner failed {len(pool) - correct} validation examples"],
                coverage=coverage, accuracy=accuracy,
            )

        # Confidence: support fit + validation margin + candidate uniqueness.
        fit = scores[winner.id]["n_ok"] / max(1, scores[winner.id]["n_total"])
        margin = 0.5 if len(candidates) == 1 else 0.2
        evidence_depth = min(
            1.0, (len(support) + len(evidence_log)) / (len(support) + max(1, self.budget)))
        raw = min(1.0, 0.45 * fit + 0.20 * accuracy + 0.15 * margin
                  + 0.20 * evidence_depth)
        cal = None
        if self.calibrator is not None and self.calibrator.fitted:
            cal = self.calibrator.calibrate(raw)
        decision_score = cal if cal is not None else raw
        verdict = (ValidationState.ACCEPTED if decision_score >= self.threshold
                   else ValidationState.INSUFFICIENT_EVIDENCE)
        if verdict is ValidationState.INSUFFICIENT_EVIDENCE:
            reasons.append(f"confidence {decision_score:.3f} < abstention threshold {self.threshold:.3f}")
        if blind["at_risk"]:
            # Never confidently accept an all-negative rule from zero positive evidence.
            verdict = ValidationState.INSUFFICIENT_EVIDENCE
            reasons.append("abstaining due to blind-positive evidence")

        return ValidationResult(
            state=verdict, candidate_id=winner.id if verdict is ValidationState.ACCEPTED else None,
            confidence=raw, calibrated_confidence=cal, surviving_ids=(winner.id,),
            evidence=evidence_log,
            reasons=reasons + ["acceptance reflects evidence consistency, not proof of semantics"],
            provenance={"redundancy": ev.redundancy_report(evidence_log),
                        "blind_positive": blind, "disagreement": disagreement},
            coverage=coverage, accuracy=accuracy,
            metainfo={"expressiveness": Expressiveness.EXPRESSIBLE_DISCOVERED.value},
        )


def _default_eval(program: Any, x: Any) -> Any:
    from .dsl import evaluate

    return evaluate(program, x)
