"""J2 — Noise, corruption and concept drift.

Measures correct acceptance, wrong acceptance, abstention, false rejection,
calibration, evidence quality, detection delay and recovery. The accuracy /
coverage trade-off is reported explicitly: higher abstention is not treated as
greater intelligence.
"""

from __future__ import annotations

from typing import Any

from . import drift as DR
from . import generators as G
from . import metrics as M
from . import noise as NZ
from .config import ExpJConfig
from .engine import run_validation, to_episode
from .resources import Monitor
from .types import DriftKind, NoiseKind

NOISE_RULES = ("affine_mod", "threshold_parity", "sum_top_k",
               "count_distinct_gt", "prime_and_odd")
DRIFT_RULES = ("affine_mod", "threshold_parity", "sum_top_k", "count_distinct_gt")


def _asdict(e) -> dict[str, Any]:
    return {
        "seed": e.seed, "family": e.family, "task": e.task, "state": e.state,
        "accepted": e.accepted, "correct_acceptance": e.correct_acceptance,
        "wrong_acceptance": e.wrong_acceptance, "abstained": e.abstained,
        "false_rejection": e.false_rejection,
        "non_identifiable": e.non_identifiable, "contradictory": e.contradictory,
        "insufficient": e.insufficient, "confidence": e.confidence,
        "test_accuracy": e.test_accuracy, "coverage": e.coverage,
        "evidence_informative": e.evidence_informative,
        "evidence_redundant": e.evidence_redundant,
        "failure_category": e.failure_category,
        "noise_kind": e.meta.get("noise_kind"),
        "noise_level": e.meta.get("noise_level"),
    }


def _block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    acc = [r["correct_acceptance"] for r in rows]
    answered = sum(1 for r in rows if r["accepted"])
    correct = sum(1 for r in rows if r["correct_acceptance"])
    ev_train = [r["evidence_informative"] for r in rows]
    ev_red = [r["evidence_redundant"] for r in rows]
    return {
        "n": len(rows),
        "correct_acceptance": M.rate(acc),
        "wrong_acceptance": M.rate(r["wrong_acceptance"] for r in rows),
        "abstention": M.rate(r["abstained"] for r in rows),
        "false_rejection": M.rate(r["false_rejection"] for r in rows),
        "non_identifiable": M.rate(r["non_identifiable"] for r in rows),
        "contradictory_detection": M.rate(r["contradictory"] for r in rows),
        "insufficient_detection": M.rate(r["insufficient"] for r in rows),
        "accuracy_coverage": M.accuracy_coverage(correct, answered, len(rows)),
        "calibration": M.calibration_ece(
            [r["confidence"] for r in rows],
            [1 if r["correct_acceptance"] else 0 for r in rows]),
        "evidence_quality": {
            "informative_mean": M.stats(ev_train),
            "redundant_mean": M.stats(ev_red),
            "informative_fraction": M.stats(
                [i / max(1, i + r) for i, r in zip(ev_train, ev_red)]),
            "note": "Adaptive selection should raise informative fraction when identifiable.",
        },
    }


def run_noise(cfg: ExpJConfig, *, seeds: tuple[int, ...] | None = None) -> dict[str, Any]:
    seeds = tuple(seeds if seeds is not None else cfg.seeds)
    eps: list[dict[str, Any]] = []
    for kind in NoiseKind:
        for level in NZ.noise_levels():
            for seed in seeds:
                for ri, rule in enumerate(NOISE_RULES):
                    clean = G.make_task(rule, seed * 37 + ri, n_val=cfg.n_val)
                    noisy = NZ.apply_noise(clean, kind, level, seed * 13 + ri)
                    _r, fields = run_validation(noisy, cfg)
                    ep = to_episode(noisy, seed, fields)
                    eps.append(_asdict(ep))

    by_kind: dict[str, Any] = {}
    for kind in NoiseKind:
        by_level = {}
        for level in NZ.noise_levels():
            sel = [r for r in eps if r["noise_kind"] == kind.value
                   and abs(float(r["noise_level"]) - level) < 1e-9]
            by_level[str(level)] = _block(sel)
        by_kind[kind.value] = {"by_level": by_level,
                               "overall": _block([r for r in eps
                                                  if r["noise_kind"] == kind.value])}

    # Distinguish insufficient evidence from contradiction (not collapsed).
    discrimination = {
        "contradiction_kind_detected_as_contradictory": by_kind["contradiction"]["overall"][
            "contradictory_detection"],
        "missing_kind_detected_as_insufficient": by_kind["missing"]["overall"][
            "insufficient_detection"],
        "note": "If these collapse into one state the engine fails to distinguish them.",
    }
    return {
        "experiment_id": "J2a", "experiment_name": "Noise and Corruption",
        "status": "completed", "seeds": list(seeds),
        "noise_levels": list(NZ.noise_levels()),
        "noise_kinds": [k.value for k in NoiseKind],
        "n_episodes": len(eps), "episodes": eps,
        "by_kind": by_kind, "state_discrimination": discrimination,
        "limitations": ["Synthetic label noise only; no natural annotation noise."],
    }


def _drift_flags(tasks: list[Any], cfg: ExpJConfig, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for t in tasks:
        _r, fields = run_validation(t, cfg)
        rows.append({
            "seed": seed, "rule": t.family, "phase": t.meta["phase"],
            "step": t.meta["drift_step"], "accepted": fields["accepted"],
            "correct_acceptance": fields["correct_acceptance"],
            "wrong_acceptance": fields["wrong_acceptance"],
            "abstained": fields["abstained"], "state": fields["state"],
            "test_accuracy": fields["test_accuracy"],
        })
    return rows


def run_drift(cfg: ExpJConfig, *, seeds: tuple[int, ...] | None = None) -> dict[str, Any]:
    seeds = tuple(seeds if seeds is not None else cfg.seeds)
    by_kind: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for kind in (DriftKind.GRADUAL, DriftKind.ABRUPT):
        delays: list[int] = []
        recoveries: list[int] = []
        k_rows: list[dict[str, Any]] = []
        for seed in seeds:
            for ri, rule in enumerate(DRIFT_RULES):
                tasks, d_idx, r_idx = DR.make_drift_stream(
                    rule, seed * 41 + ri, kind)
                rows = _drift_flags(tasks, cfg, seed)
                det = M.detection_delay(
                    [not x["accepted"] or x["wrong_acceptance"] for x in rows],
                    d_idx)
                rec = M.detection_delay(
                    [x["correct_acceptance"] for x in rows], r_idx)
                if det is not None:
                    delays.append(det)
                if rec is not None:
                    recoveries.append(rec)
                for row in rows:
                    row.update({"drift_kind": kind.value, "drift_index": d_idx,
                                "recovery_index": r_idx})
                k_rows.extend(rows)
        all_rows.extend(k_rows)
        by_kind[kind.value] = {
            "n_episodes": len(k_rows),
            "detection_delay": M.stats(delays),
            "recovery_delay": M.stats(recoveries),
            "n_undetected": len(seeds) * len(DRIFT_RULES) - len(delays),
            "acceptance_by_phase": {
                p: M.rate(r["accepted"] for r in k_rows if r["phase"] == p)
                for p in ("base", "drift", "recover")
            },
            "correct_by_phase": {
                p: M.rate(r["correct_acceptance"] for r in k_rows if r["phase"] == p)
                for p in ("base", "drift", "recover")
            },
        }
    return {
        "experiment_id": "J2b", "experiment_name": "Distribution Drift",
        "status": "completed", "seeds": list(seeds),
        "n_episodes": len(all_rows), "episodes": all_rows,
        "by_kind": by_kind,
        "limitations": ["Drift is simulated by moving generator domains, not natural data."],
    }


def run_noise_drift(cfg: ExpJConfig) -> dict[str, Any]:
    mon = Monitor()
    out = {
        "experiment_id": "J2",
        "experiment_name": "Noise, Corruption and Concept Drift",
        "status": "completed",
        "noise": run_noise(cfg),
        "drift": run_drift(cfg),
    }
    out.update(mon.finish())
    return out
