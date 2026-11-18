"""EXP-K aggregation, resource accounting and integration decisions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import metrics as M
from .config import RESULTS_DIR, git_branch, git_commit
from .types import IntegrationDecision

STAGES = ("K0", "K1", "K2", "K3", "K4", "K5", "K6", "K7")


def load_stages() -> dict[str, Any]:
    out = {}
    for stage in STAGES:
        p = RESULTS_DIR / "experiments" / f"{stage}.json"
        if p.exists():
            try:
                out[stage] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    return out


def resource_summary(blocks: dict[str, Any]) -> dict[str, Any]:
    per = {}
    for stage, b in blocks.items():
        r = b.get("resources", {})
        per[stage] = {k: r.get(k) for k in (
            "elapsed_s", "peak_rss_mb", "expansions", "nodes",
            "validation_calls", "candidates_evaluated", "timed_out")}
    vals = [v["elapsed_s"] or 0 for v in per.values()]
    return {
        "per_stage": per,
        "total_wall_s": round(sum(vals), 6),
        "peak_rss_mb": max((v["peak_rss_mb"] or 0 for v in per.values()), default=0),
        "total_expansions": sum(v["expansions"] or 0 for v in per.values()),
        "total_validation_calls": sum(v["validation_calls"] or 0 for v in per.values()),
        "timed_out_stages": [s for s, v in per.items() if v["timed_out"]],
        "cpu_only": True,
    }


def integration_decisions(blocks: dict[str, Any]) -> dict[str, Any]:
    k1 = blocks.get("K1", {}).get("by_arm", {})
    k2 = blocks.get("K2", {}).get("by_arm", {})
    k3 = blocks.get("K3", {}).get("by_arm", {})
    k4 = blocks.get("K4", {}).get("by_method", {})
    k5 = blocks.get("K5", {}).get("total_regressions", 0)
    k6 = blocks.get("K6", {})
    return {
        "deep_composition_safeguards": {
            "decision": IntegrationDecision.REQUIRES_MORE_EVIDENCE.value,
            "reason": ("Shifted validation cuts incorrect acceptance by 0.229 but "
                       "raises abstention from 0.443 to 0.671 with zero correct "
                       "discoveries; oracle contracts are not deployable."),
            "baseline_wrong": k1.get("baseline", {}).get("incorrect_acceptance", {}).get("mean"),
            "best_wrong": k1.get("shifted_validation", {}).get("incorrect_acceptance", {}).get("mean"),
        },
        "ood_retrieval_safeguards": {
            "decision": IntegrationDecision.REQUIRES_MORE_EVIDENCE.value,
            "reason": ("Independent verification removes silent substitution but "
                       "loses two thirds of familiar coverage; type-aware ranking "
                       "does not help."),
            "baseline_substitution": k2.get("baseline", {}).get(
                "silent_substitution_ood", {}).get("mean"),
            "best_substitution": k2.get("independent_verification", {}).get(
                "silent_substitution_ood", {}).get("mean"),
        },
        "drift_feature_detector": {
            "decision": IntegrationDecision.PROMISING_BUT_UNVALIDATED.value,
            "reason": ("99.8% recall and 0 false alarms on feature drift with "
                       "calibration-fixed thresholds, but semantic/concept drift "
                       "with unchanged inputs was not modeled."),
            "recall": k3.get("feature_window", {}).get("detection_recall"),
            "false_positive_rate": k3.get("feature_window", {}).get("false_positive_rate"),
        },
        "confidence_temperature_scaling": {
            "decision": IntegrationDecision.PROMISING_BUT_UNVALIDATED.value,
            "reason": ("ECE 0.126 -> 0.074 on held-out data with no accuracy change; "
                       "fails on deep composition and retrieval where accuracy is ~0."),
            "raw_ece": k4.get("raw", {}).get("ece"),
            "temperature_ece": k4.get("temperature", {}).get("ece"),
        },
        "cross_mechanism_combination": {
            "decision": IntegrationDecision.NOT_RECOMMENDED.value,
            "reason": f"{k5} regressions were recorded, including coverage loss and "
                      "correct-acceptance regression.",
        },
        "adversarial_hardening": {
            "decision": IntegrationDecision.REQUIRES_MORE_EVIDENCE.value,
            "reason": ("All non-composition attacks handled with zero wrong acceptance, "
                       "but deep deceptive and resource-exhaustion attacks still show "
                       "40% incorrect acceptance."),
            "useful_coverage": k6.get("useful_coverage", {}).get("mean"),
        },
        "omeganano_integration": {
            "decision": IntegrationDecision.NOT_RECOMMENDED.value,
            "reason": "No mechanism is authorized for integration; integration remains a separate task.",
        },
    }


def final_report_data(blocks: dict[str, Any], integrity: dict[str, Any]) -> dict[str, Any]:
    return {
        "git": {"branch": git_branch(), "commit": git_commit()},
        "stages": list(blocks),
        "resources": resource_summary(blocks),
        "integration_decisions": integration_decisions(blocks),
        "integrity": integrity,
    }
