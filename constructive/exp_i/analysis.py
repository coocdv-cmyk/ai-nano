"""Cross-experiment analysis and failure taxonomy for EXP-I."""

from __future__ import annotations

from typing import Any

from .metrics import taxonomy as _taxonomy


def experiment_status(blocks: dict[str, dict[str, Any]]) -> dict[str, str]:
    return {eid: str(b.get("status", "unknown")) for eid, b in blocks.items()}


def failure_taxonomy(blocks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for eid, block in blocks.items():
        for r in block.get("episodes", []):
            row = dict(r)
            row["experiment"] = eid
            rows.append(row)
    return {"by_category": _taxonomy(rows), "total_episode_rows": len(rows)}


def headline_metrics(blocks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "I-A" in blocks:
        ia = blocks["I-A"]
        out["I-A"] = {
            "validation_states": ia.get("validation_states"),
            "wrong_acceptance_discriminating":
                ia.get("accuracy_coverage_tradeoff", {})
                  .get("discriminating", {}).get("wrong_acceptance_rate", {}).get("mean"),
            "coverage_discriminating":
                ia.get("accuracy_coverage_tradeoff", {})
                  .get("discriminating", {}).get("coverage", {}).get("mean"),
            "ece": ia.get("calibration", {}).get("ece_independent"),
        }
    if "I-C" in blocks:
        ic = blocks["I-C"].get("summary", {})
        out["I-C"] = {
            "silent_substitution": ic.get("silent_substitution_rate", {}).get("mean"),
            "correct_top": ic.get("correct_top_rate", {}).get("mean"),
            "ambiguity": ic.get("ambiguity_rate", {}).get("mean"),
        }
    if "I-D" in blocks:
        out["I-D"] = {
            d: {"correct": v["correct_acceptance"]["mean"],
                "spurious": v["spurious_acceptance"]["mean"],
                "expansions": v["expansions"]["mean"],
                "timeouts": v["timeout_rate"]["mean"]}
            for d, v in blocks["I-D"].get("summary", {}).items()
        }
    if "I-F" in blocks:
        out["I-F"] = blocks["I-F"].get("summary", {})
    return out


def integration_readiness(blocks: dict[str, dict[str, Any]],
                          integrity: dict[str, Any]) -> dict[str, Any]:
    """Evidence-based, conservative readiness assessment. No automatic promotion."""
    all_complete = bool(blocks) and all(
        b.get("status") in ("completed", "merged") for b in blocks.values())
    ia = blocks.get("I-A", {}).get("accuracy_coverage_tradeoff", {})
    wrong = (ia.get("discriminating", {}).get("wrong_acceptance_rate", {}) or {}).get("mean")
    false_verified = (blocks.get("I-F", {}).get("summary", {})
                      .get("false_verified_rate", {}) or {}).get("mean")
    governance = blocks.get("I-H", {}).get("all_pass", False)
    # Even when every synthetic benchmark improves, integration is NOT authorized:
    # no non-synthetic validation exists and EXP-I is a declared research branch.
    ready = False
    return {
        "integration_recommended": ready,
        "mechanism_evidence_strong": bool(
            all_complete and wrong is not None and wrong < 0.05
            and false_verified == 0.0 and governance
            and integrity.get("protected_files_unchanged", False)),
        "all_experiments_complete": all_complete,
        "reason": ("EXP-I remains a research branch. Synthetic DSL benchmarks cannot "
                   "establish readiness for OMEGA-NANO, and no non-synthetic "
                   "validation has been performed. No automatic promotion is authorized; "
                   "this assessment records evidence only."),
        "protected_files_unchanged": integrity.get("protected_files_unchanged"),
        "evidence_summary": headline_metrics(blocks),
    }
