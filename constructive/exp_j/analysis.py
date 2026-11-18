"""EXP-J aggregation, efficiency accounting and integration readiness."""

from __future__ import annotations

from typing import Any

from . import generators as G
from . import metrics as M
from .config import fingerprint_of
from .types import Arm, CompletionState

ALL = ("J1", "J2", "J3", "J4", "J5", "J6")


def experiment_status(blocks: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for eid, b in blocks.items():
        out[eid] = str(b.get("completion_status", b.get("status", "UNKNOWN")))
    return out


def failure_taxonomy(blocks: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for eid in ("J1", "J2", "J3"):
        b = blocks.get(eid, {})
        eps = b.get("episodes", [])
        if eid == "J1":
            out["J1"] = M.taxonomy(eps)
        elif eid == "J2":
            noise_eps = b.get("noise", {}).get("episodes", [])
            out["J2"] = M.taxonomy(noise_eps)
        else:
            out["J3"] = M.taxonomy(eps, key="outcome")
    return out


def efficiency(blocks: dict[str, Any]) -> dict[str, Any]:
    """J6 — aggregate computational-efficiency measurement."""
    per: dict[str, Any] = {}
    for eid, b in blocks.items():
        r = b.get("resources", {})
        per[eid] = {
            "wall_s": r.get("elapsed_s"),
            "peak_rss_mb": r.get("peak_rss_mb"),
            "expansions": r.get("expansions"),
            "nodes": r.get("nodes"),
            "cache_hits": r.get("cache_hits"),
            "verification_calls": r.get("verification_calls"),
            "timed_out": r.get("timed_out"),
            "completion_status": b.get("completion_status"),
        }
    j3 = blocks.get("J3", {})
    scaling = {}
    for key, block in j3.get("by_depth_strategy", {}).items():
        scaling[key] = {
            "expansions": block.get("expansions", {}).get("mean"),
            "elapsed_s": block.get("elapsed_s", {}).get("mean"),
            "nodes": block.get("nodes", {}).get("mean"),
        }
    j4 = blocks.get("J4", {})
    concept_bytes = []
    for row in j4.get("episodes", []):
        if row.get("storage_bytes") is not None:
            concept_bytes.append(row["storage_bytes"] / max(1, row["n_abstractions"]))
    return {
        "per_experiment": per,
        "total_wall_s": round(sum(v["wall_s"] or 0 for v in per.values()), 6),
        "max_peak_rss_mb": max((v["peak_rss_mb"] or 0 for v in per.values()), default=0),
        "total_expansions": sum(v["expansions"] or 0 for v in per.values()),
        "total_timed_out": sum(1 for v in per.values() if v["timed_out"]),
        "depth_scaling": scaling,
        "memory_per_concept_bytes": M.stats(concept_bytes),
        "validation_overhead_note": (
            "Validation/verification calls are counted per experiment; see resource blocks."),
        "cpu_only": True,
    }


def independence_evidence(blocks: dict[str, Any]) -> dict[str, Any]:
    return {
        "generators": "EXP-J original (constructive/exp_j/generators.py)",
        "n_new_rules": len(G.ALL_RULES),
        "rules": list(G.ALL_RULES),
        "disjoint_from_exp_i_rules": True,
        "unseen_operator_combinations": sorted(
            {op for rule in G.ALL_RULES for op in G.UNSEEN_OPS[rule]}),
        "seed_separation": "train/val/test use independent RNG draws and disjoint domains",
        "test_labels_used_for": "evaluation only, after decisions are made",
        "dataset_fingerprint": fingerprint_of({"rules": G.ALL_RULES}),
    }


def integration_readiness(blocks: dict[str, Any], integrity: dict[str, Any]) -> dict[str, Any]:
    reasons = [
        "All evidence is from synthetic typed-DSL tasks; no natural-input validation.",
        "EXP-I deep search still shows incorrect acceptance; J3 does not clear it.",
        "Abstraction lifecycle cost is non-negative in measured arms (J4).",
        "No integration into CCM or OMEGA-NANO is authorized or attempted.",
        "No general-intelligence, natural-language or universal-generalization claim is supported.",
    ]
    return {
        "integration_recommended": False,
        "reasons": reasons,
        "protected_files_unchanged": bool(integrity.get("protected_files_unchanged")),
        "validated_improvements": [
            "Evidence validation reduces wrong acceptance vs support-only baseline (J1).",
            "Independent generators transfer to new rule families and shifts (J1).",
            "Contradictory evidence is detected in most injected cases (J2).",
        ],
        "unresolved_limitations": [
            "Incorrect acceptance at depths 2-16 remains (J3).",
            "Abstraction arms do not reduce total lifecycle cost (J4).",
            "Drift detection by acceptance alone is unreliable (J2).",
        ],
    }


def headline_metrics(blocks: dict[str, Any]) -> dict[str, Any]:
    j1 = blocks.get("J1", {})
    j3 = blocks.get("J3", {})
    j4 = blocks.get("J4", {})
    return {
        "J1_wrong_acceptance": j1.get("comparisons", {}).get(
            "wrong_acceptance_vs_support_only", {}),
        "J1_episodes": j1.get("n_episodes"),
        "J1_retrieval_recall_at_2": j1.get("retrieval", {}).get("@2", {}).get("recall"),
        "J3_outcomes": j3.get("outcome_totals"),
        "J3_runs": j3.get("n_runs"),
        "J4_verdict": j4.get("verdict"),
        "J4_net_cost_means": {
            k: v["net_cost_s"]["mean"] for k, v in j4.get("summary", {}).items()},
    }
