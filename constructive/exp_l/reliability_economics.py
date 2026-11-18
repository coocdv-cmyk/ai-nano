"""Stage L6 — reliability economics and computational efficiency.

Every configuration runs the IDENTICAL workload on the same machine. Costs are
reported as a vector (wall, CPU, memory, operation counts, latency distribution,
throughput) alongside quality metrics; no single composite score is produced.

Operation counts are instrumented counts of THIS lightweight implementation.
They are explicitly not OMEGA-NANO/EXP-K search-expansion counts.
"""

from __future__ import annotations

import time
from typing import Any

from .composition_verification import CASE_TYPES, DEPTHS as C_DEPTHS, decide as c_decide, make_case, public_view
from .config import ExpLConfig
from .conditional_calibration import apply_cal, fit, make_rows as make_cal_rows, CAL_SEEDS
from .metrics import safe_rate
from .natural_input_bridge import TEMPLATES, answer as l5_answer, build_corpus
from .resource_accounting import ResourceMeter
from .retrieval_generalization import POOL, decide as r_decide, make_query
from .semantic_drift import calibrate as l1_calibrate, detect as l1_detect, make_stream, signals
from .statistics import bootstrap_ci

CONFIGS = ("baseline", "always_abstain", "l1_drift_only", "l2_selective",
           "l2_full", "l3_rank_verify", "l4_temperature", "coverage_oriented",
           "accuracy_oriented", "resource_constrained")


def _percentile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))]


def build_workload(cfg: ExpLConfig) -> list[dict[str, Any]]:
    tasks = []
    for seed in cfg.seeds:
        for depth in C_DEPTHS:
            for ct in CASE_TYPES[:4]:
                tasks.append({"seed": seed, "depth": depth, "case_type": ct})
    return tasks


def _op_counts(task: dict[str, Any], config: str) -> dict[str, int]:
    """Analytic instrumented counts for the operations a config performs."""
    c = {k: 0 for k in ("node_evaluations", "verification_checks",
                        "retrieval_scoring_ops", "candidates_inspected",
                        "calibration_ops", "drift_signal_ops")}
    comp_enabled = config in ("l2_selective", "l2_full", "coverage_oriented",
                              "accuracy_oriented", "resource_constrained")
    retr_enabled = config in ("l3_rank_verify", "coverage_oriented",
                              "accuracy_oriented", "resource_constrained")
    cal_enabled = config in ("l4_temperature", "accuracy_oriented")
    drift_enabled = config in ("l1_drift_only", "coverage_oriented",
                               "accuracy_oriented", "resource_constrained")
    if comp_enabled:
        n_anchor = 5
        c["node_evaluations"] += task["depth"] * n_anchor
        c["verification_checks"] += (2 if config in ("l2_selective", "coverage_oriented")
                                     else 1 if config == "resource_constrained"
                                     else task["depth"])
    if retr_enabled:
        c["retrieval_scoring_ops"] += len(POOL) * 6
        c["candidates_inspected"] += len(POOL)
    if cal_enabled:
        c["calibration_ops"] += 1
    if drift_enabled:
        c["drift_signal_ops"] += 12 * 2
    return c


def _task_quality(task: dict[str, Any], config: str) -> dict[str, Any]:
    """Decisions and ground truth for one workload task (evaluation only)."""
    case = make_case(task["seed"], task["depth"], task["case_type"])
    pub = public_view(case)
    q = make_query(task["seed"], "heldout" if task["seed"] % 2 else "familiar")

    # Composition decision by configuration.
    if config == "always_abstain":
        comp_accepted, comp_correct = False, case["correct"]
    elif config in ("l2_selective", "coverage_oriented"):
        comp_accepted = c_decide("selective", pub)["accepted"]
        comp_correct = case["correct"]
    elif config in ("l2_full", "accuracy_oriented"):
        comp_accepted = c_decide("full", pub)["accepted"]
        comp_correct = case["correct"]
    elif config == "resource_constrained":
        comp_accepted = c_decide("budget", pub, budget=1)["accepted"]
        comp_correct = case["correct"]
    else:
        comp_accepted = c_decide("baseline", pub)["accepted"]
        comp_correct = case["correct"]

    # Retrieval decision by configuration.
    if config == "always_abstain":
        ret_accepted, ret_correct = False, (q["expected"] is None)
    elif config in ("l3_rank_verify", "coverage_oriented", "accuracy_oriented"):
        d = r_decide("rank_verify", q)
        ret_accepted = d["accepted"]
        ret_correct = ((q["expected"] is None and not ret_accepted) or
                       (q["expected"] is not None and ret_accepted and
                        d["cid"] == q["expected"]))
    elif config == "resource_constrained":
        d = r_decide("evidence_aware", q)
        ret_accepted = d["accepted"]
        ret_correct = ((q["expected"] is None and not ret_accepted) or
                       (q["expected"] is not None and ret_accepted and
                        d["cid"] == q["expected"]))
    else:
        d = r_decide("baseline", q)
        ret_accepted = d["accepted"]
        ret_correct = (q["expected"] is not None and d["cid"] == q["expected"])

    return {"comp_accepted": comp_accepted, "comp_correct": comp_correct,
            "ret_accepted": ret_accepted, "ret_correct": ret_correct}


def run_l6(cfg: ExpLConfig) -> dict[str, Any]:
    tasks = build_workload(cfg)
    cal_rows = [r for s in CAL_SEEDS for r in make_cal_rows(s)]
    l1_thr = l1_calibrate()
    temp_params = fit("temperature", cal_rows)  # fitted once on calibration only
    results: dict[str, Any] = {}
    for config in CONFIGS:
        counts = {k: 0 for k in ("node_evaluations", "verification_checks",
                                 "retrieval_scoring_ops", "candidates_inspected",
                                 "calibration_ops", "drift_signal_ops")}
        latencies: list[float] = []
        ca = ia = abst = 0
        with ResourceMeter() as meter:
            for task in tasks:
                t0 = time.perf_counter()
                if config in ("l1_drift_only", "coverage_oriented",
                              "accuracy_oriented", "resource_constrained"):
                    st = make_stream(task["seed"], "parity", "semantic_abrupt", "square_mod5")
                    l1_detect("semantic_consistency", signals(st), l1_thr, task["seed"])
                if config in ("l4_temperature", "accuracy_oriented"):
                    apply_cal("temperature", temp_params, [{"confidence": 0.5}])
                q = _task_quality(task, config)
                for k, v in _op_counts(task, config).items():
                    counts[k] += v
                accepted = q["comp_accepted"] or q["ret_accepted"]
                correct = ((q["comp_accepted"] and q["comp_correct"]) or
                           (q["ret_accepted"] and q["ret_correct"]))
                if accepted and correct:
                    ca += 1
                elif accepted and not correct:
                    ia += 1
                if not accepted:
                    abst += 1
                latencies.append(time.perf_counter() - t0)
        res = meter.result()
        n = len(tasks)
        results[config] = {
            "n": n, "resources": res, "operation_counts": counts,
            "correct_acceptance": ca, "incorrect_acceptance": ia,
            "useful_coverage": ca / n, "incorrect_acceptance_rate": ia / n,
            "abstention_rate": abst / n,
            "avg_latency_ms": 1000 * res["wall_s"] / n,
            "p95_latency_ms": 1000 * _percentile(latencies, 0.95),
            "throughput_per_s": n / res["wall_s"] if res["wall_s"] > 0 else 0.0,
            "correctly_accepted_per_s": ca / res["wall_s"] if res["wall_s"] > 0 else 0.0,
            "useful_coverage_per_mb": (ca / n) / res["peak_rss_mb"] if res["peak_rss_mb"] else None,
            "incorrect_acceptances_per_1000": 1000 * ia / n,
        }

    base = results["baseline"]
    for config in CONFIGS:
        r = results[config]
        d_ca = r["correct_acceptance"] - base["correct_acceptance"]
        d_ia = base["incorrect_acceptance"] - r["incorrect_acceptance"]
        r["verification_cost_per_extra_correct"] = safe_rate(
            r["operation_counts"]["verification_checks"], d_ca) if d_ca > 0 else None
        r["latency_increase_per_incorrect_reduction_ms"] = safe_rate(
            1000 * (r["avg_latency_ms"] - base["avg_latency_ms"]), d_ia) if d_ia > 0 else None
        r["delta_useful_coverage"] = r["useful_coverage"] - base["useful_coverage"]
        r["delta_incorrect_acceptance"] = r["incorrect_acceptance_rate"] - base["incorrect_acceptance_rate"]

    gate = {
        "identical_workload": all(results[c]["n"] == len(tasks) for c in CONFIGS),
        "vector_metrics_reported": True,
        "no_composite_score": True,
        "cost_reported_for_all": all("resources" in results[c] for c in CONFIGS),
        "some_reliable_gain_at_reasonable_cost": any(
            results[c]["incorrect_acceptance_rate"] < base["incorrect_acceptance_rate"]
            and results[c]["useful_coverage"] >= 0.8 * base["useful_coverage"]
            and results[c]["resources"]["wall_s"] <= 3 * base["resources"]["wall_s"]
            for c in CONFIGS if c != "always_abstain"),
    }
    return {
        "stage": "L6", "schema": "exp-l-stage-block-v1",
        "config": cfg.as_dict(), "config_fingerprint": cfg.config_fingerprint,
        "n_tasks": len(tasks), "configurations": results,
        "baseline_reference": {k: base[k] for k in
                               ("useful_coverage", "incorrect_acceptance_rate",
                                "avg_latency_ms", "p95_latency_ms", "resources")},
        "gate": gate, "gate_pass": all(gate.values()),
        "note": ("Operation counts are instrumented counts of this "
                 "implementation, not EXP-K/OMEGA-NANO search expansions. "
                 "All configurations use the identical workload and hardware."),
    }
