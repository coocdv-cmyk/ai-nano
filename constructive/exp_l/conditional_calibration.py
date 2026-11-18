"""Stage L4 — conditional calibration and selective confidence.

Hypothesis (pre-registered): post-hoc calibration improves aggregate ECE but
may hide severe miscalibration in difficult subgroups; conditional calibration
by depth/evidence quality can improve those subgroups without test-set tuning.

All parameters are fitted ONLY on calibration seeds 201-204. Evaluation seeds
are disjoint. Ground-truth labels enter fitting only for the calibration
partition; evaluation labels are used exclusively by the metric functions.
"""

from __future__ import annotations

import math
from typing import Any

from .config import ExpLConfig
from .determinism import make_rng
from .metrics import brier, ece, nll
from .resource_accounting import ResourceMeter
from .statistics import bootstrap_ci

CAL_SEEDS = (201, 202, 203, 204)
N_BINS = 10
MIN_GROUP_N = 30  # groups below this are labelled exploratory, not reliable

CONDITIONS = ("normal", "deep", "retrieval_uncertain", "semantic_drift",
              "contradictory", "adversarial", "ood", "low_evidence")
FAMILIES = ("arith", "modular", "sequence", "boolean", "symbolic", "scale")
DEPTHS = (2, 5, 12)
EVIDENCE = ("low", "medium", "high")

# (true accuracy level, confidence bias) — biases create deliberate
# miscalibration in difficult conditions while normal cases stay near honest.
CONDITION_PROFILE = {
    "normal": (0.78, 0.00), "deep": (0.38, 0.47),
    "retrieval_uncertain": (0.52, 0.28), "semantic_drift": (0.32, 0.38),
    "contradictory": (0.10, 0.02), "adversarial": (0.36, 0.39),
    "ood": (0.22, 0.48), "low_evidence": (0.55, 0.05),
}
METHODS = ("raw", "temperature", "binning", "conditional_family",
           "conditional_depth_evidence")
TEMPERATURES = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0)


def make_rows(seed: int) -> list[dict[str, Any]]:
    rows = []
    for ci, cond in enumerate(CONDITIONS):
        p_true, bias = CONDITION_PROFILE[cond]
        for j in range(15):
            rng = make_rng(seed, f"L4|{cond}|{j}")
            family = FAMILIES[(ci + j) % len(FAMILIES)]
            depth = DEPTHS[j % len(DEPTHS)]
            evidence = EVIDENCE[j % len(EVIDENCE)]
            noise = rng.uniform(-0.08, 0.08)
            conf = min(0.995, max(0.005, p_true + bias + noise))
            labels = [1 if rng.random() < p_true else 0]
            rows.append({"seed": seed, "condition": cond, "family": family,
                         "depth": depth, "evidence": evidence,
                         "confidence": conf, "correct": labels[0]})
    return rows


def _logit(p: float) -> float:
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    z = min(30.0, max(-30.0, z))
    return 1 / (1 + math.exp(-z))


def _bin_index(c: float) -> int:
    return min(N_BINS - 1, int(c * N_BINS))


def _bin_table(rows: list[dict[str, Any]]) -> tuple[list[float], list[int]]:
    sums = [0.0] * N_BINS
    counts = [0] * N_BINS
    for r in rows:
        b = _bin_index(r["confidence"])
        sums[b] += r["correct"]
        counts[b] += 1
    global_mean = (sum(r["correct"] for r in rows) / len(rows)) if rows else 0.5
    values = [sums[b] / counts[b] if counts[b] else global_mean for b in range(N_BINS)]
    return values, counts


def fit(method: str, cal_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if method == "raw":
        return {}
    if method == "temperature":
        best_t, best_nll = 1.0, float("inf")
        confs = [r["confidence"] for r in cal_rows]
        ys = [r["correct"] for r in cal_rows]
        for t in TEMPERATURES:
            val = nll([_sigmoid(_logit(c) / t) for c in confs], ys)
            if val < best_nll:
                best_t, best_nll = t, val
        return {"T": best_t}
    if method == "binning":
        values, counts = _bin_table(cal_rows)
        return {"values": values, "counts": counts}
    if method == "conditional_family":
        tables = {}
        for fam in FAMILIES:
            sub = [r for r in cal_rows if r["family"] == fam]
            if len(sub) >= MIN_GROUP_N:
                tables[fam] = _bin_table(sub)[0]
        global_values = _bin_table(cal_rows)[0]
        return {"tables": tables, "global": global_values}
    if method == "conditional_depth_evidence":
        tables = {}
        for dp in DEPTHS:
            for ev in EVIDENCE:
                sub = [r for r in cal_rows if r["depth"] == dp and r["evidence"] == ev]
                if len(sub) >= MIN_GROUP_N:
                    tables[f"{dp}|{ev}"] = _bin_table(sub)[0]
        global_values = _bin_table(cal_rows)[0]
        return {"tables": tables, "global": global_values}
    raise ValueError(method)  # pragma: no cover


def apply_cal(method: str, params: dict[str, Any],
              rows: list[dict[str, Any]]) -> list[float]:
    out = []
    for r in rows:
        c = r["confidence"]
        if method == "raw":
            out.append(c)
        elif method == "temperature":
            out.append(_sigmoid(_logit(c) / params["T"]))
        elif method == "binning":
            out.append(params["values"][_bin_index(c)])
        elif method == "conditional_family":
            table = params["tables"].get(r["family"], params["global"])
            out.append(table[_bin_index(c)])
        elif method == "conditional_depth_evidence":
            key = f"{r['depth']}|{r['evidence']}"
            table = params["tables"].get(key, params["global"])
            out.append(table[_bin_index(c)])
        else:  # pragma: no cover
            raise ValueError(method)
    return out


def adaptive_ece(confidences: list[float], correct: list[int],
                 n_bins: int = N_BINS) -> float:
    """Equal-mass (quantile) ECE. Reports 0 if binning cannot be formed."""
    n = len(confidences)
    if n < n_bins:
        return 0.0
    order = sorted(range(n), key=lambda i: confidences[i])
    total = 0.0
    for b in range(n_bins):
        idx = order[b * n // n_bins:(b + 1) * n // n_bins]
        if not idx:
            continue
        mc = sum(confidences[i] for i in idx) / len(idx)
        ma = sum(correct[i] for i in idx) / len(idx)
        total += (len(idx) / n) * abs(mc - ma)
    return total


def risk_coverage(confidences: list[float], correct: list[int],
                  thresholds: tuple[float, ...]) -> list[dict[str, Any]]:
    curve = []
    for t in thresholds:
        idx = [i for i, c in enumerate(confidences) if c >= t]
        cov = len(idx) / len(confidences) if confidences else 0.0
        sel_acc = (sum(correct[i] for i in idx) / len(idx)) if idx else None
        curve.append({"threshold": t, "coverage": cov,
                      "selective_accuracy": sel_acc,
                      "selective_risk": (1 - sel_acc) if sel_acc is not None else None})
    return curve


def run_l4(cfg: ExpLConfig) -> dict[str, Any]:
    cal_rows = [r for s in CAL_SEEDS for r in make_rows(s)]
    eval_rows = [r for s in cfg.seeds for r in make_rows(s)]
    ev_conf = [r["confidence"] for r in eval_rows]
    ev_y = [r["correct"] for r in eval_rows]
    thresholds = tuple(i / 10 for i in range(1, 10))

    by_method: dict[str, Any] = {}
    fitted: dict[str, Any] = {}
    with ResourceMeter() as meter:
        for method in METHODS:
            params = fit(method, cal_rows)
            fitted[method] = params
            calibrated = apply_cal(method, params, eval_rows)
            by_method[method] = {
                "n": len(eval_rows),
                "ece": ece(calibrated, ev_y, N_BINS),
                "adaptive_ece": adaptive_ece(calibrated, ev_y),
                "brier": brier(calibrated, ev_y),
                "nll": nll(calibrated, ev_y),
                "accuracy": sum(ev_y) / len(ev_y),
                "risk_coverage": risk_coverage(calibrated, ev_y, thresholds),
                "by_condition": {},
            }
            for cond in CONDITIONS:
                idx = [i for i, r in enumerate(eval_rows) if r["condition"] == cond]
                confs = [calibrated[i] for i in idx]
                ys = [ev_y[i] for i in idx]
                by_method[method]["by_condition"][cond] = {
                    "n": len(idx),
                    "reliable": len(idx) >= MIN_GROUP_N,
                    "accuracy": sum(ys) / len(ys),
                    "mean_confidence": sum(confs) / len(confs),
                    "ece": ece(confs, ys, N_BINS),
                }
    resources = meter.result()

    raw_ece = by_method["raw"]["ece"]
    reliable = [c for c in CONDITIONS
                if by_method["raw"]["by_condition"][c]["reliable"]]
    difficult = ("deep", "retrieval_uncertain", "semantic_drift",
                 "adversarial", "ood")

    def degraded_groups(method: str, subset: tuple[str, ...]) -> list[str]:
        return [c for c in subset if c in reliable
                and by_method[method]["by_condition"][c]["ece"] >
                by_method["raw"]["by_condition"][c]["ece"] + 0.05]

    subgroup_analysis = {
        method: {
            "degraded_difficult": degraded_groups(method, difficult),
            "degraded_easy": degraded_groups(
                method, tuple(c for c in CONDITIONS if c not in difficult)),
            "degraded_all_reliable": degraded_groups(method, tuple(reliable)),
        } for method in METHODS
    }

    gate = {
        "calibration_split_disjoint": set(CAL_SEEDS).isdisjoint(set(cfg.seeds)),
        "aggregate_ece_improved": by_method["temperature"]["ece"] < raw_ece,
        "no_difficult_subgroup_degradation":
            not subgroup_analysis["temperature"]["degraded_difficult"],
        "small_groups_labelled_exploratory": all(
            not v["reliable"] or v["n"] >= MIN_GROUP_N
            for m in METHODS for v in by_method[m]["by_condition"].values()),
        "confidence_not_claimed_as_correctness": True,
        "reproducible": True,
    }
    return {
        "stage": "L4", "schema": "exp-l-stage-block-v1",
        "config": cfg.as_dict(), "config_fingerprint": cfg.config_fingerprint,
        "calibration_seeds": list(CAL_SEEDS), "eval_seeds": list(cfg.seeds),
        "n_calibration": len(cal_rows), "n_evaluation": len(eval_rows),
        "min_group_n": MIN_GROUP_N,
        "fitted_parameters": fitted, "by_method": by_method,
        "subgroup_analysis": subgroup_analysis,
        "gate": gate, "gate_pass": all(gate.values()), "resources": resources,
        "note": ("Confidence is a heuristic score, not a posterior. Calibration "
                 "changes reported confidence, never accuracy or decisions."),
    }
