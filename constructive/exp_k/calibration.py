"""K4 — Confidence-calibration root-cause investigation.

Calibration is fit only on a separate calibration split (different seeds and
conditions) and evaluated on held-out seeds. Predictive confidence is never
presented as proof of correctness.
"""

from __future__ import annotations

import math
from typing import Any

from constructive.exp_i import dsl

from . import constructors as C
from . import generators as G
from . import metrics as M
from .config import ExpKConfig
from .engine import run_validation
from .resources import Monitor

CAL_SEEDS = (201, 202, 203, 204)
METHODS = ("raw", "temperature", "isotonic", "binning", "conformal_abstention")
CONDITIONS = ("normal", "deep", "ood", "noise", "contradictory", "retrieval")


def _logit(p: float) -> float:
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))


def _collect(seeds: tuple[int, ...], cfg: ExpKConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        # Normal in-domain and shifted tasks.
        for rule in G.ALL_RULES[:8]:
            for shift in ("none", "magnitude"):
                t = G.make_task(rule, seed * 7 + len(shift), shift=shift,
                                n_val=cfg.n_val, n_test=cfg.n_test)
                _r, f = run_validation(t, cfg)
                rows.append(_row(f, "normal", rule, seed))
        # Deep compositions.
        for depth in (2, 5, 12):
            t, hidden = C.deep_chain_task(seed, depth)
            s = __import__("constructive.exp_i.search", fromlist=["Searcher"]).Searcher(
                beam=cfg.search_beam, max_cost=min(cfg.max_cost, depth + 1.0),
                max_depth=depth)
            res = s.search(t, node_budget=cfg.search_node_budget,
                           deadline_s=cfg.fit_deadline_s)
            t.meta["candidates"] = res.candidates
            _r, f = run_validation(t, cfg)
            rows.append(_row(f, "deep", f"depth-{depth}", seed))
        # OOD drifted validation.
        for rule in G.ALL_RULES[:4]:
            t = G.make_task(rule, seed * 11 + 1, shift="sign",
                            n_val=cfg.n_val, n_test=cfg.n_test)
            _r, f = run_validation(t, cfg)
            rows.append(_row(f, "ood", rule, seed))
        # Noisy evidence.
        for rule in G.ALL_RULES[:4]:
            t = G.make_task(rule, seed * 13 + 2, n_val=cfg.n_val)
            from constructive.exp_i.types import Example, Task
            noisy = Task(name=t.name + "-noise", family=t.family,
                         train=[Example(e.x, e.y + 7 if isinstance(e.y, int)
                                        and not isinstance(e.y, bool) else e.y)
                                for e in t.train],
                         validation_pool=list(t.validation_pool),
                         test=list(t.test), input_type=t.input_type,
                         output_type=t.output_type, meta=dict(t.meta))
            _r, f = run_validation(noisy, cfg)
            rows.append(_row(f, "noise", rule, seed))
        # Contradictory support.
        for rule in G.ALL_RULES[:4]:
            t = G.make_task(rule, seed * 17 + 3, n_val=cfg.n_val)
            ex0 = t.train[0]
            from constructive.exp_i.types import Example, Task
            contra = Task(name=t.name + "-contra", family=t.family,
                          train=list(t.train) + [Example(ex0.x, ex0.y + 999
                                                         if isinstance(ex0.y, int)
                                                         else not ex0.y)],
                          validation_pool=list(t.validation_pool),
                          test=list(t.test), input_type=t.input_type,
                          output_type=t.output_type, meta=dict(t.meta))
            _r, f = run_validation(contra, cfg)
            rows.append(_row(f, "contradictory", rule, seed))
        # Retrieval substitution confidence.
        store = C.concept_pool(G.ALL_RULES[:6])
        for rule in G.ALL_RULES[12:]:
            t = G.make_task(rule, seed * 19 + 4, n_val=cfg.n_val)
            obs = [(e.x, e.y) for e in t.train + t.validation_pool]
            res = store.retrieve(input_type=t.input_type,
                                 output_type=t.output_type,
                                 observed_outputs=obs, top_k=1)
            rows.append({"condition": "retrieval", "family": rule, "seed": seed,
                         "confidence": round(float(res.confidence), 6),
                         "correct": 0, "state": "RETRIEVAL_SUBSTITUTION"})
    return rows


def _row(fields, condition, family, seed):
    return {"condition": condition, "family": family, "seed": seed,
            "confidence": fields["confidence"],
            "correct": int(fields["correct"]), "state": fields["state"]}


def _fit_temperature(rows):
    best_t, best_loss = 1.0, float("inf")
    for t in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]:
        loss = M.nll([_sigmoid(_logit(r["confidence"]) / t) for r in rows],
                     [r["correct"] for r in rows]) or 0.0
        if loss < best_loss:
            best_t, best_loss = t, loss
    return best_t


def _fit_bins(rows, n_bins=10):
    counts = [0] * n_bins
    correct = [0.0] * n_bins
    for r in rows:
        b = min(n_bins - 1, int(r["confidence"] * n_bins))
        counts[b] += 1
        correct[b] += r["correct"]
    return [(correct[b] / counts[b]) if counts[b] else None for b in range(n_bins)]


def _isotonic(rows):
    pts = sorted(rows, key=lambda r: r["confidence"])
    groups: list[tuple[list[float], list[int]]] = []
    for r in pts:
        groups.append(([r["confidence"]], [r["correct"]]))
    changed = True
    while changed:
        changed = False
        i = 0
        while i + 1 < len(groups):
            a, b = groups[i], groups[i + 1]
            ma = sum(a[1]) / len(a[1])
            mb = sum(b[1]) / len(b[1])
            if ma > mb:
                groups[i:i + 2] = [(a[0] + b[0], a[1] + b[1])]
                changed = True
            else:
                i += 1
    out = {}
    for cs, ys in groups:
        v = sum(ys) / len(ys)
        for c in cs:
            out[round(c, 9)] = v
    return out


def _transform(method, rows, params):
    if method == "raw":
        return [r["confidence"] for r in rows]
    if method == "temperature":
        return [_sigmoid(_logit(r["confidence"]) / params["temperature"])
                for r in rows]
    if method == "binning":
        return [params["bins"][min(9, int(r["confidence"] * 10))]
                or r["confidence"] for r in rows]
    if method == "isotonic":
        return [params["isotonic"].get(round(r["confidence"], 9), r["confidence"])
                for r in rows]
    return [r["confidence"] for r in rows]


def run_k4(cfg: ExpKConfig, *, seeds=None, methods=METHODS) -> dict[str, Any]:
    seeds = tuple(seeds if seeds is not None else cfg.seeds)
    mon = Monitor()
    cal = _collect(CAL_SEEDS, cfg)
    ev = _collect(seeds, cfg)
    cal_y = [r["correct"] for r in cal]
    params = {
        "temperature": _fit_temperature(cal),
        "bins": _fit_bins(cal),
        "isotonic": _isotonic(cal),
        "conformal_accept_fraction": 0.8,
    }
    # Conformal-style threshold: accept the top fraction of calibration confidences.
    sorted_cal = sorted(r["confidence"] for r in cal)
    params["conformal_threshold"] = sorted_cal[
        min(len(sorted_cal) - 1, int(0.8 * len(sorted_cal)))] if sorted_cal else 0.0

    by_method: dict[str, Any] = {}
    for method in methods:
        conf = _transform(method, ev, params)
        y = [r["correct"] for r in ev]
        if method == "conformal_abstention":
            chosen = [i for i, r in enumerate(ev)
                      if r["confidence"] >= params["conformal_threshold"]]
            coverage = len(chosen) / len(ev) if ev else None
            selective = (sum(ev[i]["correct"] for i in chosen) / len(chosen)
                         if chosen else None)
            by_method[method] = {
                "n": len(ev), "acc": M.rate(y), "coverage": M.stats(
                    [1.0 if r["confidence"] >= params["conformal_threshold"]
                     else 0.0 for r in ev]),
                "selective_accuracy": selective,
                "ece_on_accepted": M.ece([ev[i]["confidence"] for i in chosen],
                                         [ev[i]["correct"] for i in chosen]),
                "note": "Risk-controlled abstention; not a probability transform.",
            }
            continue
        reliability = []
        for b in range(10):
            idx = [i for i, c in enumerate(conf)
                   if min(9, int(c * 10)) == b]
            if idx:
                reliability.append({
                    "bin": b, "n": len(idx),
                    "mean_confidence": round(sum(conf[i] for i in idx) / len(idx), 6),
                    "empirical_accuracy": round(sum(y[i] for i in idx) / len(idx), 6),
                })
        by_method[method] = {
            "n": len(ev), "ece": M.ece(conf, y), "brier": M.brier(conf, y),
            "nll": M.nll(conf, y), "accuracy": M.rate(y),
            "confidence": M.stats(conf), "reliability_diagram": reliability,
        }

    by_condition: dict[str, Any] = {}
    for condition in CONDITIONS:
        ce = [r for r in ev if r["condition"] == condition]
        if not ce:
            continue
        raw = _transform("raw", ce, params)
        best = _transform("temperature", ce, params)
        by_condition[condition] = {
            "n": len(ce), "accuracy": M.rate(r["correct"] for r in ce),
            "raw_confidence": M.stats(r["confidence"] for r in ce),
            "raw_ece": M.ece(raw, [r["correct"] for r in ce]),
            "temperature_ece": M.ece(best, [r["correct"] for r in ce]),
            "states": M.confusion(ce, key="state"),
        }

    out = {
        "stage": "K4", "experiment_id": "K4",
        "experiment_name": "Confidence Calibration",
        "status": "completed", "seeds": list(seeds),
        "calibration_seeds": list(CAL_SEEDS),
        "n_calibration": len(cal), "n_evaluation": len(ev),
        "fitted_parameters": {"temperature": params["temperature"],
                              "bins": params["bins"],
                              "conformal_threshold": params["conformal_threshold"]},
        "methods": list(methods), "episodes": ev,
        "by_method": by_method, "by_condition": by_condition,
        "baseline_reference": {"exp_j_independent_ece": 0.159648},
        "caveats": [
            "Predictive confidence is not proof of correctness.",
            "Confidence values are heuristic blends, not posterior probabilities; "
            "post-hoc transforms can reduce ECE without improving decision quality.",
            "Retrieval substitutions are scored on retrieval confidence, not answer confidence.",
        ],
    }
    out.update(mon.finish())
    return out
