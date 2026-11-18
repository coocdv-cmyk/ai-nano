"""K3 — Drift-detection root-cause investigation.

Controlled streams cover stable, gradual, abrupt, recurring, cyclic, concept,
feature, label, noise-like, missing, delayed and adversarial drift. Detector
thresholds are fixed on a separate calibration split, then evaluated on held-out
streams. Arms include the acceptance-based baseline and explicit window,
feature and combined statistics.
"""

from __future__ import annotations

from typing import Any

from . import constructors as C
from . import generators as G
from . import metrics as M
from .config import ExpKConfig
from .engine import run_validation
from .resources import Monitor
from .types import DriftKind

ARMS = ("acceptance_baseline", "accuracy_window", "feature_window",
        "two_window", "combined")
DRIFT_RULES = G.ALL_RULES[:4]
CAL_SEEDS = (101, 102, 103)
EVAL_KINDS = (
    DriftKind.STABLE, DriftKind.GRADUAL, DriftKind.ABRUPT,
    DriftKind.RECURRING, DriftKind.CYCLIC, DriftKind.CONCEPT,
    DriftKind.FEATURE, DriftKind.LABEL, DriftKind.NOISE_LIKE,
    DriftKind.MISSING, DriftKind.DELAYED, DriftKind.MISLEADING,
)


def _feature(task) -> float:
    xs = [e.x for e in task.train + task.validation_pool]
    if not xs:
        return 0.0
    if task.input_type == "int":
        return float(sum(x for x in xs if isinstance(x, int)) /
                     max(1, sum(1 for x in xs if isinstance(x, int))))
    if task.input_type == "pair":
        return float(sum(a + b for a, b in xs) / len(xs))
    return float(sum(len(x) for x in xs) / len(xs))


def _observe(rule: str, seed: int, kind: DriftKind, cfg) -> tuple[list[dict[str, Any]], int, int]:
    tasks, d_idx, r_idx = C.drift_stream(rule, seed, kind, n_val=cfg.n_val,
                                         n_test=cfg.n_test)
    rows = []
    for t in tasks:
        _r, f = run_validation(t, cfg)
        rows.append({
            "phase": t.meta["phase"], "step": t.meta["drift_step"],
            "accepted": f["accepted"], "wrong": f["wrong"],
            "val_accuracy": f["val_accuracy"], "feature": _feature(t),
            "test_accuracy": f["test_accuracy"],
        })
    return rows, d_idx, r_idx


def _calibrate(cfg) -> dict[str, float]:
    """Fixed thresholds from calibration streams only (never evaluation data)."""
    deltas: list[float] = []
    accs: list[float] = []
    for seed in CAL_SEEDS:
        for rule in DRIFT_RULES:
            rows, _d, _r = _observe(rule, seed, DriftKind.STABLE, cfg)
            base_f = sum(x["feature"] for x in rows[:3]) / 3
            deltas.extend(abs(x["feature"] - base_f) for x in rows)
            accs.extend(x["val_accuracy"] for x in rows)
    deltas.sort()
    return {
        "feature_threshold": deltas[min(len(deltas) - 1, int(0.95 * len(deltas)))]
        if deltas else 1e9,
        "accuracy_threshold": (sum(accs) / len(accs)) - 0.05 if accs else 0.95,
        "window": 2,
    }


def _flags(arm: str, rows: list[dict[str, Any]], th: dict[str, float]) -> list[bool]:
    out = []
    w = int(th["window"])
    for i, row in enumerate(rows):
        if arm == "acceptance_baseline":
            out.append((not row["accepted"]) or row["wrong"])
        elif arm == "accuracy_window":
            window = rows[max(0, i - w + 1):i + 1]
            out.append(sum(x["val_accuracy"] for x in window) / len(window)
                       < th["accuracy_threshold"])
        elif arm == "feature_window":
            base = sum(x["feature"] for x in rows[:3]) / 3
            window = rows[max(0, i - w + 1):i + 1]
            out.append(sum(abs(x["feature"] - base) for x in window) / len(window)
                       > th["feature_threshold"])
        elif arm == "two_window":
            if i < 2 * w:
                out.append(False)
            else:
                recent = sum(x["feature"] for x in rows[i - w:i]) / w
                older = sum(x["feature"] for x in rows[i - 2 * w:i - w]) / w
                out.append(abs(recent - older) > th["feature_threshold"])
        else:  # combined
            base = sum(x["feature"] for x in rows[:3]) / 3
            win = rows[max(0, i - w + 1):i + 1]
            feat = sum(abs(x["feature"] - base) for x in win) / len(win)
            acc = sum(x["val_accuracy"] for x in win) / len(win)
            out.append(feat > th["feature_threshold"] or
                       acc < th["accuracy_threshold"])
    return out


def run_k3(cfg: ExpKConfig, *, seeds=None, arms=ARMS) -> dict[str, Any]:
    seeds = tuple(seeds if seeds is not None else cfg.seeds)
    mon = Monitor()
    th = _calibrate(cfg)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for kind in EVAL_KINDS:
            for rule in DRIFT_RULES:
                obs, d_idx, r_idx = _observe(rule, seed * 31 + sum(
                    ord(c) for c in kind.value), kind, cfg)
                for arm in arms:
                    flags = _flags(arm, obs, th)
                    detected = any(flags[d_idx:r_idx])
                    false_alarm = any(flags[:d_idx])
                    delay = next((i - d_idx for i in range(d_idx, r_idx)
                                  if flags[i]), None)
                    recovery = next((i - r_idx for i in range(r_idx, len(flags))
                                     if not flags[i] and obs[i]["val_accuracy"] == 1.0),
                                    None)
                    rows.append({
                        "arm": arm, "seed": seed, "kind": kind.value,
                        "rule": rule, "n_episodes": len(obs),
                        "detected": detected, "false_alarm": false_alarm,
                        "detection_delay": delay, "recovery_delay": recovery,
                        "acc_before": M.stats(x["val_accuracy"] for x in obs[:d_idx])["mean"],
                        "acc_during": M.stats(
                            x["val_accuracy"] for x in obs[d_idx:r_idx])["mean"],
                        "acc_after": M.stats(
                            x["val_accuracy"] for x in obs[r_idx:])["mean"],
                        "acceptance_during": M.rate(
                            x["accepted"] for x in obs[d_idx:r_idx])["mean"],
                    })

    by_arm: dict[str, Any] = {}
    for arm in arms:
        sel = [r for r in rows if r["arm"] == arm]
        shifted = [r for r in sel if r["kind"] != "stable"]
        stable = [r for r in sel if r["kind"] == "stable"]
        detected = sum(1 for r in shifted if r["detected"])
        false_alarms = sum(1 for r in stable if r["false_alarm"])
        delays = [r["detection_delay"] for r in shifted if r["detection_delay"] is not None]
        by_arm[arm] = {
            "n_streams": len(sel), "n_shifted": len(shifted),
            "detection_recall": round(detected / len(shifted), 6) if shifted else None,
            "miss_rate": round(1 - detected / len(shifted), 6) if shifted else None,
            "false_positive_rate": round(false_alarms / len(stable), 6) if stable else None,
            "precision": (round(detected / (detected + false_alarms), 6)
                          if detected + false_alarms else None),
            "detection_delay": M.stats(delays),
            "recovery_delay": M.stats(r["recovery_delay"] for r in shifted),
            "acc_before": M.stats(r["acc_before"] for r in shifted),
            "acc_during": M.stats(r["acc_during"] for r in shifted),
            "acc_after": M.stats(r["acc_after"] for r in shifted),
        }
    base = {(r["seed"], r["kind"], r["rule"]): r["detected"]
            for r in rows if r["arm"] == "acceptance_baseline"}
    comparisons = {}
    for arm in arms:
        if arm == "acceptance_baseline":
            continue
        cur = {(r["seed"], r["kind"], r["rule"]): r["detected"]
               for r in rows if r["arm"] == arm}
        shared = sorted(set(base) & set(cur))
        comparisons[arm] = M.compare([base[k] for k in shared],
                                     [cur[k] for k in shared])
        comparisons[arm]["n_paired"] = len(shared)

    out = {
        "stage": "K3", "experiment_id": "K3",
        "experiment_name": "Drift-Detection Root-Cause Investigation",
        "status": "completed", "seeds": list(seeds),
        "calibration_seeds": list(CAL_SEEDS),
        "fixed_thresholds": th,
        "kinds": [k.value for k in EVAL_KINDS], "arms": list(arms),
        "n_runs": len(rows), "episodes": rows,
        "by_arm": by_arm, "comparison_to_baseline": comparisons,
        "threshold_policy": "Thresholds fixed on calibration seeds before held-out evaluation.",
        "limitations": ["Feature summaries are DSL-input statistics, not learned embeddings."],
    }
    out.update(mon.finish())
    return out
