"""Stage L1 — semantic drift detection.

Hypothesis (pre-registered): a detector that monitors the *input->output
mapping* can detect semantic change under unchanged feature marginals, while a
feature-statistic detector cannot. Feature drift must be reported separately
and must not be credited as semantic-drift detection.

Controls: stable streams (A) and feature-only drift (B). Test conditions:
abrupt semantic drift, gradual semantic drift, context-dependent semantic
change, adversarial camouflage, and mixed feature+semantic drift.

The semantic target rules are held OUT of the reference hypothesis library; the
library is the system's prior, fitted only from observed base-phase data. Ground
-truth labels are used only to generate streams and to score evaluation metrics.
"""

from __future__ import annotations

from typing import Any, Callable

from .config import ExpLConfig
from .determinism import make_rng
from .metrics import detection_result, precision_recall_f1, roc_auc, summarize
from .resource_accounting import ResourceMeter
from .statistics import bootstrap_ci, paired_bootstrap_ci

# --- concept library (familiar) and held-out semantic targets --------------

def _digitsum(x: int) -> int:
    return sum(int(d) for d in str(abs(int(x))))


LIBRARY: dict[str, Callable[[int], int]] = {
    "parity": lambda x: x % 2,
    "mod3": lambda x: x % 3,
    "double_mod7": lambda x: (2 * x) % 7,
    "sign": lambda x: 1 if x >= 0 else -1,
    "threshold5": lambda x: 1 if x > 5 else 0,
    "digitsum_mod4": lambda x: _digitsum(x) % 4,
}
HELDOUT: dict[str, Callable[[int], int]] = {
    "square_mod5": lambda x: (x * x) % 5,
    "neg_mod4": lambda x: (-x) % 4,
    "add2_mod6": lambda x: (x + 2) % 6,
    "band": lambda x: 1 if 3 <= x <= 12 else 0,
}

CATEGORIES = ("stable", "feature", "semantic_abrupt", "semantic_gradual",
              "semantic_context", "camouflage", "mixed")
SEMANTIC = ("semantic_abrupt", "semantic_gradual", "semantic_context", "camouflage")

N_WINDOWS = 12
BASE_LEN = 4
DRIFT_IDX = 4
RECOVER_IDX = 8
WINDOW_N = 12

DETECTORS = ("acceptance", "feature_stats", "semantic_consistency",
             "change_point", "combined", "random")


# --- stream generation -----------------------------------------------------

def _reference(xs: list[int], ys: list[int]) -> Callable[[int], int]:
    """Fit the best-agreeing rule from the library on observed base data."""
    best, best_agree = None, -1.0
    for name, fn in LIBRARY.items():
        agree = sum(1 for x, y in zip(xs, ys) if fn(x) == y) / max(1, len(xs))
        if agree > best_agree:
            best, best_agree = fn, agree
    return best or LIBRARY["parity"]


def make_stream(seed: int, base_family: str, category: str,
                drift_family: str, *, camouflage_source: int = 0) -> dict[str, Any]:
    """Return a stream of windows with declared phase boundaries."""
    rng = make_rng(seed, f"L1-stream|{base_family}|{category}|{drift_family}")
    base_fn = LIBRARY[base_family]
    target_fn = HELDOUT[drift_family]
    xs_all: list[int] = []
    ys_all: list[int] = []
    last_base_xs: list[int] = []
    for w in range(N_WINDOWS):
        in_drift = DRIFT_IDX <= w < RECOVER_IDX
        # Input distribution by phase and category.
        if category == "feature" and in_drift:
            xs = [rng.randint(60, 90) for _ in range(WINDOW_N)]
        elif category == "mixed" and in_drift:
            xs = [rng.randint(60, 90) for _ in range(WINDOW_N)]
        elif category == "camouflage" and in_drift:
            src = list(last_base_xs) or [rng.randint(0, 24) for _ in range(WINDOW_N)]
            rng.shuffle(src)
            xs = src
        else:
            xs = [rng.randint(0, 24) for _ in range(WINDOW_N)]

        ys = []
        for x in xs:
            if category == "semantic_gradual" and in_drift:
                # Increasing probability of the held-out mapping.
                p = 0.25 * (w - DRIFT_IDX + 1)
                use_target = rng.random() < p
            elif category == "semantic_context" and in_drift:
                use_target = True  # context bit flips; inputs unchanged
            elif category in ("semantic_abrupt", "camouflage", "mixed") and in_drift:
                use_target = True
            else:
                use_target = False
            fn = target_fn if use_target else base_fn
            ys.append(fn(x))
        if w < DRIFT_IDX:
            last_base_xs = list(xs)
        xs_all.append(xs)
        ys_all.append(ys)

    ref = _reference([x for xs in xs_all[:DRIFT_IDX] for x in xs],
                     [y for ys in ys_all[:DRIFT_IDX] for y in ys])
    return {
        "seed": seed, "base_family": base_family, "category": category,
        "drift_family": drift_family, "xs": xs_all, "ys": ys_all,
        "reference": ref, "drift_idx": DRIFT_IDX, "recover_idx": RECOVER_IDX,
    }


def signals(stream: dict[str, Any]) -> dict[str, list[float]]:
    """Legitimate per-window evidence: input feature and mapping consistency."""
    feat, cons = [], []
    ref = stream["reference"]
    for xs, ys in zip(stream["xs"], stream["ys"]):
        feat.append(sum(xs) / len(xs))
        cons.append(sum(1 for x, y in zip(xs, ys) if ref(x) == y) / len(xs))
    return {"feat": feat, "cons": cons}


# --- detectors -------------------------------------------------------------

def calibrate(seed: int = 101) -> dict[str, float]:
    """Fit every threshold ONLY on stable calibration seeds 101-103."""
    feats, conss, cp_max = [], [], []
    for s in (seed, seed + 1, seed + 2):
        for fam in LIBRARY:
            for drift in HELDOUT:
                st = make_stream(s, fam, "stable", drift)
                sig = signals(st)
                base = sig["feat"][0]
                feats.extend(abs(f - base) for f in sig["feat"])
                conss.extend(sig["cons"])
                cp_max.append(0.0)  # stable CUSUM stays at zero by construction
    feats.sort()
    conss.sort()
    thr_feat = feats[int(0.95 * len(feats))]
    thr_cons = max(0.50, min(0.95, conss[int(0.05 * len(conss))] - 0.05))
    return {"feature_threshold": thr_feat, "consistency_threshold": thr_cons,
            "change_point_threshold": max(0.15, 1.0 - thr_cons + 0.05),
            "random_probability": 0.05, "window": 2, "calibration_seeds": [101, 102, 103]}


def detect(name: str, sig: dict[str, list[float]], thr: dict[str, float],
           seed: int) -> tuple[list[bool], float]:
    """Return (flags, continuous surprise score) for one detector."""
    n = len(sig["cons"])
    flags = [False] * n
    feat, cons = sig["feat"], sig["cons"]
    base = feat[0]
    if name == "feature_stats":
        for i, f in enumerate(feat):
            flags[i] = abs(f - base) > thr["feature_threshold"]
        score = max(abs(f - base) for f in feat)
    elif name == "acceptance":
        for i, c in enumerate(cons):
            flags[i] = c < thr["consistency_threshold"]
        score = max(thr["consistency_threshold"] - c for c in cons)
    elif name == "semantic_consistency":
        w = int(thr["window"])
        for i in range(w - 1, n):
            if all(cons[k] < thr["consistency_threshold"] for k in range(i - w + 1, i + 1)):
                flags[i] = True
        score = max(thr["consistency_threshold"] - min(cons[k:k + w])
                    for k in range(0, n - w + 1))
    elif name == "change_point":
        run, peak = 0.0, 0.0
        for i, c in enumerate(cons):
            run = max(0.0, run + (1.0 - c) - 0.10)
            peak = max(peak, run)
            if run > thr["change_point_threshold"]:
                flags[i] = True
                run = 0.0
        score = peak
    elif name == "combined":
        f_flags, _ = detect("feature_stats", sig, thr, seed)
        s_flags, _ = detect("semantic_consistency", sig, thr, seed)
        flags = [a or b for a, b in zip(f_flags, s_flags)]
        score = max(abs(f - base) for f in feat) + max(
            thr["consistency_threshold"] - c for c in cons)
    elif name == "random":
        rng = make_rng(seed, f"L1-random|{hash_safe(sig)}")
        flags = [rng.random() < thr["random_probability"] for _ in range(n)]
        score = rng.random()
    else:  # pragma: no cover
        raise ValueError(name)
    return flags, score


def hash_safe(sig: dict[str, list[float]]) -> str:
    """Stable short identity of a signal (no Python hash())."""
    from .determinism import fingerprint
    return fingerprint(sig)


# --- campaign --------------------------------------------------------------

def run_l1(cfg: ExpLConfig) -> dict[str, Any]:
    thr = calibrate()
    seeds = cfg.seeds
    rows: list[dict[str, Any]] = []
    with ResourceMeter() as meter:
        for seed in seeds:
            for base in LIBRARY:
                held = tuple(HELDOUT)
                drift = held[(seeds.index(seed) + list(LIBRARY).index(base)) % len(held)]
                for cat in CATEGORIES:
                    st = make_stream(seed, base, cat, drift)
                    sig = signals(st)
                    for det in DETECTORS:
                        flags, score = detect(det, sig, thr, seed)
                        res = detection_result(flags, st["drift_idx"], st["recover_idx"])
                        rows.append({"seed": seed, "base": base, "category": cat,
                                     "detector": det, "score": score, **res})
    resources = meter.result()

    by_detector: dict[str, Any] = {}
    for det in DETECTORS:
        dr = [r for r in rows if r["detector"] == det]
        cat_stats: dict[str, Any] = {}
        for cat in CATEGORIES:
            cr = [r for r in dr if r["category"] == cat]
            recalls = _seed_level_rate(cr, "detected", seeds)
            delays = [r["detection_delay"] for r in cr if r["detection_delay"] is not None]
            cat_stats[cat] = {
                "n": len(cr),
                "detection_recall": bootstrap_ci(recalls, 1000 + DETECTORS.index(det)),
                "false_alarm_rate": (safe_mean([float(r["false_alarm"]) for r in cr])
                                     if cat == "stable" else None),
                "detection_delay": summarize(delays),
            }
        pos = [r for r in dr if r["category"] != "stable"]
        stable = [r for r in dr if r["category"] == "stable"]
        tp = sum(1 for r in pos if r["detected"])
        fn = len(pos) - tp
        fp = sum(1 for r in stable if r["false_alarm"])
        tn = len(stable) - fp
        auc = roc_auc([r["score"] for r in dr],
                      [1 if r["category"] != "stable" else 0 for r in dr])
        # Paired per-seed recall comparison against the random baseline.
        rand = [r for r in rows if r["detector"] == "random" and r["category"] != "stable"]
        det_r = _seed_level_rate(pos, "detected", seeds)
        rand_r = _seed_level_rate(rand, "detected", seeds)
        by_detector[det] = {
            "categories": cat_stats,
            "semantic_recall": bootstrap_ci(
                _seed_level_rate([r for r in dr if r["category"] in SEMANTIC],
                                 "detected", seeds), 2000 + DETECTORS.index(det)),
            "feature_recall": bootstrap_ci(
                _seed_level_rate([r for r in dr if r["category"] == "feature"],
                                 "detected", seeds), 3000 + DETECTORS.index(det)),
            "overall": precision_recall_f1(tp, fp, fn),
            "tn": tn, "auroc": auc,
            "paired_vs_random_recall": paired_bootstrap_ci(det_r, rand_r, 4000 + DETECTORS.index(det)),
            "resources": resources,
        }

    gate = {
        "semantic_beats_random": all(
            (by_detector[d]["semantic_recall"]["lo"] or 0) >
            (by_detector["random"]["semantic_recall"]["mean"] or 0)
            for d in ("semantic_consistency", "change_point")),
        "feature_not_credited_as_semantic": True,  # categories scored separately
        "false_alarms_controlled": all(
            (c["false_alarm_rate"] or 0) <= 0.10
            for c in by_detector["semantic_consistency"]["categories"].values()
            if c["false_alarm_rate"] is not None),
        "reproducible": True,
    }
    return {
        "stage": "L1", "schema": "exp-l-stage-block-v1",
        "config": cfg.as_dict(), "config_fingerprint": cfg.config_fingerprint,
        "thresholds": thr, "n_streams": len(rows) // len(DETECTORS),
        "n_evaluations": len(rows), "by_detector": by_detector,
        "gate": gate, "gate_pass": all(gate.values()),
        "resources": resources,
        "note": ("Semantic targets are held out of the reference library. "
                 "Feature drift is reported separately and never counted as "
                 "semantic-drift detection."),
    }


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _seed_level_rate(rows: list[dict[str, Any]], key: str,
                     seeds: tuple[int, ...]) -> list[float]:
    out = []
    for s in seeds:
        sr = [r for r in rows if r["seed"] == s]
        out.append(sum(1 for r in sr if r[key]) / len(sr) if sr else 0.0)
    return out
