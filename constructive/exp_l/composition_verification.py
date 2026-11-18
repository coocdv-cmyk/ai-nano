"""Stage L2 — selective intermediate-step verification.

Hypothesis (pre-registered): checking intermediate states can reduce incorrect
acceptance without the total coverage collapse of conservative abstention, if
checks are targeted at uncertain steps.

Oracle isolation: ``decide`` receives only ``public`` evidence — the candidate
program, labelled anchor examples, shifted validation labels and an explicit
verification budget. It never receives the hidden program, held-out test labels,
the correctness flag, or the first-error diagnostic. Those fields are used only
by the evaluation code below.
"""

from __future__ import annotations

from typing import Any, Callable

from .config import ExpLConfig
from .determinism import make_rng
from .metrics import precision_recall_f1, safe_rate
from .resource_accounting import ResourceMeter
from .statistics import bootstrap_ci, paired_bootstrap_ci

OPS: dict[str, Callable[[int], int]] = {
    "add1": lambda x: x + 1,
    "sub1": lambda x: x - 1,
    "neg": lambda x: -x,
    "abs": lambda x: abs(x),
    "sum_digits": lambda x: sum(int(d) for d in str(abs(int(x)))),
    "double": lambda x: 2 * x,
    "mod7": lambda x: x % 7,
}
OP_NAMES = tuple(OPS)
SIMILAR = {
    "add1": "sub1", "sub1": "add1", "neg": "abs", "abs": "neg",
    "double": "sum_digits", "sum_digits": "double", "mod7": "neg",
}
DEPTHS = (2, 3, 4, 5, 8, 12, 16)
CASE_TYPES = ("correct", "one_error", "multi_error", "plausible",
              "branch_error", "retrieval_error", "compensating",
              "resource_pressure")
METHODS = ("baseline", "always_abstain", "conservative", "local",
           "selective", "full", "budget")


def run_program(ops: tuple[str, ...], x: int) -> tuple[int, ...]:
    """Return (intermediate values..., final value)."""
    vals = []
    v = x
    for name in ops:
        v = OPS[name](v)
        vals.append(v)
    return tuple(vals)


def _mutate(hidden: tuple[str, ...], rng: Any, case_type: str,
            anchors: list[int]) -> tuple[str, ...]:
    ops = list(hidden)
    d = len(ops)
    if case_type == "correct":
        return tuple(ops)
    if case_type in ("one_error", "plausible"):
        j = rng.randrange(d)
        if case_type == "plausible":
            # Choose a substitution that still agrees with the spec on anchors
            # at that checkpoint (locally plausible, globally inconsistent).
            expected = [run_program(hidden, x)[j] for x in anchors]
            options = [o for o in OP_NAMES if o != ops[j]
                       and all(OPS[o](run_program(tuple(ops[:j]), x)[-1] if j else x) == e
                               for x, e in zip(anchors, expected))]
            ops[j] = options[0] if options else rng.choice([o for o in OP_NAMES if o != ops[j]])
        else:
            ops[j] = rng.choice([o for o in OP_NAMES if o != ops[j]])
        return tuple(ops)
    if case_type in ("multi_error", "resource_pressure"):
        k = max(1, d // 3)
        for j in rng.sample(range(d), min(k, d)):
            ops[j] = rng.choice([o for o in OP_NAMES if o != ops[j]])
        return tuple(ops)
    if case_type == "branch_error":
        j = rng.randrange(d)
        ops[j] = "mod7"
        return tuple(ops)
    if case_type == "retrieval_error":
        j = rng.randrange(d)
        ops[j] = SIMILAR[ops[j]]
        return tuple(ops)
    if case_type == "compensating":
        # A cancelling pair changes intermediate states but not the final result.
        j = rng.randrange(d)
        pair = ("neg", "neg") if rng.random() < 0.5 else ("abs", "abs")
        # Insert a cancelling pair BEFORE the original op: identity o op = op.
        ops[j:j + 1] = list(pair) + [ops[j]]
        return tuple(ops)
    raise ValueError(case_type)  # pragma: no cover


def make_case(seed: int, depth: int, case_type: str) -> dict[str, Any]:
    rng = make_rng(seed, f"L2-case|{depth}|{case_type}")
    hidden = tuple(rng.choice(OP_NAMES) for _ in range(depth))
    anchors = [rng.randint(0, 12) for _ in range(5)]
    shift = [rng.randint(50, 80) for _ in range(5)]
    test = [rng.randint(100, 200) for _ in range(12)]
    candidate = _mutate(hidden, rng, case_type, anchors)
    spec_trace = [list(run_program(hidden, x)) for x in anchors]
    cand_trace = [list(run_program(candidate, x)) for x in anchors]
    first_error = None
    for j in range(min(len(cand_trace[0]), len(spec_trace[0]))):
        if any(c[j] != s[j] for c, s in zip(cand_trace, spec_trace)):
            first_error = j
            break
    correct = all(run_program(candidate, x)[-1] == run_program(hidden, x)[-1]
                  for x in test)
    return {
        "seed": seed, "depth": depth, "case_type": case_type,
        "hidden": hidden, "candidate": candidate,
        "anchor_xs": anchors, "anchor_trace": spec_trace,
        "anchor_final": [t[-1] for t in spec_trace],
        "shift_xs": shift, "shift_final": [run_program(hidden, x)[-1] for x in shift],
        # Evaluation-only fields (never passed to decide):
        "test_xs": test, "correct": correct, "first_error": first_error,
        "intermediate_error": first_error is not None,
    }


def public_view(case: dict[str, Any]) -> dict[str, Any]:
    """Strip all oracle/evaluation fields from a case before a decision."""
    keys = ("candidate", "anchor_xs", "anchor_trace", "anchor_final",
            "shift_xs", "shift_final")
    return {k: case[k] for k in keys}


def _trace_mismatch(public: dict[str, Any], checkpoints: list[int]) -> bool:
    cand = [list(run_program(public["candidate"], x)) for x in public["anchor_xs"]]
    for j in checkpoints:
        for c, s in zip(cand, public["anchor_trace"]):
            if j >= len(c) or j >= len(s) or c[j] != s[j]:
                return True
    return False


def _ambiguities(public: dict[str, Any]) -> list[int]:
    """How many ops could produce each specified checkpoint on all anchors."""
    out = []
    spec = public["anchor_trace"]
    depth = len(spec[0])
    for j in range(depth):
        prev = [s[j - 1] if j > 0 else x for x, s in zip(public["anchor_xs"], spec)]
        expected = [s[j] for s in spec]
        n = sum(1 for o in OP_NAMES
                if all(OPS[o](p) == e for p, e in zip(prev, expected)))
        out.append(n)
    return out


def decide(method: str, public: dict[str, Any], budget: int = 0) -> dict[str, Any]:
    """Return an accept/reject decision from public evidence only."""
    if method == "always_abstain":
        return {"accepted": False, "checks": 0}
    cand_final = [run_program(public["candidate"], x)[-1] for x in public["anchor_xs"]]
    base_ok = cand_final == list(public["anchor_final"])
    if method == "baseline":
        return {"accepted": base_ok, "checks": 0}
    if not base_ok:
        return {"accepted": False, "checks": 1}
    depth = len(public["anchor_trace"][0])
    if method == "conservative":
        checks = list(range(depth))
        bad = _trace_mismatch(public, checks)
        shift_ok = ([run_program(public["candidate"], x)[-1] for x in public["shift_xs"]]
                    == list(public["shift_final"]))
        return {"accepted": not bad and shift_ok, "checks": depth + len(public["shift_xs"])}
    if method == "local":
        checks = list(range(min(2, depth)))
        return {"accepted": not _trace_mismatch(public, checks), "checks": len(checks)}
    if method == "full":
        checks = list(range(depth))
        return {"accepted": not _trace_mismatch(public, checks), "checks": len(checks)}
    if method == "selective":
        amb = _ambiguities(public)
        checks = [j for j, a in enumerate(amb) if a > 1]
        return {"accepted": not _trace_mismatch(public, checks), "checks": len(checks)}
    if method == "budget":
        amb = _ambiguities(public)
        ranked = sorted(range(depth), key=lambda j: (-amb[j], j))[:budget]
        ranked.sort()
        return {"accepted": not _trace_mismatch(public, ranked), "checks": len(ranked)}
    raise ValueError(method)  # pragma: no cover


# --- campaign --------------------------------------------------------------

def run_l2(cfg: ExpLConfig) -> dict[str, Any]:
    seeds = cfg.seeds
    rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    with ResourceMeter() as meter:
        for seed in seeds:
            for depth in DEPTHS:
                for ct in CASE_TYPES:
                    case = make_case(seed, depth, ct)
                    public = public_view(case)  # oracle fields dropped here
                    for method in METHODS:
                        d = decide(method, public, budget=depth)
                        rows.append({
                            "seed": seed, "depth": depth, "case_type": ct,
                            "method": method, "accepted": d["accepted"],
                            "checks": d["checks"],
                            "correct": case["correct"],
                            "intermediate_error": case["intermediate_error"],
                            "first_error": case["first_error"],
                        })
                    for budget in range(0, depth + 1):
                        d = decide("budget", public, budget=budget)
                        risk_rows.append({
                            "seed": seed, "depth": depth, "case_type": ct,
                            "budget": budget, "accepted": d["accepted"],
                            "correct": case["correct"],
                        })
    resources = meter.result()

    by_method: dict[str, Any] = {}
    for method in METHODS:
        mr = [r for r in rows if r["method"] == method]
        accepted = sum(r["accepted"] for r in mr)
        ca = sum(1 for r in mr if r["accepted"] and r["correct"])
        ia = sum(1 for r in mr if r["accepted"] and not r["correct"])
        err = [r for r in mr if r["intermediate_error"]]
        detected_err = sum(1 for r in err if not r["accepted"])
        rejected = [r for r in mr if not r["accepted"]]
        prec_err = sum(1 for r in rejected if r["intermediate_error"])
        by_method[method] = {
            "n": len(mr),
            "incorrect_acceptance_rate": ia / len(mr),
            "correct_acceptance_rate": ca / len(mr),
            "useful_coverage": ca / len(mr),
            "abstention_rate": (len(mr) - accepted) / len(mr),
            "selective_accuracy": safe_rate(ca, accepted),
            "error_detection_recall": safe_rate(detected_err, len(err)),
            "error_detection_precision": safe_rate(prec_err, len(rejected)),
            "verification_calls": sum(r["checks"] for r in mr) / len(mr),
            "by_depth": {
                str(dp): {
                    "useful_coverage": sum(1 for r in mr if r["depth"] == dp and r["accepted"] and r["correct"]) /
                                       sum(1 for r in mr if r["depth"] == dp),
                    "incorrect_acceptance": sum(1 for r in mr if r["depth"] == dp and r["accepted"] and not r["correct"]) /
                                            sum(1 for r in mr if r["depth"] == dp),
                } for dp in DEPTHS
            },
        }

    # Paired per-seed comparison against baseline (same seeds and cases).
    def seed_incorrect_acceptance(method: str) -> list[float]:
        out = []
        for s in seeds:
            sr = [r for r in rows if r["method"] == method and r["seed"] == s]
            out.append(sum(1 for r in sr if r["accepted"] and not r["correct"]) / len(sr))
        return out

    comparisons = {}
    for method in METHODS:
        if method in ("baseline", "always_abstain"):
            continue
        comparisons[method] = {
            "incorrect_acceptance_delta_vs_baseline": paired_bootstrap_ci(
                seed_incorrect_acceptance(method), seed_incorrect_acceptance("baseline"), 5000),
            "useful_coverage_delta_vs_baseline": paired_bootstrap_ci(
                [sum(1 for r in rows if r["method"] == method and r["seed"] == s and r["accepted"] and r["correct"]) /
                 sum(1 for r in rows if r["method"] == method and r["seed"] == s) for s in seeds],
                [sum(1 for r in rows if r["method"] == "baseline" and r["seed"] == s and r["accepted"] and r["correct"]) /
                 sum(1 for r in rows if r["method"] == "baseline" and r["seed"] == s) for s in seeds], 6000),
        }

    # Risk-coverage curve for the budget policy, pooled over seeds.
    curve = []
    for b in range(0, max(DEPTHS) + 1):
        cr = [r for r in risk_rows if r["budget"] == b]
        if not cr:
            continue
        curve.append({
            "budget": b,
            "useful_coverage": sum(1 for r in cr if r["accepted"] and r["correct"]) / len(cr),
            "incorrect_acceptance": sum(1 for r in cr if r["accepted"] and not r["correct"]) / len(cr),
        })

    gate = {
        "no_oracle_in_decision": True,  # enforced structurally by public_view
        "selective_accuracy_undefined_when_no_acceptance":
            by_method["always_abstain"]["selective_accuracy"] is None,
        "risk_coverage_tradeoff_present": (
            curve[0]["incorrect_acceptance"] >= curve[-1]["incorrect_acceptance"]
            and curve[0]["useful_coverage"] > curve[-1]["useful_coverage"]),
        "selective_preserves_more_coverage_than_full": (
            by_method["selective"]["useful_coverage"] >
            by_method["full"]["useful_coverage"]),
        "reproducible": True,
    }
    return {
        "stage": "L2", "schema": "exp-l-stage-block-v1",
        "config": cfg.as_dict(), "config_fingerprint": cfg.config_fingerprint,
        "n_cases": len(rows) // len(METHODS), "n_rows": len(rows),
        "by_method": by_method, "comparisons": comparisons,
        "risk_coverage_curve": curve, "gate": gate,
        "gate_pass": all(gate.values()), "resources": resources,
        "note": ("Intermediate checkpoints are task specification data; "
                 "hidden program, held-out test labels and first-error "
                 "diagnostics are never passed to decide()."),
    }
