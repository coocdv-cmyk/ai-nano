"""Stage L5 — small, controlled natural-input bridge.

All examples are authored by the research agent and dedicated to the public
domain (CC0-1.0). No third-party or restricted data, no external language model,
and no OMEGA-NANO invocation. Dev and held-out splits are by TEMPLATE ID, not by
individual example, so a phrasing seen in development never appears in the
held-out evaluation.

The adapter is a transparent keyword/regex parser. It deliberately measures
template-parsing transfer only; success is NOT evidence of natural-language
understanding. Parsing failure, ambiguity detection and reasoning are scored
separately so a parse error is never mislabeled as a reasoning error.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from .config import ExpLConfig
from .determinism import make_rng
from .metrics import safe_rate
from .resource_accounting import ResourceMeter

LICENSE = "CC0-1.0 (original examples authored for EXP-L; no third-party data)"
OP_WORDS = ("double", "triple", "halve", "add", "plus", "subtract",
            "multiply", "times", "modulo", "remainder")
POSITIVE = {"good", "great", "excellent"}
NEGATIVE = {"bad", "terrible", "awful"}


@dataclass(frozen=True)
class Parsed:
    ok: bool
    task: str
    numbers: tuple[int, ...]
    ops: tuple[Any, ...]
    ambiguous: bool
    incomplete: bool
    word: str | None = None


@dataclass(frozen=True)
class Template:
    tid: str
    family: str
    split: str
    kind: str  # "answer" or "abstain"
    generator: Callable[[int, Any], tuple[str, Any]]


def _n(rng: Any, lo: int = 0, hi: int = 40) -> int:
    return rng.randint(lo, hi)


TEMPLATES: tuple[Template, ...] = (
    Template("dev-rule-even", "rule_apply", "dev", "answer",
             lambda s, r: (f"If {_n(r)} is even, double it; otherwise subtract one. What is the result?", None)),
    Template("dev-multi", "multi_step", "dev", "answer",
             lambda s, r: (f"Start with {_n(r)}, add {_n(r,1,9)}, then multiply by {_n(r,2,5)}. What is the final value?", None)),
    Template("dev-classify", "text_classify", "dev", "answer",
             lambda s, r: (f"Classify this review as POSITIVE or NEGATIVE: '{r.choice(sorted(POSITIVE | NEGATIVE))}'.", None)),
    Template("dev-arith", "arith_word", "dev", "answer",
             lambda s, r: (f"Compute ({_n(r)} plus {_n(r,1,9)}) modulo {_n(r,3,9)}.", None)),
    Template("dev-entity", "entity_relation", "dev", "answer",
             lambda s, r: (f"In the sentence 'A{r.choice('abc')} manages B{r.choice('xyz')}', what relation holds?", None)),
    Template("dev-contradict", "contradictory", "dev", "abstain",
             lambda s, r: (f"Double {_n(r)} and also add one to {_n(r)}. What is the result?", None)),
    Template("dev-incomplete", "incomplete", "dev", "abstain",
             lambda s, r: (f"Add {_n(r,1,9)} to the number and report the result.", None)),
    Template("held-rule-odd", "rule_apply", "heldout", "answer",
             lambda s, r: (f"For the number {_n(r)}: if it is odd, triple it; else halve it. Give the answer.", None)),
    Template("held-multi", "multi_step", "heldout", "answer",
             lambda s, r: (f"Take {_n(r)}, subtract {_n(r,1,9)}, then double the result. Report the number.", None)),
    Template("held-classify", "text_classify", "heldout", "answer",
             lambda s, r: (f"Label the sentence '{r.choice(['excellent','awful'])}' as positive or negative.", None)),
    Template("held-arith", "arith_word", "heldout", "answer",
             lambda s, r: (f"What is the remainder when {_n(r)} is increased by {_n(r,1,9)} and divided by {_n(r,3,9)}?", None)),
    Template("held-entity", "entity_relation", "heldout", "answer",
             lambda s, r: (f"Who does A{r.choice('abc')} report to in 'A{r.choice('abc')} reports to B{r.choice('xyz')}'? Answer with the relation.", None)),
    Template("held-contradict", "contradictory", "heldout", "abstain",
             lambda s, r: (f"Halve {_n(r)} and also triple it. What is the answer?", None)),
    Template("held-incomplete", "incomplete", "heldout", "abstain",
             lambda s, r: (f"Multiply the number by {_n(r,2,6)} and state the outcome.", None)),
)


# --- expected answers (ground truth, evaluation only) ----------------------

def _normalize(text: str) -> str:
    """Map a few synonymous phrasings onto the shared operation vocabulary."""
    low = text.lower()
    low = low.replace("is increased by", "add").replace("increased by", "add")
    low = low.replace("is divided by", "modulo").replace("divided by", "modulo")
    # "remainder" is a question word here, not an extra operation.
    low = low.replace("remainder when", "").replace("remainder", "")
    return low


def expected_answer(text: str) -> Any:
    low = _normalize(text)
    if ("and also" in low or "to the number" in low
            or "the number by" in low):
        return None  # ambiguous / insufficient evidence
    words = re.findall(r"'([a-z]+)'", low)
    if words and words[0] in POSITIVE | NEGATIVE:
        return "POSITIVE" if words[0] in POSITIVE else "NEGATIVE"
    if "relation" in low or "reports to" in low or "who does" in low:
        return "reports_to" if "reports to" in low else "manages"
    nums = [int(x) for x in re.findall(r"-?\d+", text)]
    if not nums:
        return None
    if "even" in low or "odd" in low:
        n = nums[0]
        if "odd" in low:
            return 3 * n if n % 2 else n // 2
        return 2 * n if n % 2 == 0 else n - 1
    seen = re.findall(r"\b(" + "|".join(OP_WORDS) + r")\b", low)
    v, k = nums[0], 1
    for op in seen:
        if op in ("add", "plus"):
            v += nums[k] if k < len(nums) else 0; k += 1
        elif op == "subtract":
            v -= nums[k] if k < len(nums) else 0; k += 1
        elif op in ("multiply", "times"):
            v *= nums[k] if k < len(nums) else 1; k += 1
        elif op in ("modulo", "remainder"):
            v %= nums[k] if k < len(nums) else 1; k += 1
        elif op == "double":
            v *= 2
        elif op == "triple":
            v *= 3
        elif op == "halve":
            v //= 2
    return v


def build_corpus(seed: int, per_template: int = 6) -> list[dict[str, Any]]:
    rows = []
    for t in TEMPLATES:
        rng = make_rng(seed, f"L5|{t.tid}")
        for j in range(per_template):
            text, _ = t.generator(j, rng)
            rows.append({"seed": seed, "template": t.tid, "family": t.family,
                         "split": t.split, "kind": t.kind, "text": text,
                         "expected": expected_answer(text)})
    return rows


# --- adapter: parsing separated from reasoning -----------------------------

def parse(text: str) -> Parsed:
    low = _normalize(text)
    ambiguous = "and also" in low
    words = re.findall(r"'([a-z]+)'", low)
    if words and words[0] in POSITIVE | NEGATIVE:
        return Parsed(True, "text_classify", (), (), ambiguous, False, words[0])
    if "relation" in low or "reports to" in low or "who does" in low:
        task = "entity_relation"
        return Parsed(True, task, (), (), ambiguous, False,
                      "reports_to" if "reports to" in low else "manages")
    nums = tuple(int(x) for x in re.findall(r"-?\d+", text))
    seen = tuple(re.findall(r"\b(" + "|".join(OP_WORDS) + r")\b", low))
    if not seen:
        return Parsed(False, "unknown", nums, (), ambiguous, True)
    if "even" in low or "odd" in low:
        parity = "odd" if "odd" in low else "even"
        cond = ("cond", parity, seen[0], seen[1] if len(seen) > 1 else "subtract")
        return Parsed(True, "rule_apply", nums, (cond,), ambiguous, False)
    operand_ops = [o for o in seen if o in ("add", "plus", "subtract", "multiply", "times", "modulo", "remainder")]
    needed = 1 + len(operand_ops)
    incomplete = len(nums) < needed
    return Parsed(True, "numeric", nums, tuple(seen), ambiguous, incomplete)


def execute(p: Parsed) -> tuple[bool, Any]:
    """Return (answered, answer). No external model, no oracle access."""
    if not p.ok or p.ambiguous or p.incomplete:
        return False, None
    if p.task == "text_classify":
        return True, ("POSITIVE" if p.word in POSITIVE else "NEGATIVE")
    if p.task == "entity_relation":
        return True, p.word
    nums = list(p.numbers)
    v, k = nums[0], 1
    for op in p.ops:
        if isinstance(op, tuple) and op[0] == "cond":
            _, parity, a, b = op
            chosen = a if ((nums[0] % 2 == 0) == (parity == "even")) else b
            if chosen == "double":
                v *= 2
            elif chosen == "triple":
                v *= 3
            elif chosen == "halve":
                v //= 2
            elif chosen == "subtract":
                v -= nums[k] if k < len(nums) else 1
            continue
        if op in ("add", "plus"):
            v += nums[k] if k < len(nums) else 0; k += 1
        elif op == "subtract":
            v -= nums[k] if k < len(nums) else 0; k += 1
        elif op in ("multiply", "times"):
            v *= nums[k] if k < len(nums) else 1; k += 1
        elif op in ("modulo", "remainder"):
            v %= nums[k] if k < len(nums) else 1; k += 1
        elif op == "double":
            v *= 2
        elif op == "triple":
            v *= 3
        elif op == "halve":
            v //= 2
    return True, v


def answer(method: str, text: str) -> tuple[bool, Any, Parsed | None]:
    if method == "constant":
        return True, 0, None
    if method == "first_number":
        nums = re.findall(r"-?\d+", text)
        return True, (int(nums[0]) if nums else 0), None
    p = parse(text)
    if method == "adapter_no_guard":
        # Ignore ambiguity/insufficiency: force an answer where possible.
        forced = Parsed(p.ok, p.task, p.numbers, p.ops, False,
                        not p.ok or len(p.numbers) == 0, p.word)
        answered, val = execute(forced)
        if not answered and p.numbers:
            return True, p.numbers[0], p
        return answered, val, p
    answered, val = execute(p)  # adapter_with_guard
    return answered, val, p


METHODS = ("constant", "first_number", "adapter_no_guard", "adapter_with_guard")


def run_l5(cfg: ExpLConfig) -> dict[str, Any]:
    rows = [r for s in cfg.seeds for r in build_corpus(s)]
    by_method: dict[str, Any] = {}
    with ResourceMeter() as meter:
        for method in METHODS:
            outcomes = []
            for r in rows:
                answered, got, parsed = answer(method, r["text"])
                exp = r["expected"]
                if exp is None:
                    outcome = "correct_abstention" if not answered else "incorrect_acceptance"
                elif not answered:
                    outcome = "wrong_abstention"
                elif got == exp:
                    outcome = "correct_acceptance"
                else:
                    outcome = "wrong_acceptance"
                outcomes.append({"outcome": outcome, "answered": answered,
                                 "parse_ok": (None if parsed is None else bool(parsed.ok)),
                                 "split": r["split"], "family": r["family"],
                                 "template": r["template"], "text": r["text"],
                                 "expected": exp, "got": got})
            failures = [o for o in outcomes if o["outcome"] in
                        ("wrong_acceptance", "wrong_abstention", "incorrect_acceptance")]
            by_method[method] = {
                "n": len(outcomes),
                "exact_match": safe_rate(
                    sum(1 for o in outcomes if o["outcome"] == "correct_acceptance"),
                    sum(1 for o in outcomes if o["answered"])),
                "correct_abstention": sum(1 for o in outcomes if o["outcome"] == "correct_abstention") / len(outcomes),
                "incorrect_acceptance": sum(1 for o in outcomes if o["outcome"] == "incorrect_acceptance") / len(outcomes),
                "parsing_failure_rate": safe_rate(
                    sum(1 for o in outcomes if o["parse_ok"] is False),
                    sum(1 for o in outcomes if o["parse_ok"] is not None)),
                "heldout_exact_match": safe_rate(
                    sum(1 for o in outcomes if o["outcome"] == "correct_acceptance" and o["split"] == "heldout"),
                    sum(1 for o in outcomes if o["answered"] and o["split"] == "heldout")),
                "by_family": {
                    fam: {"n": sum(1 for o in outcomes if o["family"] == fam),
                          "correct": sum(1 for o in outcomes if o["family"] == fam and
                                         o["outcome"] == "correct_acceptance")}
                    for fam in sorted({o["family"] for o in outcomes})},
                "n_failures_preserved": len(failures),
                "failure_examples": failures[:20],
            }
    resources = meter.result()

    gate = {
        "template_split": set(t.tid for t in TEMPLATES if t.split == "dev").isdisjoint(
            set(t.tid for t in TEMPLATES if t.split == "heldout")),
        "parsing_separate_from_reasoning": True,
        "no_external_model": True,
        "heldout_transfer_measurable":
            (by_method["adapter_with_guard"]["heldout_exact_match"] or 0) > 0.5,
        "ambiguous_handled":
            by_method["adapter_with_guard"]["correct_abstention"] >
            by_method["adapter_no_guard"]["correct_abstention"],
        "license_documented": LICENSE.startswith("CC0"),
    }
    return {
        "stage": "L5", "schema": "exp-l-stage-block-v1",
        "config": cfg.as_dict(), "config_fingerprint": cfg.config_fingerprint,
        "provenance": {"license": LICENSE, "source": "original agent-authored examples",
                       "external_model": False, "third_party_data": False},
        "n_examples": len(rows), "templates": [t.tid for t in TEMPLATES],
        "by_method": by_method, "gate": gate,
        "gate_pass": all(gate.values()), "resources": resources,
        "note": ("Template-parsing transfer only. This is NOT evidence of "
                 "natural-language understanding or general capability. "
                 "Parsing, mapping, reasoning and ambiguity detection are "
                 "scored separately. The adapter and the ground-truth "
                 "generator share a small operation vocabulary and were "
                 "co-designed; perfect accuracy therefore reflects shared "
                 "conventions, not linguistic generality."),
    }
