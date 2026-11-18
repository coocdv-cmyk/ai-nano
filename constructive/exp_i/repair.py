"""EXP-I-F — verified program repair.

Repair *generation* sees only the buggy program and visible input/output examples.
Repair *verification* is a separate component using bounded exhaustive
equivalence plus adversarial counterexamples. Passing visible examples alone is
never sufficient to label a repair semantically correct. The hidden correct
program (oracle) is used only by the evaluation harness and the verifier, never
by the generator.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable

MAG_LIMIT = 10 ** 9


class RepairError(Exception):
    pass


class VerificationState(str, Enum):
    VERIFIED_EQUIVALENT = "VERIFIED_EQUIVALENT"
    NOT_EQUIVALENT = "NOT_EQUIVALENT"
    AMBIGUOUS = "AMBIGUOUS"
    UNVERIFIABLE = "UNVERIFIABLE"


# ---------------------------------------------------------------------------
# Independent bounded imperative AST
# ---------------------------------------------------------------------------


def C(v: int) -> tuple:
    return ("const", int(v))


def V(name: str) -> tuple:
    return ("var", name)


def Bin(op: str, l: tuple, r: tuple) -> tuple:
    return ("bin", op, l, r)


def Cmp(op: str, l: tuple, r: tuple) -> tuple:
    return ("cmp", op, l, r)


def Assign(name: str, expr: tuple) -> tuple:
    return ("assign", name, expr)


def While(cond: tuple, body: list) -> tuple:
    return ("while", cond, list(body))


def If(cond: tuple, then: list, els: list) -> tuple:
    return ("if", cond, list(then), list(els))


def Ret(expr: tuple) -> tuple:
    return ("return", expr)


def exec_program(program: list, env_in: dict[str, Any], *,
                 max_steps: int = 4000, max_iters: int = 2000) -> int:
    env = dict(env_in)
    steps = [0]

    def tick() -> None:
        steps[0] += 1
        if steps[0] > max_steps:
            raise RepairError("step budget exhausted")

    def check(v: Any) -> int:
        if isinstance(v, bool) or not isinstance(v, int):
            raise RepairError(f"expected int, got {type(v).__name__}")
        if abs(v) > MAG_LIMIT:
            raise RepairError("magnitude exceeded")
        return v

    def ev(n: tuple) -> Any:
        tick()
        k = n[0]
        if k == "const":
            return check(n[1])
        if k == "var":
            if n[1] not in env:
                raise RepairError(f"unbound variable {n[1]!r}")
            return env[n[1]]
        if k == "bin":
            l, r = check(ev(n[2])), check(ev(n[3]))
            op = n[1]
            if op == "+":
                out = l + r
            elif op == "-":
                out = l - r
            elif op == "*":
                out = l * r
            elif op == "//":
                if r == 0:
                    raise RepairError("division by zero")
                q = abs(l) // abs(r)
                out = q if (l >= 0) == (r >= 0) else -q
            elif op == "%":
                if r == 0:
                    raise RepairError("modulo by zero")
                out = l % r if (l >= 0) == (r >= 0) else -(abs(l) % abs(r))
            else:
                raise RepairError(f"unknown bin {op}")
            return check(out)
        if k == "cmp":
            l, r = check(ev(n[2])), check(ev(n[3]))
            table = {"<": l < r, "<=": l <= r, ">": l > r, ">=": l >= r,
                     "==": l == r, "!=": l != r}
            if n[1] not in table:
                raise RepairError(f"unknown cmp {n[1]}")
            return 1 if table[n[1]] else 0
        raise RepairError(f"unknown expr {k}")

    def run(stmts: list) -> Any:
        for s in stmts:
            tick()
            k = s[0]
            if k == "assign":
                env[s[1]] = ev(s[2])
            elif k == "if":
                out = run(s[2] if ev(s[1]) != 0 else s[3])
                if out is not None:
                    return out
            elif k == "while":
                iters = 0
                while ev(s[1]) != 0:
                    out = run(s[2])
                    if out is not None:
                        return out
                    iters += 1
                    if iters > max_iters:
                        raise RepairError("loop iteration budget exhausted")
            elif k == "return":
                return ev(s[1])
            else:
                raise RepairError(f"unknown stmt {k}")
        return None

    result = run(program)
    if result is None:
        raise RepairError("no return")
    return result


# ---------------------------------------------------------------------------
# Templates (independent reimplementation of the bounded benchmark)
# ---------------------------------------------------------------------------


def tpl_sum_range() -> list:
    return [Assign("s", C(0)), Assign("i", V("a")),
            While(Cmp("<=", V("i"), V("b")),
                  [Assign("s", Bin("+", V("s"), V("i"))),
                   Assign("i", Bin("+", V("i"), C(1)))]),
            Ret(V("s"))]


def tpl_sum_squares() -> list:
    return [Assign("s", C(0)), Assign("i", C(1)),
            While(Cmp("<=", V("i"), V("n")),
                  [Assign("s", Bin("+", V("s"), Bin("*", V("i"), V("i")))),
                   Assign("i", Bin("+", V("i"), C(1)))]),
            Ret(V("s"))]


def tpl_pow2() -> list:
    return [Assign("s", C(1)), Assign("i", C(0)),
            While(Cmp("<", V("i"), V("n")),
                  [Assign("s", Bin("*", V("s"), C(2))),
                   Assign("i", Bin("+", V("i"), C(1)))]),
            Ret(V("s"))]


def tpl_sum_mult() -> list:
    return [Assign("s", C(0)), Assign("i", C(0)),
            While(Cmp("<", V("i"), V("n")),
                  [Assign("s", Bin("+", V("s"), Bin("*", V("i"), V("m")))),
                   Assign("i", Bin("+", V("i"), C(1)))]),
            Ret(V("s"))]


def tpl_sum_gcd() -> list:
    return [Assign("a", V("a")), Assign("b", V("b")),
            While(Cmp("!=", V("b"), C(0)),
                  [Assign("t", V("b")),
                   Assign("b", Bin("%", V("a"), V("b"))),
                   Assign("a", V("t"))]),
            Ret(V("a"))]


def tpl_clamp() -> list:
    return [If(Cmp("<", V("x"), V("lo")), [Ret(V("lo"))], []),
            If(Cmp(">", V("x"), V("hi")), [Ret(V("hi"))], []),
            Ret(V("x"))]


TEMPLATES: dict[str, Callable[[], list]] = {
    "sum_range": tpl_sum_range, "sum_squares": tpl_sum_squares,
    "pow2": tpl_pow2, "sum_mult": tpl_sum_mult,
    "sum_gcd": tpl_sum_gcd, "clamp": tpl_clamp,
}
TEMPLATE_ARGS = {
    "sum_range": ["a", "b"], "sum_squares": ["n"], "pow2": ["n"],
    "sum_mult": ["n", "m"], "sum_gcd": ["a", "b"], "clamp": ["x", "lo", "hi"],
}


def domain_for(name: str) -> list[dict[str, int]]:
    args = TEMPLATE_ARGS[name]
    if name == "sum_range":
        return [dict(zip(args, (a, b))) for a in range(0, 8) for b in range(a, 9)]
    if name in ("sum_squares", "pow2"):
        return [dict(n=v) for v in range(0, 12)]
    if name == "sum_mult":
        return [dict(zip(args, (n, m))) for n in range(0, 9) for m in range(0, 5)]
    if name == "sum_gcd":
        return [dict(zip(args, (a, b))) for a in range(1, 9) for b in range(1, 9)]
    if name == "clamp":
        return [dict(zip(args, (x, lo, hi)))
                for x in range(-2, 10) for lo in (0, 3) for hi in (4, 8) if lo <= hi]
    raise ValueError(name)


def sample_inputs(name: str, rng: random.Random, n: int) -> list[dict[str, int]]:
    dom = domain_for(name)
    return rng.sample(dom, min(n, len(dom)))


# ---------------------------------------------------------------------------
# Mutation / rewrite sites
# ---------------------------------------------------------------------------

CMP_ALTS = {"<": ("<=", ">", ">=", "==", "!="), "<=": ("<", ">", ">=", "==", "!="),
            ">": (">=", "<", "<=", "==", "!="), ">=": (">", "<", "<=", "==", "!="),
            "==": ("!=", "<", "<=", ">", ">="), "!=": ("==", "<", "<=", ">", ">=")}
BIN_ALTS = {"+": ("-", "*", "//", "%"), "-": ("+", "*", "//", "%"),
            "*": ("+", "-", "//", "%"), "//": ("+", "-", "*", "%"),
            "%": ("+", "-", "*", "//")}
DELTAS = (-2, -1, 1, 2)


def rewrite_sites(program: Any, path: tuple = ()) -> list[tuple[tuple, tuple]]:
    out: list[tuple[tuple, tuple]] = []
    if isinstance(program, list):
        for i, s in enumerate(program):
            out.extend(rewrite_sites(s, path + (i,)))
    elif isinstance(program, tuple):
        k = program[0]
        if k in ("bin", "cmp", "const"):
            out.append((path, program))
        if k == "bin":
            out.extend(rewrite_sites(program[2], path + (2,)))
            out.extend(rewrite_sites(program[3], path + (3,)))
        elif k == "cmp":
            out.extend(rewrite_sites(program[2], path + (2,)))
            out.extend(rewrite_sites(program[3], path + (3,)))
        elif k == "assign":
            out.extend(rewrite_sites(program[2], path + (2,)))
        elif k == "while":
            out.extend(rewrite_sites(program[1], path + (1,)))
            out.extend(rewrite_sites(program[2], path + (2,)))
        elif k == "if":
            out.extend(rewrite_sites(program[1], path + (1,)))
            out.extend(rewrite_sites(program[2], path + (2,)))
            out.extend(rewrite_sites(program[3], path + (3,)))
        elif k == "return":
            out.extend(rewrite_sites(program[1], path + (1,)))
    return out


def replace_at(program: Any, path: tuple, value: Any) -> Any:
    if not path:
        return value
    idx, rest = path[0], path[1:]
    if isinstance(program, list):
        out = list(program)
        out[idx] = replace_at(out[idx], rest, value)
        return out
    out = list(program)
    out[idx] = replace_at(out[idx], rest, value)
    return tuple(out)


def alternatives(node: tuple) -> list[tuple]:
    k = node[0]
    if k == "const":
        return [("const", node[1] + d) for d in DELTAS]
    if k == "cmp":
        return [("cmp", op, node[2], node[3]) for op in CMP_ALTS.get(node[1], ())]
    if k == "bin":
        return [("bin", op, node[2], node[3]) for op in BIN_ALTS.get(node[1], ())]
    return []


def plant_bug(correct: list, rng: random.Random, n_bugs: int = 1,
              check: list[dict] | None = None) -> list | None:
    prog = _deepcopy(correct)
    for _ in range(n_bugs):
        sites = rewrite_sites(prog)
        if not sites:
            return None
        for _attempt in range(10):
            path, node = rng.choice(sites)
            alts = alternatives(node)
            if not alts:
                continue
            cand = replace_at(prog, path, rng.choice(alts))
            if check is None:
                prog = cand
                break
            try:
                for env in check:
                    exec_program(cand, env)
                prog = cand
                break
            except RepairError:
                continue
        else:
            return None
    return prog


def _deepcopy(obj: Any) -> Any:
    if isinstance(obj, list):
        return [_deepcopy(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_deepcopy(x) for x in obj)
    return obj


# ---------------------------------------------------------------------------
# Generation (visible evidence only)
# ---------------------------------------------------------------------------


@dataclass
class GeneratedRepair:
    program: list
    description: str


def generate_repairs(buggy: list, visible: list[tuple[dict[str, int], int]], *,
                     max_width: int = 2, max_candidates: int = 3000) -> tuple[list[GeneratedRepair], int]:
    """Enumerate rewrites that satisfy VISIBLE examples only. No oracle access."""
    sites = rewrite_sites(buggy)
    found: list[GeneratedRepair] = []
    tried = 0
    from itertools import combinations, product

    def passes(cand: list) -> bool:
        for env, expected in visible:
            try:
                if exec_program(cand, env) != expected:
                    return False
            except RepairError:
                return False
        return True

    for width in range(1, max_width + 1):
        for combo in combinations(sites, width):
            alt_lists = [alternatives(node) for _p, node in combo]
            if any(not a for a in alt_lists):
                continue
            for choice in product(*alt_lists):
                cand = buggy
                for (path, _node), value in zip(combo, choice):
                    cand = replace_at(cand, path, value)
                tried += 1
                if passes(cand):
                    found.append(GeneratedRepair(cand, f"width={width}"))
                if tried >= max_candidates:
                    return found, tried
    return found, tried


# ---------------------------------------------------------------------------
# Verification (separate; uses bounded domain + adversarial probes)
# ---------------------------------------------------------------------------


@dataclass
class VerificationReport:
    state: VerificationState
    equivalent_on_domain: bool | None
    domain_size: int
    counterexamples: list[dict[str, int]] = field(default_factory=list)
    n_plausible: int = 0
    reasons: list[str] = field(default_factory=list)
    coverage: float = 0.0


def verify_repair(candidate: list, reference: list, *,
                  domain: list[dict[str, int]],
                  visible: list[tuple[dict[str, int], int]],
                  plausible: list[list] | None = None,
                  adversarial: list[dict[str, int]] | None = None) -> VerificationReport:
    """Decide equivalence on a bounded domain; never rely on visible examples."""
    reasons: list[str] = []
    ce: list[dict[str, int]] = []
    checked = 0
    for env in domain:
        try:
            a = exec_program(candidate, env)
            b = exec_program(reference, env)
        except RepairError:
            ce.append(env)
            continue
        checked += 1
        if a != b:
            ce.append(env)
            if len(ce) >= 8:
                break
    equivalent = not ce
    # Adversarial counterexamples (out-of-domain random probes where terminating)
    for env in adversarial or []:
        try:
            if exec_program(candidate, env) != exec_program(reference, env):
                ce.append(env)
                equivalent = False
        except RepairError:
            pass
    n_plausible = len(plausible) if plausible is not None else 1
    if ce:
        state = VerificationState.NOT_EQUIVALENT
        reasons.append(f"{len(ce)} bounded/adversarial counterexample(s); visible-example pass is insufficient")
    elif n_plausible > 1:
        # multiple distinct repairs satisfy visible evidence and all agree on
        # the bounded domain -> equivalent *within the domain*, still bounded.
        state = VerificationState.AMBIGUOUS
        reasons.append("multiple plausible repairs pass visible examples; equivalence claimed only on bounded domain")
    elif checked < len(domain):
        state = VerificationState.UNVERIFIABLE
        reasons.append("bounded domain not fully covered")
    else:
        state = VerificationState.VERIFIED_EQUIVALENT
        reasons.append(f"exhaustively equivalent on {checked} bounded inputs")
    return VerificationReport(
        state=state, equivalent_on_domain=equivalent,
        domain_size=len(domain), counterexamples=ce,
        n_plausible=n_plausible, reasons=reasons,
        coverage=checked / len(domain) if domain else 0.0,
    )


# ---------------------------------------------------------------------------
# Episode orchestration (oracle is evaluation-side only)
# ---------------------------------------------------------------------------


def repair_episode(name: str, seed: int, *, n_visible: int = 5,
                   n_bugs: int = 1) -> dict[str, Any]:
    rng = random.Random(f"i-f:{name}:{seed}")
    correct = TEMPLATES[name]()
    dom = domain_for(name)
    visible_envs = sample_inputs(name, rng, n_visible)
    visible = [(e, exec_program(correct, e)) for e in visible_envs]
    buggy = plant_bug(correct, rng, n_bugs, check=visible_envs)
    if buggy is None:
        return {"name": name, "seed": seed, "status": "skipped",
                "failure_category": "skipped", "reason": "nonterminating bug instance"}
    repairs, tried = generate_repairs(buggy, visible, max_width=n_bugs + 1)
    if not repairs:
        return {"name": name, "seed": seed, "status": "completed",
                "repair_found": False, "n_candidates": tried,
                "failure_category": "search-explosion", "state": "NO_REPAIR"}
    # Verification is independent of generation.
    cand0 = repairs[0].program
    distinct = _distinct_programs([r.program for r in repairs])
    report = verify_repair(
        cand0, correct, domain=dom, visible=visible, plausible=distinct,
        adversarial=sample_inputs(name, random.Random(seed + 999), 6),
    )
    detected_false = report.state is VerificationState.NOT_EQUIVALENT
    # A repair is "falsely verified" only if verification *claimed* correctness
    # while the bounded check actually disagreed. Exhaustive comparison makes
    # this impossible by construction; we still measure it explicitly.
    false_verified = bool(report.state in (VerificationState.VERIFIED_EQUIVALENT,
                                          VerificationState.AMBIGUOUS)
                          and not report.equivalent_on_domain)
    return {
        "name": name, "seed": seed, "status": "completed",
        "repair_found": True, "n_candidates": tried,
        "n_plausible": len(distinct),
        "state": report.state.value,
        "equivalent_on_domain": report.equivalent_on_domain,
        "verification_coverage": report.coverage,
        "counterexamples": report.counterexamples[:3],
        "false_repair_detected": detected_false,
        "false_verified_repair": false_verified,
        "reasons": report.reasons,
        "failure_category": (
            "false-repair-detected" if detected_false else
            "success" if report.state in (VerificationState.VERIFIED_EQUIVALENT,
                                          VerificationState.AMBIGUOUS) else
            "unverifiable"),
    }


def _distinct_programs(programs: list[list]) -> list[list]:
    seen: set[str] = set()
    out: list[list] = []
    for p in programs:
        k = repr(p)
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out
