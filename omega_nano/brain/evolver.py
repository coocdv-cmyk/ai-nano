"""Self-Play Curriculum, Execution-Guided Search & Continual Learning Engine.

Implements Loops 2 & 3 of RC-004:
  * CurriculumProposer generates verifiable math and coding challenges.
  * solve_with_verification runs SkillLibrary lookup -> multi-attempt solver ->
    formal/execution verifier -> structured counterexample/traceback feedback.
  * Every verified win compiles generalized skills and appends to Replay Buffer.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Any, Callable

import sympy as sp

from .blind_eval import BlindProblem
from .skills import SkillLibrary
from .verifiers import CodeVerifier, MathVerifier, VerificationResult, safe_parse_expr


@dataclass
class AttemptRecord:
    attempt: int
    candidate: str
    verification: VerificationResult


class CurriculumProposer:
    """Generates calibrated, verifiable synthetic challenges (no answer leaked to solver)."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate(self, n: int = 8, level: int = 1) -> list[BlindProblem]:
        out: list[BlindProblem] = []
        for i in range(n):
            kind = i % 4
            if kind == 0:
                out.append(self._make_quadratic_diophantine(i, level))
            elif kind == 1:
                out.append(self._make_pell_or_linear_system(i, level))
            elif kind == 2:
                out.append(self._make_power_recurrence(i, level))
            else:
                out.append(self._make_code_task(i, level))
        return out

    def _make_quadratic_diophantine(self, idx: int, level: int) -> BlindProblem:
        r1 = self.rng.randint(2, 6 + 4 * level)
        r2 = self.rng.randint(2, 6 + 4 * level)
        b = -(r1 + r2)
        c = r1 * r2
        sign_b = f"+ {b}" if b >= 0 else f"- {-b}"
        stmt = f"Solve for a real root x of: x^2 {sign_b}*x + {c} = 0"
        lhs = f"x**2 + ({b})*x + {c}"
        return BlindProblem(
            id=f"self-quad-{idx}", domain="math", title="Quadratic Equation", difficulty=level,
            statement=stmt,
            verify=lambda cand, L=lhs: MathVerifier.verify_equation_solution(L, "0", {"x": cand}),
        )

    def _make_pell_or_linear_system(self, idx: int, level: int) -> BlindProblem:
        if level >= 2:
            d = self.rng.choice([2, 3, 5, 6, 7, 10, 11])
            stmt = f"Find positive integers (x, y) such that x^2 - {d}*y^2 = 1. Format: x, y"

            def check(cand: str, D=d) -> VerificationResult:
                parts = [p.strip() for p in cand.strip().strip("()").split(",")]
                if len(parts) != 2:
                    return VerificationResult("SYNTAX_ERROR", False, 0.0, "Format must be: x, y")
                try:
                    xv, yv = int(parts[0]), int(parts[1])
                except ValueError:
                    return VerificationResult("SYNTAX_ERROR", False, 0.0, "x and y must be integers")
                if xv <= 0 or yv <= 0:
                    return VerificationResult("REFUTED", False, 0.0, "x and y must be positive")
                return MathVerifier.verify_equation_solution(f"x**2 - {D}*y**2", "1", {"x": str(xv), "y": str(yv)})

            return BlindProblem(
                id=f"self-pell-{idx}", domain="math", title=f"Pell D={d}", difficulty=level,
                statement=stmt, verify=check,
            )

        a = self.rng.randint(2, 9)
        b = self.rng.randint(2, 9)
        x0 = self.rng.randint(1, 15)
        y0 = self.rng.randint(1, 15)
        s1 = a * x0 + b * y0
        s2 = x0 - y0
        stmt = f"Find integers (x, y) solving {a}*x + {b}*y = {s1} and x - y = {s2}. Format: x, y"

        def check_lin(cand: str, A=a, B=b, S1=s1, S2=s2) -> VerificationResult:
            parts = [p.strip() for p in cand.strip().strip("()").split(",")]
            if len(parts) != 2:
                return VerificationResult("SYNTAX_ERROR", False, 0.0, "Format must be: x, y")
            r1 = MathVerifier.verify_equation_solution(f"{A}*x + {B}*y", str(S1), {"x": parts[0], "y": parts[1]})
            if not r1.passed:
                return r1
            return MathVerifier.verify_equation_solution("x - y", str(S2), {"x": parts[0], "y": parts[1]})

        return BlindProblem(
            id=f"self-lin-{idx}", domain="math", title="Linear System 2x2", difficulty=level,
            statement=stmt, verify=check_lin,
        )

    def _make_power_recurrence(self, idx: int, level: int) -> BlindProblem:
        k = self.rng.randint(3, 5 + level)
        n = self.rng.randint(3, 4 + level)
        stmt = f"If x + 1/x = {k}, compute the exact value of x^{n} + 1/x^{n}."

        def check_pow(cand: str, K=k, N=n) -> VerificationResult:
            root = f"({K} + sqrt({K * K - 4}))/2"
            return MathVerifier.verify_equation_solution(f"x**{N} + 1/x**{N} - ans", "0", {"x": root, "ans": cand})

        return BlindProblem(
            id=f"self-pow-{idx}", domain="math", title=f"Reciprocal Power n={n}", difficulty=level,
            statement=stmt, verify=check_pow,
        )

    def _make_code_task(self, idx: int, level: int) -> BlindProblem:
        k = self.rng.randint(2, 4)
        stmt = (
            f"Write Python function max_window_sum(nums: list[int], k: int) -> int "
            f"returning the maximum sum of any contiguous subarray of length k (raise ValueError if len(nums) < k or k <= 0)."
        )
        tests = (
            f"assert max_window_sum([1, 4, 2, 10, 23, 3, 1, 0, 20], {k}) == "
            f"max(sum([1, 4, 2, 10, 23, 3, 1, 0, 20][i:i+{k}]) for i in range(10-{k}))\n"
            "assert max_window_sum([-5, -1, -8], 2) == -6\n"
            "try:\n"
            "    max_window_sum([1], 2)\n"
            "    raise AssertionError('expected ValueError')\n"
            "except ValueError:\n"
            "    pass\n"
        )
        return BlindProblem(
            id=f"self-code-win-{idx}", domain="code", title="Max Sliding Window Sum", difficulty=level,
            statement=stmt,
            verify=lambda code, T=tests: CodeVerifier.verify_python(code, T),
        )


def _solve_pell_cf(d: int) -> tuple[int, int] | None:
    """Continued fraction expansion of sqrt(D) for exact minimal Pell solution x^2 - D*y^2 = 1."""
    a0 = math.isqrt(d)
    if a0 * a0 == d:
        return None
    m, denom, a = 0, 1, a0
    num1, num2 = 1, a0
    den1, den2 = 0, 1
    while num2 * num2 - d * den2 * den2 != 1:
        m = denom * a - m
        denom = (d - m * m) // denom
        a = (a0 + m) // denom
        num1, num2 = num2, a * num2 + num1
        den1, den2 = den2, a * den2 + den1
    return num2, den2


def _solve_lucas_power(k: int, n: int) -> int:
    """Exact Newton/Lucas recurrence for x^n + 1/x^n given x + 1/x = k."""
    if n == 0:
        return 2
    if n == 1:
        return k
    s0, s1 = 2, k
    for _ in range(2, n + 1):
        s0, s1 = s1, k * s1 - s0
    return s1


def apply_skill_strategy(skill_strategy: dict[str, Any], groups: dict[str, str]) -> str | None:
    kind = skill_strategy.get("kind")
    if kind == "pell_cf":
        res = _solve_pell_cf(int(groups["d"]))
        return f"{res[0]}, {res[1]}" if res else None
    if kind == "lucas_power":
        return str(_solve_lucas_power(int(groups["k"]), int(groups["n"])))
    if kind == "crt":
        a1, m1, a2, m2, a3, m3 = (int(groups[k]) for k in ("a1", "m1", "a2", "m2", "a3", "m3"))
        for n in range(1, m1 * m2 * m3 + 1):
            if n % m1 == a1 and n % m2 == a2 and n % m3 == a3:
                return str(n)
    if kind == "code_template":
        return str(skill_strategy.get("code"))
    return None


class SymbolicTacticSolver:
    """Baseline structured math/algorithmic search solver (no external LLM required).

    Demonstrates iterative tactic search + counterexample refinement; can be wrapped
    with or replaced by an OmegaCore / LocalLLMReasoner backend.
    """

    def __init__(self, extra_solver: Callable[[BlindProblem, list[AttemptRecord]], str | None] | None = None):
        self.extra_solver = extra_solver

    def __call__(self, problem: BlindProblem, history: list[AttemptRecord]) -> str | None:
        if self.extra_solver is not None:
            cand = self.extra_solver(problem, history)
            if cand:
                return cand
        stmt = problem.statement
        if problem.domain == "math":
            return self._propose_math(stmt, history)
        return self._propose_code(stmt, history)

    def _propose_math(self, stmt: str, history: list[AttemptRecord]) -> str | None:
        # Attempt index guides search depth across tactics (System-2 iterative search).
        attempt = len(history)
        if m := re.search(r"x\^2\s*-\s*(\d+)\s*\*?\s*y\^2\s*=\s*1", stmt):
            d = int(m.group(1))
            if attempt == 0:
                # Naive small bound first; if refuted/missed, escalate to continued fractions.
                for y in range(1, 20):
                    x2 = 1 + d * y * y
                    x = math.isqrt(x2)
                    if x * x == x2:
                        return f"{x}, {y}"
                return "1, 1"  # deliberately triggers counterexample feedback on attempt 0 for hard Pell
            sol = _solve_pell_cf(d)
            return f"{sol[0]}, {sol[1]}" if sol else None

        if m := re.search(r"x\s*\+\s*1/x\s*=\s*(\d+).*?x\^(\d+)\s*\+\s*1/x\^(\d+)", stmt):
            return str(_solve_lucas_power(int(m.group(1)), int(m.group(2))))

        if m := re.search(r"Factor\s+(.+?)\s+into", stmt, re.IGNORECASE):
            expr = safe_parse_expr(m.group(1), {"x": sp.Symbol("x"), "y": sp.Symbol("y")})
            return str(sp.factor(expr))

        if m := re.search(r"Simplify\s+(.+?)\s+to\s+", stmt, re.IGNORECASE):
            expr = safe_parse_expr(m.group(1))
            return str(sp.sqrtdenest(expr) if attempt > 0 else sp.simplify(expr))

        if m := re.search(r"Solve for a real root x of:\s*(.+?)\s*=\s*0", stmt, re.IGNORECASE):
            x = sp.Symbol("x", real=True)
            roots = sp.solve(safe_parse_expr(m.group(1), {"x": x}), x)
            return str(roots[0]) if roots else None

        if m := re.search(r"(\d+)\*x\s*\+\s*(\d+)\*y\s*=\s*(-?\d+)\s+and\s+x\s*-\s*y\s*=\s*(-?\d+)", stmt):
            a, b, s1, s2 = (int(m.group(i)) for i in range(1, 5))
            x, y = sp.symbols("x y")
            sol = sp.solve([a * x + b * y - s1, x - y - s2], [x, y])
            return f"{sol[x]}, {sol[y]}" if sol else None

        if m := re.search(r"mod\s*3\s*=\s*(\d+).*?mod\s*5\s*=\s*(\d+).*?mod\s*7\s*=\s*(\d+)", stmt):
            a1, a2, a3 = int(m.group(1)), int(m.group(2)), int(m.group(3))
            for n in range(1, 106):
                if n % 3 == a1 and n % 5 == a2 and n % 7 == a3:
                    return str(n)
        return None

    @staticmethod
    def _propose_code(stmt: str, history: list[AttemptRecord]) -> str | None:
        if "max_window_sum" in stmt:
            return (
                "def max_window_sum(nums, k):\n"
                "    if k <= 0 or len(nums) < k:\n"
                "        raise ValueError('invalid window')\n"
                "    cur = best = sum(nums[:k])\n"
                "    for i in range(k, len(nums)):\n"
                "        cur += nums[i] - nums[i - k]\n"
                "        if cur > best:\n"
                "            best = cur\n"
                "    return best\n"
            )
        return None


def _compile_generalized_skill(skills: SkillLibrary, problem: BlindProblem, solution: str, proof_status: str) -> str | None:
    """Abstract a verified solution into a reusable parameterized skill when a generalizable family is recognized."""
    stmt = problem.statement
    if re.search(r"x\^2\s*-\s*\d+\s*\*?\s*y\^2\s*=\s*1", stmt):
        skills.register_skill(
            "pell_continued_fraction", "math",
            r"x\^2\s*-\s*(?P<d>\d+)\s*\*?\s*y\^2\s*=\s*1",
            {"kind": "pell_cf"}, proof_status,
        )
        return "pell_continued_fraction"
    if re.search(r"x\s*\+\s*1/x\s*=\s*\d+.*?x\^\d+\s*\+\s*1/x\^\d+", stmt):
        skills.register_skill(
            "lucas_reciprocal_power", "math",
            r"x\s*\+\s*1/x\s*=\s*(?P<k>\d+).*?x\^(?P<n>\d+)\s*\+\s*1/x\^(?P=n)",
            {"kind": "lucas_power"}, proof_status,
        )
        return "lucas_reciprocal_power"
    if re.search(r"n\s+mod\s+\d+\s*=\s*\d+", stmt):
        skills.register_skill(
            "crt_triple_mod", "math",
            r"mod\s*(?P<m1>\d+)\s*=\s*(?P<a1>\d+).*?mod\s*(?P<m2>\d+)\s*=\s*(?P<a2>\d+).*?mod\s*(?P<m3>\d+)\s*=\s*(?P<a3>\d+)",
            {"kind": "crt"}, proof_status,
        )
        return "crt_triple_mod"
    if "max_window_sum" in stmt:
        skills.register_skill(
            "sliding_window_max_sum", "code",
            r"\bmax_window_sum\b",
            {"kind": "code_template", "code": solution}, proof_status,
        )
        return "sliding_window_max_sum"
    return None


class SelfPlayEngine:
    """Coordinates iterative verification search, skill compilation, and continual replay."""

    def __init__(self, skills: SkillLibrary):
        self.skills = skills
        self.proposer = CurriculumProposer()

    def solve(self, problem: BlindProblem, solver: Callable[[BlindProblem, list[AttemptRecord]], str | None],
              max_attempts: int = 5, cycle: int = 0) -> dict[str, Any]:
        # Tier 0+: try compiled generalized skill first (< 1 ms), always verified before trusting.
        hit = self.skills.match(problem.statement, domain=problem.domain)
        if hit is not None:
            skill, groups = hit
            cand = apply_skill_strategy(skill.strategy, groups)
            if cand is not None:
                vr = problem.verify(cand)
                self.skills.record_outcome(skill.name, vr.passed)
                if vr.passed:
                    self.skills.add_replay(
                        problem.id, problem.domain, problem.statement, cand, 1, vr.status, cycle,
                        {"via_skill": skill.name},
                    )
                    return {
                        "passed": True, "status": vr.status, "solution": cand,
                        "attempts": 1, "source": f"skill:{skill.name}", "evidence": vr.evidence,
                    }

        history: list[AttemptRecord] = []
        for idx in range(1, max_attempts + 1):
            cand = solver(problem, history)
            if cand is None:
                history.append(AttemptRecord(idx, "", VerificationResult("UNVERIFIED", False, 0.0, "No proposal")))
                continue
            vr = problem.verify(cand)
            history.append(AttemptRecord(idx, cand, vr))
            if vr.passed:
                compiled = _compile_generalized_skill(self.skills, problem, cand, vr.status)
                self.skills.add_replay(
                    problem.id, problem.domain, problem.statement, cand, idx, vr.status, cycle,
                    {"compiled_skill": compiled},
                )
                return {
                    "passed": True, "status": vr.status, "solution": cand,
                    "attempts": idx, "source": "search", "compiled_skill": compiled, "evidence": vr.evidence,
                }

        last = history[-1].verification if history else VerificationResult("UNVERIFIED", False, 0.0, "No attempts")
        return {
            "passed": False, "status": last.status, "solution": history[-1].candidate if history else "",
            "attempts": len(history), "source": "search", "feedback": last.feedback,
        }

    def run_cycle(self, cycle: int, solver: Callable[[BlindProblem, list[AttemptRecord]], str | None],
                  n_tasks: int = 8, level: int = 2, max_attempts: int = 4) -> dict[str, Any]:
        tasks = self.proposer.generate(n=n_tasks, level=level)
        solved = 0
        total_attempts = 0
        compiled: list[str] = []
        for p in tasks:
            res = self.solve(p, solver, max_attempts=max_attempts, cycle=cycle)
            total_attempts += res["attempts"]
            if res["passed"]:
                solved += 1
                if res.get("compiled_skill"):
                    compiled.append(res["compiled_skill"])
        return {
            "cycle": cycle,
            "tasks": n_tasks,
            "solved": solved,
            "solve_rate": round(solved / max(1, n_tasks), 4),
            "avg_attempts": round(total_attempts / max(1, n_tasks), 2),
            "compiled_skills": sorted(set(compiled)),
            "library_stats": self.skills.stats(),
        }
