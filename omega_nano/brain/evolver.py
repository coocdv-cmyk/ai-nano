"""Self-Play Curriculum, Execution-Guided Search & Continual Learning Engine.

Implements Loops 2 & 3 of RC-004:
  * CurriculumProposer generates verifiable math and coding challenges.
  * solve_with_verification runs SkillLibrary lookup -> multi-attempt solver ->
    formal/execution verifier -> structured counterexample/traceback feedback.
  * Every verified win compiles generalized skills and appends to Replay Buffer.
"""

from __future__ import annotations

import ast
import math
import random
import re
from dataclasses import dataclass
from typing import Any, Callable

import sympy as sp

from ..semantic import DriftStatus, SemanticDriftMonitor, verify_behavioral_candidates
from .blind_eval import BlindProblem
from .skills import SkillLibrary
from .synthesizer import CandidateHypothesis, ExecutionGuidedSynthesizer, NeuralPolicyRanker
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

    def generate(self, n: int = 8, level: int = 1, diverse_code: bool = False) -> list[BlindProblem]:
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
                out.append(self._make_code_task(i, level, diverse=diverse_code))
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

    def _make_code_task(self, idx: int, level: int, diverse: bool = False) -> BlindProblem:
        slot = ((idx // 4) % 6) if diverse else 0
        if slot == 1:
            a = self.rng.randint(-15, -5)
            b = self.rng.randint(2, 5)
            expected_q = int(a / b)
            stmt = (
                "Write Python function eval_postfix(tokens: list[str]) -> int "
                "evaluating Reverse Polish Notation (+, -, *, /) where division truncates toward zero."
            )
            tests = (
                f"assert eval_postfix(['{a}', '{b}', '/']) == {expected_q}\n"
                "assert eval_postfix(['2', '1', '+', '3', '*']) == 9\n"
                "assert eval_postfix(['10', '6', '9', '3', '+', '-11', '*', '/', '*', '17', '+', '5', '+']) == 22\n"
            )
            return BlindProblem(
                id=f"self-code-rpn-{idx}", domain="code", title="Postfix RPN Truncated Division",
                difficulty=level, statement=stmt,
                verify=lambda code, T=tests: CodeVerifier.verify_python(code, T),
            )
        if slot == 2:
            x = self.rng.randint(1, 5)
            y = x + self.rng.randint(2, 4)
            z = y + self.rng.randint(2, 4)
            stmt = (
                "Write Python function merge_ranges(intervals: list[list[int]]) -> list[list[int]] "
                "merging all overlapping or touching intervals sorted ascending."
            )
            tests = (
                f"assert merge_ranges([[{x}, {y}], [{y}, {z}], [{z + 2}, {z + 5}]]) == [[{x}, {z}], [{z + 2}, {z + 5}]]\n"
                "assert merge_ranges([]) == []\n"
            )
            return BlindProblem(
                id=f"self-code-merge-{idx}", domain="code", title="Merge Touching Intervals",
                difficulty=level, statement=stmt,
                verify=lambda code, T=tests: CodeVerifier.verify_python(code, T),
            )
        if slot == 3:
            stmt = (
                "Write Python function longest_unique_len(s: str) -> int "
                "returning the length of the longest substring without repeating characters."
            )
            tests = (
                "assert longest_unique_len('abcabcbb') == 3\n"
                "assert longest_unique_len('bbbbb') == 1\n"
                "assert longest_unique_len('pwwkew') == 3\n"
                "assert longest_unique_len('') == 0\n"
            )
            return BlindProblem(
                id=f"self-code-uniq-{idx}", domain="code", title="Longest Unique Substring",
                difficulty=level, statement=stmt,
                verify=lambda code, T=tests: CodeVerifier.verify_python(code, T),
            )
        if slot == 4:
            stmt = (
                "Write Python function is_valid_brackets(s: str) -> bool "
                "checking whether brackets (), [], {} are balanced and properly nested."
            )
            tests = (
                "assert is_valid_brackets('()[]{}') is True\n"
                "assert is_valid_brackets('([{}])') is True\n"
                "assert is_valid_brackets('([)]') is False\n"
                "assert is_valid_brackets('(') is False\n"
            )
            return BlindProblem(
                id=f"self-code-brk-{idx}", domain="code", title="Balanced Brackets",
                difficulty=level, statement=stmt,
                verify=lambda code, T=tests: CodeVerifier.verify_python(code, T),
            )
        if slot == 5:
            t = self.rng.choice([2, 4, 5])
            stmt = (
                "Write Python function lower_bound_idx(nums: list[int], target: int) -> int "
                "returning the leftmost binary search insertion index in sorted array nums."
            )
            tests = (
                "assert lower_bound_idx([1, 2, 2, 4, 6], 2) == 1\n"
                "assert lower_bound_idx([1, 2, 2, 4, 6], 3) == 3\n"
                f"assert lower_bound_idx([1, 2, 2, 4, 6], {t}) == {[1, 2, 2, 4, 6].index(t) if t in [1,2,2,4,6] else 4}\n"
                "assert lower_bound_idx([], 5) == 0\n"
            )
            return BlindProblem(
                id=f"self-code-lb-{idx}", domain="code", title="Binary Search Lower Bound",
                difficulty=level, statement=stmt,
                verify=lambda code, T=tests: CodeVerifier.verify_python(code, T),
            )

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


class NeuroSymbolicSolver(SymbolicTacticSolver):
    """Unified neuro-symbolic solver combining exact math tactics + ExecutionGuidedSynthesizer.

    Keeps iterative math tactics (so Pell uses small bound -> CF counterexample refinement)
    while using learned policy ranking + CEGIS counterexample repair for arbitrary code tasks.
    """

    def __init__(
        self,
        policy: NeuralPolicyRanker | None = None,
        extra_solver: Callable[[BlindProblem, list[AttemptRecord]], str | None] | None = None,
        neural_proposer: Callable[[BlindProblem, list[AttemptRecord]], str | None] | None = None,
    ):
        super().__init__(extra_solver=extra_solver)
        self.policy = policy or NeuralPolicyRanker()
        self.synthesizer = ExecutionGuidedSynthesizer(policy=self.policy)
        self.neural_proposer = neural_proposer

    def bind_policy(self, policy: NeuralPolicyRanker) -> None:
        self.policy = policy
        self.synthesizer.policy = policy

    def propose_with_meta(
        self, problem: BlindProblem, history: list[AttemptRecord]
    ) -> tuple[str | None, CandidateHypothesis | None]:
        if self.extra_solver is not None:
            cand = self.extra_solver(problem, history)
            if cand:
                return cand, None
        if problem.domain == "math":
            return self._propose_math(problem.statement, history), None
        code, hyp = self.synthesizer.propose(problem.statement, history)
        if code is not None:
            return code, hyp
        if self.neural_proposer is not None:
            cand = self.neural_proposer(problem, history)
            if cand:
                return cand, None
        return None, None

    def _propose_code(self, stmt: str, history: list[AttemptRecord]) -> str | None:
        code, _ = self.synthesizer.propose(stmt, history)
        return code


def _extract_primary_symbol(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return node.name
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
    if problem.domain == "code":
        sym = _extract_primary_symbol(solution)
        if sym and len(sym) >= 5 and sym.lower() not in {"solve", "solution", "helper", "apply"}:
            skill_name = f"code_{sym.lower()}"
            skills.register_skill(
                skill_name, "code",
                rf"\b{re.escape(sym)}\b",
                {"kind": "code_template", "code": solution}, proof_status,
            )
            return skill_name
    return None


class SelfPlayEngine:
    """Coordinates L3-verified skill recall, L1 drift monitoring, online policy learning, and replay."""

    def __init__(self, skills: SkillLibrary, drift_monitor: SemanticDriftMonitor | None = None):
        self.skills = skills
        self.proposer = CurriculumProposer()
        self.drift_monitor = drift_monitor or SemanticDriftMonitor(window=12, baseline_rate=0.85, threshold=1.5)
        self.policy = NeuralPolicyRanker(weights=self.skills.load_policy())
        self.last_drift_status: DriftStatus | None = None

    def solve(self, problem: BlindProblem, solver: Callable[[BlindProblem, list[AttemptRecord]], str | None],
              max_attempts: int = 5, cycle: int = 0) -> dict[str, Any]:
        if hasattr(solver, "bind_policy"):
            solver.bind_policy(self.policy)

        # Tier 0+: evaluate all pattern-matched skills via EXP-L L3 behavioral verification guard.
        skill_l3_rejected = False
        hits = self.skills.match_candidates(problem.statement, domain=problem.domain, limit=4)
        if hits:
            vr_cache: dict[str, VerificationResult] = {}
            skill_outcomes: list[tuple[str, bool]] = []
            gen_pairs: list[tuple[str, Callable[[], str]]] = []
            for skill_item, groups in hits:
                cand = apply_skill_strategy(skill_item["strategy"], groups)
                if cand is None:
                    continue
                sname = skill_item["name"]
                gen_pairs.append((sname, lambda s=cand: s))

            def _verify_cached(cand_code: str) -> bool:
                if cand_code not in vr_cache:
                    vr_cache[cand_code] = problem.verify(cand_code)
                return vr_cache[cand_code].passed

            chosen_name, chosen_sol, l3_status = verify_behavioral_candidates(gen_pairs, _verify_cached)
            for skill_item, groups in hits:
                cand = apply_skill_strategy(skill_item["strategy"], groups)
                if cand is None:
                    continue
                sname = skill_item["name"]
                passed_i = bool(cand in vr_cache and vr_cache[cand].passed)
                self.skills.record_outcome(sname, passed_i, count_use=True)
                skill_outcomes.append((sname, passed_i))

            drift = self.drift_monitor.observe(chosen_name is not None)
            self.last_drift_status = drift
            if chosen_name is not None and chosen_sol is not None:
                vr = vr_cache[chosen_sol]
                self.skills.add_replay(
                    problem.id, problem.domain, problem.statement, chosen_sol, 1, vr.status, cycle,
                    {"via_skill": chosen_name, "l3_status": l3_status, "drifted": drift.drifted},
                )
                return {
                    "passed": True, "status": vr.status, "solution": chosen_sol,
                    "attempts": 1, "source": f"skill:{chosen_name}",
                    "l3_status": l3_status, "l3_rejected_skill": False,
                    "drift": drift.drifted, "drift_reason": drift.reason,
                    "evidence": vr.evidence,
                }
            skill_l3_rejected = True

        history: list[AttemptRecord] = []
        refuted_hyps: list[CandidateHypothesis] = []
        for idx in range(1, max_attempts + 1):
            if hasattr(solver, "propose_with_meta"):
                cand, hyp = solver.propose_with_meta(problem, history)
            else:
                cand, hyp = solver(problem, history), None

            if cand is None:
                history.append(AttemptRecord(idx, "", VerificationResult("UNVERIFIED", False, 0.0, "No proposal")))
                continue
            vr = problem.verify(cand)
            history.append(AttemptRecord(idx, cand, vr))
            if vr.passed:
                if hyp is not None:
                    self.policy.update(problem.statement, hyp, refuted_hyps, lr=0.35)
                    self.skills.save_policy(self.policy.export_weights())
                compiled = _compile_generalized_skill(self.skills, problem, cand, vr.status)
                self.skills.add_replay(
                    problem.id, problem.domain, problem.statement, cand, idx, vr.status, cycle,
                    {"compiled_skill": compiled, "l3_rejected_skill": skill_l3_rejected},
                )
                return {
                    "passed": True, "status": vr.status, "solution": cand,
                    "attempts": idx, "source": "search", "compiled_skill": compiled,
                    "l3_rejected_skill": skill_l3_rejected,
                    "drift": self.last_drift_status.drifted if self.last_drift_status else False,
                    "evidence": vr.evidence,
                }
            if hyp is not None:
                refuted_hyps.append(hyp)

        last = history[-1].verification if history else VerificationResult("UNVERIFIED", False, 0.0, "No attempts")
        return {
            "passed": False, "status": last.status, "solution": history[-1].candidate if history else "",
            "attempts": len(history), "source": "search",
            "l3_rejected_skill": skill_l3_rejected,
            "drift": self.last_drift_status.drifted if self.last_drift_status else False,
            "feedback": last.feedback,
        }

    def run_cycle(self, cycle: int, solver: Callable[[BlindProblem, list[AttemptRecord]], str | None],
                  n_tasks: int = 8, level: int = 2, max_attempts: int = 4) -> dict[str, Any]:
        tasks = self.proposer.generate(
            n=n_tasks, level=level, diverse_code=isinstance(solver, NeuroSymbolicSolver)
        )
        solved = 0
        total_attempts = 0
        compiled: list[str] = []
        drift_events = 0
        for p in tasks:
            res = self.solve(p, solver, max_attempts=max_attempts, cycle=cycle)
            total_attempts += res["attempts"]
            if res.get("drift"):
                drift_events += 1
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
            "drift_events": drift_events,
            "library_stats": self.skills.stats(),
        }
