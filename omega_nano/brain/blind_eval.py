"""Contamination-free Sealed Blind Evaluation Vault (Constitution §7.2 & §2.3).

Solvers receive ONLY `BlindProblem` + independent `verify_fn`. The reference
solution (`_SEALED`) is physically isolated and revealed ONLY in the post-hoc
evaluation report after all attempts conclude.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .verifiers import CodeVerifier, MathVerifier, VerificationResult


@dataclass(frozen=True)
class BlindProblem:
    id: str
    domain: str  # "math" | "code"
    title: str
    statement: str
    difficulty: int
    verify: Callable[[str], VerificationResult]


@dataclass(frozen=True)
class SealedReference:
    id: str
    reference_solution: str
    source_note: str


def _math_eq_checker(lhs: str, rhs: str, assignments_fn: Callable[[str], dict[str, str]],
                     positive_ints: tuple[str, ...] = ()) -> Callable[[str], VerificationResult]:
    def check(candidate: str) -> VerificationResult:
        try:
            assign = assignments_fn(candidate)
        except ValueError as exc:
            return VerificationResult("SYNTAX_ERROR", False, 0.0, str(exc))
        if positive_ints:
            import sympy as sp
            for name in positive_ints:
                try:
                    val = sp.Integer(assign[name])
                except (KeyError, TypeError, ValueError):
                    return VerificationResult("REFUTED", False, 0.0, f"{name} must be a positive integer")
                if val <= 0:
                    return VerificationResult("REFUTED", False, 0.0, f"{name}={val} must be > 0")
        return MathVerifier.verify_equation_solution(lhs, rhs, assign)
    return check


def _parse_pair(text: str, a: str = "x", b: str = "y") -> dict[str, str]:
    cleaned = text.strip().strip("()[] ")
    parts = [p.strip() for p in cleaned.split(",")]
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Expected two comma-separated values for ({a}, {b}), got {text!r}")
    return {a: parts[0], b: parts[1]}


def _parse_single(name: str) -> Callable[[str], dict[str, str]]:
    def fn(text: str) -> dict[str, str]:
        s = text.strip()
        if not s:
            raise ValueError(f"Empty answer for {name}")
        return {name: s}
    return fn


def _crt_check(candidate: str) -> VerificationResult:
    try:
        n = int(candidate.strip())
    except ValueError:
        return VerificationResult("SYNTAX_ERROR", False, 0.0, f"Expected integer n, got {candidate!r}")
    if n <= 0:
        return VerificationResult("REFUTED", False, 0.0, f"n={n} must be positive")
    if n % 3 == 2 and n % 5 == 3 and n % 7 == 2:
        return VerificationResult("PROVEN_EXACT", True, 1.0, f"n={n} satisfies all modular congruences", {"n": n})
    return VerificationResult(
        "REFUTED", False, 0.0,
        f"n={n} fails congruences: mod3={n % 3} (want 2), mod5={n % 5} (want 3), mod7={n % 7} (want 2)",
    )


def _sophie_germain_check(candidate: str) -> VerificationResult:
    import sympy as sp
    from .verifiers import safe_parse_expr
    x, y = sp.Symbol("x", real=True), sp.Symbol("y", real=True)
    try:
        expr = safe_parse_expr(candidate, {"x": x, "y": y})
    except (ValueError, sp.SympifyError, TypeError) as exc:
        return VerificationResult("SYNTAX_ERROR", False, 0.0, str(exc))
    if not isinstance(expr, sp.Mul) or len(expr.args) < 2:
        return VerificationResult("REFUTED", False, 0.0, "Expression is not factored as a product of polynomials")
    return MathVerifier.verify_identity(candidate, "x**4 + 4*y**4", var_names=("x", "y"))


def _power_sum_check(candidate: str) -> VerificationResult:
    # Given x + 1/x = 3, verify x^5 + 1/x^5 == candidate using exact root phi^2 = (3+sqrt(5))/2
    return MathVerifier.verify_equation_solution(
        "x**5 + 1/x**5 - ans", "0",
        {"x": "(3 + sqrt(5))/2", "ans": candidate},
    )


_PROBLEMS: list[BlindProblem] = [
    BlindProblem(
        id="math-pell-13", domain="math", title="Pell Equation D=13", difficulty=3,
        statement="Find positive integers (x, y) such that x^2 - 13*y^2 = 1. Format: x, y",
        verify=_math_eq_checker("x**2 - 13*y**2", "1", lambda s: _parse_pair(s, "x", "y"), ("x", "y")),
    ),
    BlindProblem(
        id="math-sophie-germain", domain="math", title="Sophie Germain Factorization", difficulty=2,
        statement="Factor x^4 + 4*y^4 into a product of two non-constant polynomials in x and y.",
        verify=_sophie_germain_check,
    ),
    BlindProblem(
        id="math-nested-radical", domain="math", title="Nested Radical Difference", difficulty=2,
        statement="Simplify sqrt(11 + 6*sqrt(2)) - sqrt(11 - 6*sqrt(2)) to a clean exact form.",
        verify=lambda cand: MathVerifier.verify_identity(cand, "sqrt(11 + 6*sqrt(2)) - sqrt(11 - 6*sqrt(2))"),
    ),
    BlindProblem(
        id="math-power-sum-5", domain="math", title="Reciprocal Power Sum", difficulty=2,
        statement="If x + 1/x = 3, compute the exact value of x^5 + 1/x^5.",
        verify=_power_sum_check,
    ),
    BlindProblem(
        id="math-crt-congruence", domain="math", title="Chinese Remainder System", difficulty=1,
        statement="Find a positive integer n such that n mod 3 = 2, n mod 5 = 3, and n mod 7 = 2.",
        verify=_crt_check,
    ),
    BlindProblem(
        id="code-merge-intervals", domain="code", title="Merge Overlapping Intervals", difficulty=1,
        statement="Write Python function merge_intervals(intervals: list[tuple[int,int]]) -> list[tuple[int,int]] merging overlapping or touching intervals sorted by start.",
        verify=lambda code: CodeVerifier.verify_python(
            code,
            "assert merge_intervals([]) == []\n"
            "assert merge_intervals([(1, 3), (2, 6), (8, 10), (15, 18)]) == [(1, 6), (8, 10), (15, 18)]\n"
            "assert merge_intervals([(5, 7), (1, 5)]) == [(1, 7)]\n"
            "assert merge_intervals([(1, 10), (2, 3), (4, 8)]) == [(1, 10)]\n",
        ),
    ),
    BlindProblem(
        id="code-lru-cache", domain="code", title="O(1) LRU Cache", difficulty=2,
        statement="Implement Python class LRUCache(capacity: int) with get(key) -> int (-1 if absent) and put(key, value).",
        verify=lambda code: CodeVerifier.verify_python(
            code,
            "c = LRUCache(2)\n"
            "c.put(1, 1); c.put(2, 2)\n"
            "assert c.get(1) == 1\n"
            "c.put(3, 3)\n"
            "assert c.get(2) == -1\n"
            "c.put(4, 4)\n"
            "assert c.get(1) == -1 and c.get(3) == 3 and c.get(4) == 4\n"
            "c1 = LRUCache(1); c1.put(1, 10); c1.put(1, 20); assert c1.get(1) == 20\n",
        ),
    ),
    BlindProblem(
        id="code-topo-sort", domain="code", title="Topological Sort with Cycle Detection", difficulty=2,
        statement="Write Python function topo_sort(n: int, edges: list[tuple[int, int]]) -> list[int] | None returning a valid topological order for vertices 0..n-1, or None if a cycle exists.",
        verify=lambda code: CodeVerifier.verify_python(
            code,
            "res = topo_sort(4, [(0, 1), (0, 2), (1, 3), (2, 3)])\n"
            "assert isinstance(res, list) and sorted(res) == [0, 1, 2, 3]\n"
            "pos = {v: i for i, v in enumerate(res)}\n"
            "assert pos[0] < pos[1] and pos[0] < pos[2] and pos[1] < pos[3] and pos[2] < pos[3]\n"
            "assert topo_sort(2, [(0, 1), (1, 0)]) is None\n"
            "assert topo_sort(1, [(0, 0)]) is None\n"
            "assert sorted(topo_sort(3, [])) == [0, 1, 2]\n",
        ),
    ),
    BlindProblem(
        id="code-utf8-arabic-chunk", domain="code", title="UTF-8 Safe Word Chunker", difficulty=2,
        statement="Write Python function chunk_utf8_words(text: str, max_bytes: int) -> list[str] splitting whitespace-delimited words into chunks each <= max_bytes in UTF-8 without splitting words (unless a single word > max_bytes, raise ValueError).",
        verify=lambda code: CodeVerifier.verify_python(
            code,
            "chunks = chunk_utf8_words('مرحبا بالعالم هذا اختبار', 20)\n"
            "assert all(len(c.encode('utf-8')) <= 20 for c in chunks)\n"
            "assert ' '.join(chunks) == 'مرحبا بالعالم هذا اختبار'\n"
            "assert chunk_utf8_words('   ', 10) == []\n",
        ),
    ),
    BlindProblem(
        id="code-rpn-eval", domain="code", title="Reverse Polish Notation Evaluator", difficulty=1,
        statement="Write Python function eval_rpn(tokens: list[str]) -> int supporting +, -, *, / (truncating division toward zero).",
        verify=lambda code: CodeVerifier.verify_python(
            code,
            "assert eval_rpn(['2', '1', '+', '3', '*']) == 9\n"
            "assert eval_rpn(['4', '13', '5', '/', '+']) == 6\n"
            "assert eval_rpn(['10', '6', '9', '3', '+', '-11', '*', '/', '*', '17', '+', '5', '+']) == 22\n"
            "assert eval_rpn(['-7', '2', '/']) == -3\n",
        ),
    ),
]

_SEALED: dict[str, SealedReference] = {
    "math-pell-13": SealedReference("math-pell-13", "649, 180", "Minimal Pell solution via continued fraction sqrt(13)=[3;1,1,1,1,6]"),
    "math-sophie-germain": SealedReference("math-sophie-germain", "(x**2 - 2*x*y + 2*y**2)*(x**2 + 2*x*y + 2*y**2)", "Sophie Germain identity"),
    "math-nested-radical": SealedReference("math-nested-radical", "2*sqrt(2)", "(3+sqrt(2)) - (3-sqrt(2))"),
    "math-power-sum-5": SealedReference("math-power-sum-5", "123", "Lucas/Newton power-sum recurrence S_k = 3*S_{k-1} - S_{k-2}"),
    "math-crt-congruence": SealedReference("math-crt-congruence", "23", "Sunzi / Chinese Remainder Theorem"),
    "code-merge-intervals": SealedReference("code-merge-intervals", "sort + greedy merge", "Standard O(n log n) sweep"),
    "code-lru-cache": SealedReference("code-lru-cache", "OrderedDict", "O(1) get/put cache"),
    "code-topo-sort": SealedReference("code-topo-sort", "Kahn BFS", "O(V+E) cycle-aware topological sort"),
    "code-utf8-arabic-chunk": SealedReference("code-utf8-arabic-chunk", "UTF-8 byte length accumulator", "Multibyte Arabic UTF-8 safety"),
    "code-rpn-eval": SealedReference("code-rpn-eval", "stack + int(a/b)", "Truncate toward zero"),
}


class BlindVault:
    """Runs contamination-free blind evaluation against sealed benchmarks."""

    def __init__(self, problems: list[BlindProblem] | None = None,
                 sealed: dict[str, SealedReference] | None = None):
        self._problems = list(problems or _PROBLEMS)
        self._sealed = dict(sealed or _SEALED)

    def problems(self, domain: str | None = None) -> list[BlindProblem]:
        return [p for p in self._problems if domain is None or p.domain == domain]

    def evaluate(self, solve_fn: Callable[[BlindProblem], dict[str, Any]],
                 domain: str | None = None) -> dict[str, Any]:
        """Evaluate a solver blindly, revealing sealed references ONLY in the post-hoc summary."""
        results = []
        passed = 0
        for p in self.problems(domain):
            outcome = solve_fn(p)
            ok = bool(outcome.get("passed"))
            if ok:
                passed += 1
            ref = self._sealed.get(p.id)
            results.append({
                "id": p.id,
                "domain": p.domain,
                "title": p.title,
                "passed": ok,
                "status": outcome.get("status", "UNVERIFIED"),
                "attempts": outcome.get("attempts", 0),
                "source": outcome.get("source", "solver"),
                "candidate_preview": str(outcome.get("solution", ""))[:200],
                "post_hoc_reference": ref.reference_solution if ref else None,
                "post_hoc_note": ref.source_note if ref else None,
            })
        n = len(results)
        return {
            "total": n,
            "passed": passed,
            "pass_rate": round(passed / max(1, n), 4),
            "results": results,
        }
