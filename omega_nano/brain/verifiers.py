"""Zero-hallucination formal math and isolated Python code verifiers."""

from __future__ import annotations

import ast
import os
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mpmath as mp
import sympy as sp

MAX_EXPR_CHARS = 2000
MAX_OUT_CHARS = 16000

_SYMPY_FUNCS: dict[str, Any] = {
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "Abs": sp.Abs,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "factorial": sp.factorial,
    "binomial": sp.binomial,
    "gcd": sp.gcd,
    "lcm": sp.lcm,
    "floor": sp.floor,
    "ceiling": sp.ceiling,
    "Rational": sp.Rational,
    "Integer": sp.Integer,
    "Mod": sp.Mod,
    "Sum": sp.Sum,
    "Product": sp.Product,
    "pi": sp.pi,
    "E": sp.E,
    "e": sp.E,
    "I": sp.I,
    "oo": sp.oo,
}

_ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Constant,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.UAdd,
    ast.USub,
)

_FORBIDDEN_IMPORTS = frozenset({
    "subprocess", "socket", "urllib", "http", "ftplib", "smtplib", "ctypes",
    "winreg", "shutil", "importlib", "multiprocessing", "webbrowser", "requests", "httpx",
})
_FORBIDDEN_CALLS = frozenset({"eval", "exec", "compile", "__import__", "system", "popen", "rmtree"})


@dataclass
class VerificationResult:
    status: str  # PROVEN_EXACT | TESTS_PASSED | NUMERICAL_VERIFIED | REFUTED | SYNTAX_ERROR | SECURITY_BLOCKED | TIMEOUT | ERROR
    passed: bool
    confidence: float
    feedback: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "passed": self.passed,
            "confidence": self.confidence,
            "feedback": self.feedback,
            "evidence": self.evidence,
        }


def safe_parse_expr(expr: str, extra_symbols: dict[str, sp.Symbol] | None = None) -> sp.Expr:
    """Safely parse a mathematical expression into SymPy without eval/attribute access."""
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError("empty expression")
    if len(expr) > MAX_EXPR_CHARS:
        raise ValueError("expression exceeds maximum length")
    norm = expr.strip().replace("^", "**").replace("×", "*").replace("÷", "/").replace("−", "-")
    try:
        tree = ast.parse(norm, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"syntax error at offset {exc.offset}: {exc.msg}") from None

    sym_names = set(extra_symbols or {})
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise ValueError(f"disallowed expression element: {type(node).__name__}")
        if isinstance(node, ast.Name):
            if node.id.startswith("_") or (node.id not in _SYMPY_FUNCS and node.id not in sym_names):
                raise ValueError(f"unknown symbol or function: {node.id!r}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _SYMPY_FUNCS or node.keywords:
                raise ValueError("only whitelisted positional math functions are permitted")

    ns: dict[str, Any] = {**_SYMPY_FUNCS, **(extra_symbols or {})}
    return sp.sympify(norm, locals=ns, evaluate=True)


def _symbolic_zero(diff: sp.Expr) -> tuple[bool, sp.Expr]:
    for fn in (sp.simplify, sp.expand, sp.factor, sp.radsimp, sp.trigsimp):
        cand = fn(diff)
        if cand == 0:
            return True, sp.Integer(0)
        diff = cand
    return diff.equals(0) is True, diff


class MathVerifier:
    """Symbolic + 50-digit arbitrary-precision mathematical verifier."""

    @staticmethod
    def verify_identity(lhs: str, rhs: str, var_names: list[str] | tuple[str, ...] = (),
                        samples: int = 12, seed: int = 7) -> VerificationResult:
        syms = {v: sp.Symbol(v, real=True) for v in var_names}
        try:
            l_expr = safe_parse_expr(lhs, syms)
            r_expr = safe_parse_expr(rhs, syms)
        except (ValueError, sp.SympifyError, TypeError) as exc:
            return VerificationResult("SYNTAX_ERROR", False, 0.0, f"Invalid math expression: {exc}")

        diff = l_expr - r_expr
        is_zero, reduced = _symbolic_zero(diff)
        if is_zero:
            return VerificationResult(
                "PROVEN_EXACT", True, 1.0,
                f"Symbolically proven: ({lhs}) - ({rhs}) == 0",
                {"lhs": str(l_expr), "rhs": str(r_expr)},
            )

        if not syms:
            mp.mp.dps = 50
            try:
                val = abs(mp.mpc(diff.evalf(50)))
            except (TypeError, ValueError) as exc:
                return VerificationResult("ERROR", False, 0.0, f"Evaluation failed: {exc}")
            if val < mp.mpf("1e-35"):
                return VerificationResult(
                    "NUMERICAL_VERIFIED", True, 0.99,
                    f"Verified numerically to 50 digits (residual={mp.nstr(val, 4)})",
                    {"residual": mp.nstr(val, 6)},
                )
            return VerificationResult(
                "REFUTED", False, 0.0,
                f"Not equal: simplified difference is {reduced} (residual={mp.nstr(val, 6)})",
                {"simplified_diff": str(reduced), "residual": mp.nstr(val, 6)},
            )

        rng = random.Random(seed)
        mp.mp.dps = 50
        sym_list = [syms[v] for v in var_names]
        max_err = mp.mpf("0")
        checked = 0
        for _ in range(samples * 3):
            if checked >= samples:
                break
            point = {s: sp.Rational(rng.randint(2, 37), rng.randint(1, 7)) for s in sym_list}
            try:
                val = abs(mp.mpc(diff.subs(point).evalf(50)))
            except (ZeroDivisionError, TypeError, ValueError):
                continue
            if mp.isnan(val) or mp.isinf(val):
                continue
            checked += 1
            if val > max_err:
                max_err = val
            if val > mp.mpf("1e-30"):
                return VerificationResult(
                    "REFUTED", False, 0.0,
                    f"Counterexample at { {str(k): str(v) for k, v in point.items()} }: residual={mp.nstr(val, 6)}",
                    {"counterexample": {str(k): str(v) for k, v in point.items()}, "residual": mp.nstr(val, 6)},
                )

        if checked == 0:
            return VerificationResult("ERROR", False, 0.0, "No valid domain sample points evaluated cleanly")
        return VerificationResult(
            "NUMERICAL_VERIFIED", True, 0.98,
            f"Verified across {checked} random rational points at 50-digit precision (max residual={mp.nstr(max_err, 4)})",
            {"samples_checked": checked, "max_residual": mp.nstr(max_err, 6)},
        )

    @staticmethod
    def verify_equation_solution(lhs: str, rhs: str, assignments: dict[str, str],
                                 extra_symbols: list[str] | tuple[str, ...] = ()) -> VerificationResult:
        all_names = sorted(set(assignments) | set(extra_symbols))
        syms = {v: sp.Symbol(v) for v in all_names}
        try:
            l_expr = safe_parse_expr(lhs, syms)
            r_expr = safe_parse_expr(rhs, syms)
            subs = {syms[k]: safe_parse_expr(str(v), syms) for k, v in assignments.items()}
        except (ValueError, sp.SympifyError, TypeError) as exc:
            return VerificationResult("SYNTAX_ERROR", False, 0.0, f"Invalid expression: {exc}")

        residual = (l_expr - r_expr).subs(subs)
        free = sorted(str(s) for s in residual.free_symbols)
        return MathVerifier.verify_identity(str(residual), "0", var_names=free)


def audit_python_ast(code: str) -> str | None:
    """Return None if code parses and avoids forbidden unsafe constructs, else a concise reason."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError at line {exc.lineno}, col {exc.offset}: {exc.msg}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _FORBIDDEN_IMPORTS:
                    return f"Security policy blocks import of {root!r} inside verification sandbox"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in _FORBIDDEN_IMPORTS:
                return f"Security policy blocks import from {root!r} inside verification sandbox"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
                return f"Security policy blocks call to {node.func.id!r} inside verification sandbox"
            if isinstance(node.func, ast.Attribute) and node.func.attr in _FORBIDDEN_CALLS:
                return f"Security policy blocks call to .{node.func.attr}() inside verification sandbox"
    return None


class CodeVerifier:
    """AST-guarded isolated Python execution verifier with structured failure feedback."""

    @staticmethod
    def verify_python(code: str, tests: str, timeout_s: float = 5.0) -> VerificationResult:
        if not isinstance(code, str) or not code.strip():
            return VerificationResult("SYNTAX_ERROR", False, 0.0, "Empty candidate code")
        err = audit_python_ast(code)
        if err:
            kind = "SYNTAX_ERROR" if err.startswith("SyntaxError") else "SECURITY_BLOCKED"
            return VerificationResult(kind, False, 0.0, err)
        test_err = audit_python_ast(tests)
        if test_err:
            return VerificationResult("ERROR", False, 0.0, f"Invalid test harness: {test_err}")

        script = f"{code.rstrip()}\n\n# --- OMEGA VERIFIER TESTS ---\n{tests.strip()}\nprint('__OMEGA_VERIFIED_OK__')\n"
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="omega_verify_") as tmp:
            target = Path(tmp) / "verify_run.py"
            target.write_text(script, encoding="utf-8")
            safe_env = {
                k: os.environ[k]
                for k in ("PATH", "SYSTEMROOT", "SystemRoot", "WINDIR", "TEMP", "TMP")
                if k in os.environ
            }
            safe_env["PYTHONIOENCODING"] = "utf-8"
            safe_env["PYTHONNOUSERSITE"] = "1"
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", str(target)],
                    cwd=tmp,
                    capture_output=True,
                    stdin=subprocess.DEVNULL,
                    timeout=max(0.5, min(float(timeout_s), 30.0)),
                    env=safe_env,
                )
            except subprocess.TimeoutExpired:
                return VerificationResult("TIMEOUT", False, 0.0, f"Execution timed out after {timeout_s}s")

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        stdout = proc.stdout[:MAX_OUT_CHARS].decode("utf-8", errors="replace")
        stderr = proc.stderr[:MAX_OUT_CHARS].decode("utf-8", errors="replace")
        if proc.returncode == 0 and "__OMEGA_VERIFIED_OK__" in stdout:
            return VerificationResult(
                "TESTS_PASSED", True, 1.0,
                f"All verification tests passed in {elapsed_ms} ms",
                {"elapsed_ms": elapsed_ms, "stdout": stdout.replace("__OMEGA_VERIFIED_OK__", "").strip()},
            )
        detail = (stderr or stdout or f"exit code {proc.returncode}").strip()
        return VerificationResult(
            "REFUTED", False, 0.0,
            f"Test execution failed: {detail[-1200:]}",
            {"exit_code": proc.returncode, "elapsed_ms": elapsed_ms, "stderr": stderr[-1200:]},
        )
