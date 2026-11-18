"""Exact math engine: a specialized algorithm instead of token generation.

Evaluates arithmetic with arbitrary-precision integers and exact rationals
via a whitelisted AST walk (no eval). Falls back to float only for functions
that are inherently irrational (sqrt of non-squares, trig, logs).
"""

from __future__ import annotations

import ast
import math
import operator
from fractions import Fraction

from .base import Param, Tool, ToolContext, ToolError

MAX_EXP_BITS = 100_000  # bound result size so 9**9**9 cannot exhaust memory
MAX_NODES = 500

_BIN = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
}
_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _isqrt_exact(x):
    if isinstance(x, Fraction) and x >= 0:
        n, d = math.isqrt(x.numerator), math.isqrt(x.denominator)
        if n * n == x.numerator and d * d == x.denominator:
            return Fraction(n, d)
    if x < 0:
        raise ToolError("sqrt of a negative number")
    return math.sqrt(float(x))


def _factorial(x):
    if isinstance(x, Fraction) and x.denominator == 1 and 0 <= x <= 5000:
        return Fraction(math.factorial(int(x)))
    raise ToolError("factorial needs an integer in [0, 5000]")


def _gcd(a, b):
    if isinstance(a, Fraction) and isinstance(b, Fraction) and a.denominator == b.denominator == 1:
        return Fraction(math.gcd(int(a), int(b)))
    raise ToolError("gcd needs integers")


_FUNCS = {
    "sqrt": _isqrt_exact, "abs": abs, "factorial": _factorial, "gcd": _gcd,
    "sin": lambda x: math.sin(float(x)), "cos": lambda x: math.cos(float(x)),
    "tan": lambda x: math.tan(float(x)), "log": lambda x: math.log(float(x)),
    "log10": lambda x: math.log10(float(x)), "log2": lambda x: math.log2(float(x)),
    "exp": lambda x: math.exp(float(x)), "floor": lambda x: Fraction(math.floor(x)),
    "ceil": lambda x: Fraction(math.ceil(x)), "round": lambda x: Fraction(round(x)),
}
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


def _norm(expr: str) -> str:
    table = str.maketrans("٠١٢٣٤٥٦٧٨٩×÷−", "0123456789*/-")
    expr = expr.translate(table).replace("^", "**").replace("√", "sqrt").strip().rstrip("=?؟").strip()
    return expr


def _ev(node, budget):
    budget[0] += 1
    if budget[0] > MAX_NODES:
        raise ToolError("expression too large")
    if isinstance(node, ast.Expression):
        return _ev(node.body, budget)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return Fraction(node.value) if isinstance(node.value, int) else Fraction(str(node.value))
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
        return _UN[type(node.op)](_ev(node.operand, budget))
    if isinstance(node, ast.BinOp):
        left, right = _ev(node.left, budget), _ev(node.right, budget)
        if isinstance(node.op, ast.Pow):
            return _pow(left, right)
        if type(node.op) in _BIN:
            try:
                return _BIN[type(node.op)](left, right)
            except ZeroDivisionError:
                raise ToolError("division by zero") from None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*[_ev(a, budget) for a in node.args])
    raise ToolError(f"unsupported expression element: {type(node).__name__}")


def _pow(base, exp):
    if isinstance(exp, Fraction) and exp.denominator == 1 and isinstance(base, Fraction):
        e = int(exp)
        bits = abs(e) * max(base.numerator.bit_length(), base.denominator.bit_length(), 1)
        if bits > MAX_EXP_BITS:
            raise ToolError("result too large (bounded at ~100k bits)")
        if base == 0 and e < 0:
            raise ToolError("division by zero")
        return base ** e
    return float(base) ** float(exp)


def _format(v) -> tuple[str, bool]:
    if isinstance(v, Fraction):
        if v.denominator == 1:
            return str(v.numerator), True
        return f"{v.numerator}/{v.denominator}", True
    if isinstance(v, float) and v.is_integer() and abs(v) < 2**53:
        return str(int(v)), False
    return repr(v), False


def evaluate(expr: str) -> dict:
    src = _norm(expr)
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as e:
        raise ToolError(f"cannot parse expression: {e.msg}") from None
    value = _ev(tree, [0])
    text, exact = _format(value)
    out = {"ok": True, "expr": src, "value": text, "exact": exact}
    if isinstance(value, Fraction) and value.denominator != 1:
        out["decimal"] = f"{float(value):.15g}"
    return out


def _tool(ctx: ToolContext, expr: str) -> dict:
    return evaluate(expr)


TOOLS = [
    Tool(
        name="math.eval",
        description="Exact arithmetic (big integers, rationals) plus sqrt, factorial, gcd, trig, logs.",
        params={"expr": Param("str")}, scopes=frozenset(), fn=_tool,
    )
]
