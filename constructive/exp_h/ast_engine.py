"""Restricted AST interpreter and repair search for H5 (programming intelligence).

Programs are immutable nested tuples interpreted by a step-bounded machine.
Nothing executes Python source: every node is written in a closed schema.
Loops are bounded by construction (``MAX_LOOP_ITERS``) and total evaluation is
bounded by ``MAX_STEPS``.

Schema
------
("const", value)
("var", name)
("binop", op, left, right)         -- + - * // %
("cmp", op, left, right)          -- < <= > >= == !=  -> 0/1 int
("call", name, *args)             -- len, sum, abs, min, max
("list", [items])                 -- constant list of ints
("assign", name, expr)            -- stmt
("if", cond, then_stmts, else_stmts)  -- stmt
("while", cond, body_stmts)       -- stmt (bounded)
("return", expr)                  -- stmt

Comparisons and arithmetic produce plain Python ints (arbitrary precision);
booleans are 0/1 ints by design.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

Node = tuple
Stmt = tuple

MAX_STEPS = 20_000
MAX_LOOP_ITERS = 2_000
#: programs whose arithmetic escapes this bound are treated as nonsense by
#: construction (a bomb candidate can never be a legitimate repair).
MAG_LIMIT = 10 ** 18


class AstError(Exception):
    """Program, type or budget failure.  Never silently absorbed."""


def C(v) -> Node:
    return ("const", int(v))


def V(name: str) -> Node:
    return ("var", str(name))


def Li(items) -> Node:
    return ("list", [int(i) for i in items])


def B(op: str, l: Node, r: Node) -> Node:
    return ("binop", op, l, r)


def Cmp(op: str, l: Node, r: Node) -> Node:
    return ("cmp", op, l, r)


def Call(name: str, *args: Node) -> Node:
    return ("call", name, *args)


def Assign(name: str, expr: Node) -> Stmt:
    return ("assign", name, expr)


def If(cond: Node, then_: list[Stmt], else_: list[Stmt]) -> Stmt:
    return ("if", cond, list(then_), list(else_))


def While(cond: Node, body: list[Stmt]) -> Stmt:
    return ("while", cond, list(body))


def Ret(expr: Node) -> Stmt:
    return ("return", expr)


# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------


def evaluate_ast(program: list[Stmt], env_in: dict[str, Any],
                 max_steps: int | None = None) -> int:
    steps = [0]
    cap = max_steps or MAX_STEPS
    env = dict(env_in)

    def tick() -> None:
        steps[0] += 1
        if steps[0] > cap:
            raise AstError("step budget exhausted")

    def need_int(v: Any, ctx: str) -> int:
        if isinstance(v, bool) or not isinstance(v, int):
            raise AstError(f"expected int for {ctx}, got {type(v).__name__}")
        return v

    def ev(node: Node):
        tick()
        kind = node[0]
        if kind == "const":
            return int(node[1])
        if kind == "var":
            if node[1] not in env:
                raise AstError(f"unbound variable {node[1]!r}")
            return env[node[1]]
        if kind == "list":
            return [need_int(v, "list constant") for v in node[1]]
        if kind == "binop":
            op, lnode, rnode = node[1], node[2], node[3]
            l = need_int(ev(lnode), "binop lhs")
            r = need_int(ev(rnode), "binop rhs")
            if abs(l) > MAG_LIMIT or abs(r) > MAG_LIMIT:
                raise AstError("integer magnitude exceeded")
            if op == "+":
                out = l + r
            elif op == "-":
                out = l - r
            elif op == "*":
                out = l * r
            elif op == "//":
                if r == 0:
                    raise AstError("integer division by zero")
                q = abs(l) // abs(r)
                out = q if (l >= 0) == (r >= 0) else -q
            elif op == "%":
                if r == 0:
                    raise AstError("modulo by zero")
                out = l % r if (l >= 0) == (r >= 0) else -(abs(l) % abs(r))
            else:
                raise AstError(f"unknown binop {op!r}")
            if abs(out) > MAG_LIMIT:
                raise AstError("integer magnitude exceeded")
            return out
        if kind == "cmp":
            op, lnode, rnode = node[1], node[2], node[3]
            l = need_int(ev(lnode), "cmp lhs")
            r = need_int(ev(rnode), "cmp rhs")
            table = {"<": l < r, "<=": l <= r, ">": l > r, ">=": l >= r,
                     "==": l == r, "!=": l != r}
            if op not in table:
                raise AstError(f"unknown cmp {op!r}")
            return 1 if table[op] else 0
        if kind == "call":
            fn = node[1]
            args = [ev(a) for a in node[2:]]
            if fn == "len":
                if not isinstance(args[0], list):
                    raise AstError("len expects a list")
                return len(args[0])
            if fn == "sum":
                if not isinstance(args[0], list):
                    raise AstError("sum expects a list")
                return sum(args[0])
            if fn == "abs":
                return abs(need_int(args[0], "abs"))
            if fn == "min":
                return min(need_int(a, "min") for a in args)
            if fn == "max":
                return max(need_int(a, "max") for a in args)
            raise AstError(f"unknown builtin {fn!r}")
        if kind == "assign":  # not legal as expression
            raise AstError("assign is a statement, not an expression")
        raise AstError(f"unknown expression node {node[0]!r}")

    def run(stmts: list[Stmt]):
        for stmt in stmts:
            tick()
            kind = stmt[0]
            if kind == "assign":
                env[stmt[1]] = ev(stmt[2])
            elif kind == "if":
                branch = stmt[2] if ev(stmt[1]) != 0 else stmt[3]
                out = run(branch)
                if out is not None:
                    return out
            elif kind == "while":
                guard = stmt[1]
                body = stmt[2]
                iters = 0
                while ev(guard) != 0:
                    out = run(body)
                    if out is not None:
                        return out
                    iters += 1
                    if iters > MAX_LOOP_ITERS:
                        raise AstError("loop iter budget exhausted")
            elif kind == "return":
                v = ev(stmt[1])
                return v
            else:
                raise AstError(f"unknown stmt {stmt[0]!r}")
        return None

    result = run(program)
    if result is None:
        raise AstError("program fell off the end without returning")
    return result


# ---------------------------------------------------------------------------
# Expression-site enumeration (only repair-relevant sites)
# ---------------------------------------------------------------------------


def enumerate_expr_sites(program):
    """Return [(path, node)] for every binop/cmp/const expression node."""
    out: list[tuple[tuple, tuple]] = []

    def visit(node, path):
        kind = node[0]
        if kind in ("binop", "cmp"):
            out.append((path, node))
            visit(node[2], path + (2,))
            visit(node[3], path + (3,))
        elif kind == "const":
            out.append((path, node))
        elif kind == "call":
            out.append((path, node))
            for i, a in enumerate(node[2:]):
                visit(a, path + (2 + i,))

    def visit_stmt_list(stmts, path):
        for i, stmt in enumerate(stmts):
            visit_stmt(stmt, path + (i,))

    def visit_stmt(stmt, path):
        kind = stmt[0]
        if kind == "assign":
            visit(stmt[2], path + (2,))
        elif kind == "if":
            visit(stmt[1], path + (1,))
            visit_stmt_list(stmt[2], path + (2,))
            visit_stmt_list(stmt[3], path + (3,))
        elif kind == "while":
            visit(stmt[1], path + (1,))
            visit_stmt_list(stmt[2], path + (2,))
        elif kind == "return":
            visit(stmt[1], path + (1,))
        else:
            raise AstError(f"unknown stmt {stmt[0]!r}")

    visit_stmt_list(program, ())
    return out


def replace_at(program, path, value):
    def t(node, p):
        if not p:
            return value
        idx = p[0]
        rest = p[1:]
        if isinstance(node, list):
            out = list(node)
            out[idx] = t(out[idx], rest)
            return out
        lst = list(node)
        lst[idx] = t(lst[idx], rest)
        return tuple(lst)
    return t(program, list(path))


CMP_ALTS = {"<": ("<=", ">", ">=", "==", "!="),
            "<=": ("<", ">", ">=", "==", "!="),
            ">": (">=", "<", "<=", "==", "!="),
            ">=": (">", "<", "<=", "==", "!="),
            "==": ("!=", "<", "<=", ">", ">="),
            "!=": ("==", "<", "<=", ">", ">=")}
BINOP_ALTS = {"+": ("-", "*", "//", "%"),
              "-": ("+", "*", "//", "%"),
              "*": ("+", "-", "//", "%"),
              "//": ("+", "-", "*", "%"),
              "%": ("+", "-", "*", "//")}
CONST_DELTAS = (-2, -1, 1, 2)


def apply_bug(program, rng):
    import copy
    prog = copy.deepcopy(program)
    sites = enumerate_expr_sites(prog)
    if not sites:
        return prog
    path, node = rng.choice(sites)
    return rewrite_site(prog, path, node, rng)


def rewrite_site(program, path, node, rng):
    kind = node[0]
    if kind == "const":
        return replace_at(program, path, ("const", node[1] + rng.choice(CONST_DELTAS)))
    if kind == "cmp":
        return replace_at(program, path, ("cmp", rng.choice(CMP_ALTS[node[1]]), node[2], node[3]))
    if kind == "binop":
        return replace_at(program, path, ("binop", rng.choice(BINOP_ALTS[node[1]]), node[2], node[3]))
    raise AstError(f"cannot rewrite site {node[0]!r}")


def candidate_rewrites(path, node):
    """The replacement values another reasonable engine would propose here."""
    kind = node[0]
    if kind == "const":
        return [(path, ("const", node[1] + d)) for d in CONST_DELTAS]
    if kind == "cmp":
        return [(path, ("cmp", op, node[2], node[3])) for op in CMP_ALTS[node[1]]]
    if kind == "binop":
        return [(path, ("binop", op, node[2], node[3])) for op in BINOP_ALTS[node[1]]]
    return []


# ---------------------------------------------------------------------------
# Repair search
# ---------------------------------------------------------------------------


def repair_program(buggy, tests, verify, *, max_width=1, max_candidates=2000,
                   node_budget=None):
    """Find the smallest fix passing all tests.

    ``tests``   list of (env dict, expected output) pairs the fix must satisfy.
    ``verify``  optional extra predicate(candidate) -> bool (boundary/oracle);
                return False to veto candidates.
    Returns (candidate_or_None, n_candidates_tried, reason).
    """
    from itertools import combinations, product

    sites = enumerate_expr_sites(buggy)
    if not sites:
        return None, 0, "nothing to rewrite"

    def test_ok(cand):
        try:
            for env, expected in tests:
                if evaluate_ast(cand, env, max_steps=300) != expected:
                    return False
        except AstError:
            return False
        if verify is not None:
            try:
                return bool(verify(cand))
            except AstError:
                return False
        return True

    tried = 0
    widths = range(1, max_width + 1)
    for width in widths:
        # theoretical size of the combination space per site group
        combos = combinations(sites, width)
        for combo in combos:
            alt_lists = [candidate_rewrites(p, n) for p, n in combo]
            if any(not a for a in alt_lists):
                continue
            for alt_combo in product(*alt_lists):
                cand = buggy
                for path, value in alt_combo:
                    cand = replace_at(cand, path, value)
                tried += 1
                if test_ok(cand):
                    return cand, tried, "ok"
                if node_budget is not None and tried >= node_budget:
                    return None, tried, "budget-exhausted"
                if tried >= max_candidates:
                    return None, tried, "candidate-cap-exhausted"
    return None, tried, "no-repair-found"


def exhaustive_bounded_diff(candidate, reference, domain):
    """True iff candidate and reference agree on the bounded integer domain."""
    for env in domain:
        try:
            if evaluate_ast(candidate, env) != evaluate_ast(reference, env):
                return False
        except AstError:
            return False
    return True


# ---------------------------------------------------------------------------
# Template library (correct programs to corrupt)
# ---------------------------------------------------------------------------


def build_sum_range():
    return [
        Assign("s", C(0)), Assign("i", V("a")),
        While(Cmp("<=", V("i"), V("b")),
              [Assign("s", B("+", V("s"), V("i"))),
               Assign("i", B("+", V("i"), C(1)))]),
        Ret(V("s")),
    ]


def build_sum_squares():
    return [
        Assign("s", C(0)), Assign("i", C(1)),
        While(Cmp("<=", V("i"), V("n")),
              [Assign("s", B("+", V("s"), B("*", V("i"), V("i")))),
               Assign("i", B("+", V("i"), C(1)))]),
        Ret(V("s")),
    ]


def build_pow2():
    return [
        Assign("s", C(1)), Assign("i", C(0)),
        While(Cmp("<", V("i"), V("n")),
              [Assign("s", B("*", V("s"), C(2))),
               Assign("i", B("+", V("i"), C(1)))]),
        Ret(V("s")),
    ]


def build_sum_mult():
    return [
        Assign("s", C(0)), Assign("i", C(0)),
        While(Cmp("<", V("i"), V("n")),
              [Assign("s", B("+", V("s"), B("*", V("i"), V("m")))),
               Assign("i", B("+", V("i"), C(1)))]),
        Ret(V("s")),
    ]


def build_sum_gcd():
    return [
        Assign("a", V("a")), Assign("b", V("b")),
        While(Cmp("!=", V("b"), C(0)),
              [Assign("t", V("b")),
               Assign("b", B("%", V("a"), V("b"))),
               Assign("a", V("t"))]),
        Ret(V("a")),
    ]


def build_clamp():
    return [
        If(Cmp("<", V("x"), V("lo")), [Ret(V("lo"))], []),
        If(Cmp(">", V("x"), V("hi")), [Ret(V("hi"))], []),
        Ret(V("x")),
    ]


TEMPLATE_BUILDERS_H5 = {
    "sum_range": build_sum_range,
    "sum_squares": build_sum_squares,
    "pow2": build_pow2,
    "sum_mult": build_sum_mult,
    "sum_gcd": build_sum_gcd,
    "clamp": build_clamp,
}

#: templates seen in the repair "curriculum" (L1/L2), vs held out (L4)
SEEN_TEMPLATES = ("sum_range", "sum_squares", "pow2")
UNSEEN_TEMPLATES = ("sum_mult", "sum_gcd", "clamp")


def template_correct(name: str) -> list[Stmt]:
    return TEMPLATE_BUILDERS_H5[name]()


def template_args(name: str) -> list[str]:
    return {"sum_range": ["a", "b"], "sum_squares": ["n"], "pow2": ["n"],
            "sum_mult": ["n", "m"], "sum_gcd": ["a", "b"],
            "clamp": ["x", "lo", "hi"]}[name]


def sample_inputs(name: str, rng, n: int, lo: int = 0, hi: int = 8):
    args = template_args(name)
    out = []
    seen = set()
    span = hi - lo + 1
    space = span ** len(args)

    def valid(vals):
        return not (name == "clamp" and vals[1] > vals[2])

    if space < n:
        # not enough distinct inputs: uniqueness cannot be enforced, so
        # fall back to unrestricted (duplicated) sampling rather than hang
        tries = 0
        while len(out) < n and tries < 10_000:
            tries += 1
            vals = tuple(rng.randint(lo, hi) for _ in args)
            if not valid(vals):
                continue
            out.append(dict(zip(args, vals)))
        return out

    while len(out) < n:
        vals = tuple(rng.randint(lo, hi) for _ in args)
        if not valid(vals):
            continue
        if vals in seen:
            continue
        seen.add(vals)
        out.append(dict(zip(args, vals)))
    return out


def buggy_from_template(name: str, rng, n_bugs: int = 1,
                        check_envs: list[dict] | None = None):
    prog = template_correct(name)
    for _ in range(n_bugs):
        for _attempt in range(8):
            candidate = apply_bug(prog, rng)
            if check_envs is None:
                prog = candidate
                break
            try:
                for env in check_envs:
                    out = evaluate_ast(candidate, env, max_steps=2000)
                    if not isinstance(out, int):
                        raise AstError("non-int output")
                prog = candidate
                break
            except AstError:
                continue
        else:
            return None  # could not produce a terminating bug; drop honestly
    return prog
