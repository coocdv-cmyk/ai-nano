"""Execution-verified AST repair for multi-file Python workspaces (EXP-N N1).

The engine is deliberately conservative:

* A candidate is accepted only after the workspace's own test command has run
  and returned ``0`` inside an isolated copy. There is no path that reports a
  repair without executing the suite.
* It never reads a sealed reference. The only specification it sees is the
  failing test suite, the tracebacks, and the repository's own AST.
* Every candidate is a real AST transformation; the search is bounded by a
  verifier-call budget that is reported with the result.
* Localisation uses the failing test bodies plus ``RepoGraph`` symbol and call
  edges, not file names guessed from the task description.

Known limitations (measured, not hidden):

* Files touched by a patch are emitted with ``ast.unparse``, so comments and
  blank-line style inside a rewritten file are not preserved.
* Nine operator families are supported (see ``OPERATORS``). A bug outside them
  is refuted and reported, never guessed at.
* The engine executes the workspace's test suite. That is arbitrary code with
  the privileges of the caller; it is *not* an OS sandbox (repository defect
  #4). Only run it on workspaces whose tests you would run yourself.
"""

from __future__ import annotations

import ast
import copy
import difflib
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .repo_graph import RepoGraph, SymbolRecord, index_repository

__all__ = [
    "Failure",
    "PatchCandidate",
    "RepairEngine",
    "RepairReport",
    "RepairTarget",
    "TestRun",
    "analyze_failures",
    "OPERATORS",
    "parse_pytest_output",
    "run_pytest",
]

# Deterministic environment for test subprocesses: no inherited secrets.
_SAFE_ENV_KEYS = ("PATH", "SYSTEMROOT", "SystemRoot", "WINDIR", "TEMP", "TMP", "PATHEXT", "COMSPEC")

_FAILED_RE = re.compile(r"^FAILED\s+(?P<nodeid>\S+)(?:\s+-\s+(?P<message>.*))?$")
_ERROR_RE = re.compile(r"^ERROR\s+(?P<nodeid>\S+)(?:\s+-\s+(?P<message>.*))?$")
_COUNT_RE = re.compile(r"(?P<count>\d+)\s+(?:failed|error|errors|passed|skipped)")
_TRACEBACK_SITE_RE = re.compile(r"^(?P<file>[A-Za-z0-9_./\\-]+\.py):(?P<line>\d+):\s")

OPERATORS = (
    "mutable_default",
    "exception_scope",
    "cache_key",
    "closure_binding",
    "boundary_guard",
    "off_by_one",
    "integer_distribution",
    "mapping_fallback",
    "comparison_flip",
)

_PRIORITY = {
    "mutable_default": 10,
    "exception_scope": 15,
    "cache_key": 20,
    "closure_binding": 22,
    "boundary_guard": 25,
    "off_by_one": 30,
    "integer_distribution": 35,
    "mapping_fallback": 40,
    "comparison_flip": 45,
}

_IGNORED_DIRS = frozenset({"__pycache__", ".pytest_cache", ".git", ".hg", ".svn", ".omega", ".venv", "venv"})


# --------------------------------------------------------------------------- #
# Running the suite
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Failure:
    nodeid: str
    file: str
    function: str
    message: str = ""


@dataclass(frozen=True)
class TestRun:
    passed: bool
    returncode: int
    failed: int
    failures: tuple[Failure, ...]
    summary: str
    elapsed_s: float
    stdout: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "returncode": self.returncode,
            "failed": self.failed,
            "summary": self.summary,
            "elapsed_s": self.elapsed_s,
            "failures": [f.nodeid for f in self.failures],
        }


def _safe_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {k: os.environ[k] for k in _SAFE_ENV_KEYS if k in os.environ}
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # The verifier must be hermetic: autoloaded third-party pytest plugins can
    # shadow a workspace module name (e.g. `filters` or `renderers`) and hang the
    # subprocess, which would be misrecorded as an unsolvable repair task.
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    if extra:
        env.update(extra)
    return env


def parse_pytest_output(stdout: str, stderr: str = "", returncode: int = 1) -> tuple[bool, int, tuple[Failure, ...], str]:
    """Parse ``pytest -q`` output into (passed, failed_count, failures, summary)."""
    text = f"{stdout}\n{stderr}"
    failures: list[Failure] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        match = _FAILED_RE.match(line) or _ERROR_RE.match(line)
        if not match:
            continue
        nodeid = match.group("nodeid")
        if nodeid in seen:
            continue
        seen.add(nodeid)
        file_part, _, rest = nodeid.partition("::")
        function = rest.split("::")[-1] if rest else ""
        failures.append(Failure(nodeid=nodeid, file=file_part, function=function,
                                message=(match.group("message") or "").strip()))

    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    summary = lines[-1] if lines else f"exit {returncode}"

    if returncode == 0:
        return True, 0, tuple(failures), summary

    failed = 0
    match = re.search(r"(\d+)\s+failed", summary)
    if match:
        failed = int(match.group(1))
    elif failures:
        failed = len(failures)
    else:
        failed = 999  # collection/import error with no per-test failure lines
    return False, failed, tuple(failures), summary


def run_pytest(workspace: str | Path, tests_dir: str = "tests", timeout_s: float = 60.0) -> TestRun:
    """Run the workspace's own suite in place, with a sanitised environment."""
    workspace = Path(workspace)
    cmd = [sys.executable, "-m", "pytest", "-q", tests_dir, "-p", "no:cacheprovider"]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(workspace),
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=max(5.0, float(timeout_s)),
            env=_safe_env({"PYTHONPATH": str(workspace)}),
        )
    except subprocess.TimeoutExpired:
        return TestRun(False, None, 999, (), f"TIMEOUT after {timeout_s}s",
                       round(time.perf_counter() - t0, 3), "")
    stdout = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace")
    passed, failed, failures, summary = parse_pytest_output(stdout, stderr, proc.returncode)
    return TestRun(passed, proc.returncode, failed, failures, summary,
                   round(time.perf_counter() - t0, 3), stdout)


# --------------------------------------------------------------------------- #
# Localisation
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RepairTarget:
    symbol: SymbolRecord
    score: float
    depth: int
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.symbol.name,
            "qualname": self.symbol.qualname,
            "file": self.symbol.file,
            "lineno": self.symbol.lineno,
            "score": round(self.score, 2),
            "depth": self.depth,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class CallSite:
    function: str
    module: str | None
    args: tuple[int | None, ...]
    raises: str | None = None


def _literal_int(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _literal_int(node.operand)
        return -inner if inner is not None else None
    return None


def _call_name(node: ast.Call) -> tuple[str, str | None]:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id, None
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name):
            return func.attr, func.value.id
        return func.attr, None
    try:
        return ast.unparse(func), None
    except Exception:
        return "", None


def _raises_exception(expr: ast.AST) -> str | None:
    if not isinstance(expr, ast.Call):
        return None
    name, _ = _call_name(expr)
    if name != "raises" or not expr.args:
        return None
    try:
        return ast.unparse(expr.args[0])
    except Exception:
        return None


class _CallCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.sites: list[CallSite] = []
        self._stack: list[str | None] = []

    def visit_With(self, node: ast.With) -> None:
        entered: list[str | None] = []
        for item in node.items:
            exc = _raises_exception(item.context_expr)
            entered.append(exc)
            self._stack.append(exc)
        for stmt in node.body:
            self.visit(stmt)
        for _ in entered:
            self._stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        name, module = _call_name(node)
        args = tuple(_literal_int(arg) for arg in node.args)
        raises = next((exc for exc in reversed(self._stack) if exc), None)
        self.sites.append(CallSite(function=name, module=module, args=args, raises=raises))
        self.generic_visit(node)


def _find_test_function(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _collect_calls(workspace: Path, failure: Failure) -> list[CallSite]:
    path = workspace / failure.file
    if not path.is_file() or not failure.function:
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except SyntaxError:
        return []
    fn = _find_test_function(tree, failure.function)
    if fn is None:
        return []
    collector = _CallCollector()
    for stmt in fn.body:
        collector.visit(stmt)
    return collector.sites


def _module_name(rel: str) -> str:
    parts = list(Path(rel).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _is_test_file(rel: str, tests_dir: str) -> bool:
    prefix = tests_dir.replace("\\", "/").rstrip("/") + "/"
    rel_posix = rel.replace("\\", "/")
    return rel_posix.startswith(prefix) or "/test_" in f"/{rel_posix}" or Path(rel_posix).name.startswith("test_")


def analyze_failures(
    workspace: str | Path,
    graph: RepoGraph,
    run: TestRun,
    tests_dir: str = "tests",
) -> tuple[list[RepairTarget], dict[str, list[CallSite]]]:
    """Rank the repo symbols implicated by the failing tests."""
    workspace = Path(workspace)
    by_name: dict[str, list[SymbolRecord]] = {}
    for sym in graph.symbols:
        if _is_test_file(sym.file, tests_dir):
            continue
        by_name.setdefault(sym.name, []).append(sym)

    calls_by_test: dict[str, list[CallSite]] = {}
    scores: dict[tuple[str, str, int], dict[str, Any]] = {}

    def bump(sym: SymbolRecord, amount: float, depth: int, reason: str) -> None:
        key = (sym.file, sym.name, sym.lineno)
        row = scores.setdefault(key, {"symbol": sym, "score": 0.0, "depth": depth, "reasons": []})
        row["score"] += amount
        row["depth"] = min(row["depth"], depth)
        if reason not in row["reasons"]:
            row["reasons"].append(reason)

    direct: list[SymbolRecord] = []
    for failure in run.failures:
        sites = _collect_calls(workspace, failure)
        calls_by_test[failure.nodeid] = sites
        for site in sites:
            candidates = by_name.get(site.function, [])
            if not candidates:
                continue
            chosen = candidates
            if site.module:
                narrowed = [s for s in candidates if s.file.startswith(f"{site.module}.py")
                            or s.file.startswith(f"{site.module}/")]
                chosen = narrowed or candidates
            for sym in chosen:
                bump(sym, 10.0, 0, f"called by {failure.nodeid}")
                if sym not in direct:
                    direct.append(sym)
        if failure.message:
            for name, syms in by_name.items():
                if name and name in failure.message:
                    for sym in syms:
                        bump(sym, 2.0, 0, "named in failure message")

    # Traceback sites (exceptions raised inside the repo, not in the test body).
    for raw in (run.stdout or "").splitlines():
        match = _TRACEBACK_SITE_RE.match(raw.strip())
        if not match:
            continue
        rel = match.group("file").replace("\\", "/")
        if _is_test_file(rel, tests_dir):
            continue
        line = int(match.group("line"))
        for sym in graph.symbols:
            if sym.file.replace("\\", "/") == rel and sym.lineno <= line <= sym.end_lineno:
                bump(sym, 20.0, 0, f"traceback frame {rel}:{line}")

    # One hop of call-graph context: the bug is often in a callee.
    for sym in list(direct):
        for callee in graph.callees_of(sym.name):
            name = callee.split(".")[-1]
            for cal in by_name.get(name, []):
                if cal.file == sym.file or True:
                    bump(cal, 3.0, 1, f"callee of {sym.name}")

    targets = [
        RepairTarget(symbol=row["symbol"], score=row["score"], depth=row["depth"],
                     reasons=tuple(row["reasons"]))
        for row in scores.values()
    ]
    targets.sort(key=lambda t: (-t.score, t.depth, t.symbol.file, t.symbol.lineno))
    return targets, calls_by_test


# --------------------------------------------------------------------------- #
# Candidate patches
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PatchCandidate:
    operator: str
    file: str
    target: str
    source: str
    description: str
    priority: int


@dataclass
class _Context:
    workspace: Path
    files: dict[str, str]
    targets: list[RepairTarget]
    calls: dict[str, list[CallSite]]


def _parse(source: str) -> ast.Module | None:
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _unparse(tree: ast.Module) -> str:
    ast.fix_missing_locations(tree)
    return ast.unparse(tree).rstrip() + "\n"


def _functions(tree: ast.Module) -> Iterator[ast.FunctionDef | ast.AsyncFunctionDef]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _target_index(ctx: _Context) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for target in ctx.targets:
        index.setdefault(target.symbol.name, set()).add(target.symbol.file)
    return index


def _is_target_function(fn: ast.FunctionDef | ast.AsyncFunctionDef, rel: str,
                        index: dict[str, set[str]]) -> bool:
    files = index.get(fn.name)
    if files is None:
        return False
    return not files or rel in files


def _positional_params(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    args = fn.args
    return [a.arg for a in list(args.posonlyargs) + list(args.args) if a.arg not in ("self", "cls")]


def _all_params(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    args = fn.args
    names = [a.arg for a in list(args.posonlyargs) + list(args.args)]
    names += [a.arg for a in args.kwonlyargs]
    if args.vararg:
        names.append(args.vararg.arg)
    if args.kwarg:
        names.append(args.kwarg.arg)
    return [n for n in names if n not in ("self", "cls")]


def _body_insert_index(fn: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    body = fn.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        return 1
    return 0


def _statement_list_owner(tree: ast.AST, target: ast.AST) -> list[ast.AST] | None:
    for node in ast.walk(tree):
        for _field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if item is target:
                        return value
    return None


def _replace_statement(tree: ast.AST, target: ast.AST, replacement: list[ast.stmt]) -> bool:
    owner = _statement_list_owner(tree, target)
    if owner is None:
        return False
    idx = owner.index(target)
    owner[idx:idx + 1] = replacement
    return True


def _candidate(ctx: _Context, operator: str, rel: str, source: str, target: str, description: str) -> PatchCandidate:
    return PatchCandidate(operator=operator, file=rel, target=target, source=source,
                          description=description, priority=_PRIORITY[operator])


# --- 1. mutable default arguments ------------------------------------------ #

_MUTABLE_LITERALS = (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)


def _is_mutable_default(node: ast.AST | None) -> bool:
    if node is None:
        return False
    if isinstance(node, _MUTABLE_LITERALS):
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in {"list", "dict", "set"} and not node.args and not node.keywords:
        return True
    return False


def _op_mutable_default(ctx: _Context) -> list[PatchCandidate]:
    index = _target_index(ctx)
    out: list[PatchCandidate] = []
    for rel, source in ctx.files.items():
        if _is_test_file(rel, "tests"):
            continue
        tree = _parse(source)
        if tree is None:
            continue
        touched: list[str] = []
        for fn in _functions(tree):
            if not _is_target_function(fn, rel, index):
                continue
            args = fn.args
            positional = list(args.posonlyargs) + list(args.args)
            offset = len(positional) - len(args.defaults)
            pairs: list[tuple[ast.arg, ast.AST, list[ast.AST]]] = []
            for i, default in enumerate(args.defaults):
                pairs.append((positional[offset + i], default, args.defaults))
            for arg, default in zip(args.kwonlyargs, args.kw_defaults):
                if default is not None:
                    pairs.append((arg, default, args.kw_defaults))
            guards: list[ast.stmt] = []
            changed = False
            for arg, default, owner in pairs:
                if not _is_mutable_default(default):
                    continue
                idx = next(i for i, item in enumerate(owner) if item is default)
                owner[idx] = ast.Constant(value=None)
                guard = ast.If(
                    test=ast.Compare(left=ast.Name(id=arg.arg, ctx=ast.Load()), ops=[ast.Is()],
                                     comparators=[ast.Constant(value=None)]),
                    body=[ast.Assign(targets=[ast.Name(id=arg.arg, ctx=ast.Store())],
                                     value=copy.deepcopy(default))],
                    orelse=[],
                )
                guards.append(guard)
                changed = True
            if changed:
                at = _body_insert_index(fn)
                fn.body[at:at] = guards
                touched.append(fn.name)
        if touched:
            out.append(_candidate(ctx, "mutable_default", rel, _unparse(tree),
                                  ",".join(touched),
                                  f"replace mutable default(s) in {', '.join(sorted(set(touched)))}"))
    return out


# --- 2. exception scope ---------------------------------------------------- #

def _base_name(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _workspace_exceptions(ctx: _Context) -> dict[str, str]:
    found: dict[str, str] = {}
    for rel, source in ctx.files.items():
        if _is_test_file(rel, "tests"):
            continue
        tree = _parse(source)
        if tree is None:
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and any(
                _base_name(base).split(".")[-1].endswith(("Error", "Exception")) for base in node.bases
            ):
                found[node.name] = _module_name(rel)
    return found


def _ensure_import(tree: ast.Module, module: str, name: str) -> None:
    if not module:
        return
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == module:
            if any(alias.name == name for alias in node.names):
                return
            node.names.append(ast.alias(name=name))
            return
        if isinstance(node, ast.Import):
            if any(alias.name == module for alias in node.names):
                return
    at = _body_insert_index(tree)
    while at < len(tree.body) and isinstance(tree.body[at], (ast.Import, ast.ImportFrom)):
        at += 1
    tree.body.insert(at, ast.ImportFrom(module=module, names=[ast.alias(name=name)], level=0))


def _broad_handlers(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.ExceptHandler]:
    out: list[ast.ExceptHandler] = []
    for node in ast.walk(fn):
        if isinstance(node, ast.ExceptHandler):
            if node.type is None:
                out.append(node)
            elif isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"}:
                out.append(node)
    return out


def _op_exception_scope(ctx: _Context) -> list[PatchCandidate]:
    exceptions = _workspace_exceptions(ctx)
    if not exceptions:
        return []
    index = _target_index(ctx)
    out: list[PatchCandidate] = []
    for rel, source in ctx.files.items():
        if _is_test_file(rel, "tests"):
            continue
        tree = _parse(source)
        if tree is None:
            continue
        for cls, module in sorted(exceptions.items()):
            trial = copy.deepcopy(tree)
            touched: list[str] = []
            for fn in _functions(trial):
                if not _is_target_function(fn, rel, index):
                    continue
                handlers = _broad_handlers(fn)
                if not handlers:
                    continue
                for handler in handlers:
                    handler.type = ast.Name(id=cls, ctx=ast.Load())
                touched.append(fn.name)
            if not touched:
                continue
            if module and module != _module_name(rel):
                _ensure_import(trial, module, cls)
            out.append(_candidate(
                ctx, "exception_scope", rel, _unparse(trial), ",".join(touched),
                f"narrow broad handler(s) in {', '.join(sorted(set(touched)))} to {cls}",
            ))
    return out


# --- 3. cache key ---------------------------------------------------------- #

def _op_cache_key(ctx: _Context) -> list[PatchCandidate]:
    index = _target_index(ctx)
    out: list[PatchCandidate] = []
    for rel, source in ctx.files.items():
        if _is_test_file(rel, "tests"):
            continue
        tree = _parse(source)
        if tree is None:
            continue
        touched: list[str] = []
        for fn in _functions(tree):
            if not _is_target_function(fn, rel, index):
                continue
            params = _positional_params(fn)
            if len(params) < 2:
                continue
            changed = False
            for node in ast.walk(fn):
                if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                    continue
                if not isinstance(node.targets[0], ast.Name):
                    continue
                value = node.value
                used: set[str] = set()
                if isinstance(value, ast.Name) and value.id in params:
                    used = {value.id}
                elif isinstance(value, ast.Tuple) and all(isinstance(e, ast.Name) for e in value.elts):
                    used = {e.id for e in value.elts if isinstance(e, ast.Name)}
                else:
                    continue
                missing = [p for p in params if p not in used]
                if not missing:
                    continue
                if isinstance(value, ast.Tuple):
                    value.elts.extend(ast.Name(id=p, ctx=ast.Load()) for p in missing)
                else:
                    node.value = ast.Tuple(
                        elts=[value, *[ast.Name(id=p, ctx=ast.Load()) for p in missing]],
                        ctx=ast.Load(),
                    )
                changed = True
            if changed:
                touched.append(fn.name)
        if touched:
            out.append(_candidate(
                ctx, "cache_key", rel, _unparse(tree), ",".join(touched),
                f"include every parameter in the cache key of {', '.join(sorted(set(touched)))}",
            ))
    return out


# --- 4. closure late binding ----------------------------------------------- #

def _target_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        out: set[str] = set()
        for elt in node.elts:
            out |= _target_names(elt)
        return out
    return set()


def _op_closure_binding(ctx: _Context) -> list[PatchCandidate]:
    index = _target_index(ctx)
    out: list[PatchCandidate] = []
    for rel, source in ctx.files.items():
        if _is_test_file(rel, "tests"):
            continue
        tree = _parse(source)
        if tree is None:
            continue
        touched: list[str] = []
        for fn in _functions(tree):
            if not _is_target_function(fn, rel, index):
                continue
            changed = False
            for loop in (n for n in ast.walk(fn) if isinstance(n, ast.For)):
                loop_vars = _target_names(loop.target)
                if not loop_vars:
                    continue
                for lam in (n for n in ast.walk(loop) if isinstance(n, ast.Lambda)):
                    loaded = {n.id for n in ast.walk(lam.body)
                              if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
                    captured = sorted(loop_vars & loaded)
                    existing = {a.arg for a in [*lam.args.posonlyargs, *lam.args.args, *lam.args.kwonlyargs]}
                    for name in captured:
                        if name in existing:
                            continue
                        lam.args.kwonlyargs.append(ast.arg(arg=name))
                        lam.args.kw_defaults.append(ast.Name(id=name, ctx=ast.Load()))
                        changed = True
            if changed:
                touched.append(fn.name)
        if touched:
            out.append(_candidate(
                ctx, "closure_binding", rel, _unparse(tree), ",".join(touched),
                f"bind loop variables by default in closures of {', '.join(sorted(set(touched)))}",
            ))
    return out


# --- helpers for function lookup by name ----------------------------------- #

def _find_function(tree: ast.Module, rel: str, name: str, files: set[str] | None = None
                   ) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    if files and rel not in files:
        return None
    for fn in _functions(tree):
        if fn.name == name:
            return fn
    return None


def _resolve_target(ctx: _Context, function: str, module: str | None) -> tuple[str, str, ast.FunctionDef | ast.AsyncFunctionDef] | None:
    """Find the first repo function named ``function`` (optionally in ``module``)."""
    ordered = sorted(ctx.targets, key=lambda t: (-t.score, t.depth))
    for target in ordered:
        if target.symbol.name != function:
            continue
        rel = target.symbol.file
        if module and not (rel.startswith(f"{module}.py") or rel.startswith(f"{module}/")):
            continue
        tree = _parse(ctx.files.get(rel, ""))
        if tree is None:
            continue
        fn = _find_function(tree, rel, function)
        if fn is not None:
            return rel, function, fn
    return None


# --- 5. boundary guard from pytest.raises ---------------------------------- #

def _has_raising_guard(fn: ast.FunctionDef | ast.AsyncFunctionDef, param: str) -> bool:
    for stmt in fn.body[:_body_insert_index(fn) + 3]:
        if not isinstance(stmt, ast.If):
            continue
        names = {n.id for n in ast.walk(stmt.test) if isinstance(n, ast.Name)}
        raises = any(isinstance(n, ast.Raise) for n in ast.walk(stmt))
        if param in names and raises:
            return True
    return False


def _op_boundary_guard(ctx: _Context) -> list[PatchCandidate]:
    # Split argument values into ordinary calls and calls the tests expect to raise.
    # A value is only "invalid" if it never appears in an ordinary call.
    normal: dict[tuple[str, str | None], dict[int, set[int]]] = {}
    raising: dict[tuple[str, str | None], dict[int, dict[int, str]]] = {}
    for sites in ctx.calls.values():
        for site in sites:
            key = (site.function, site.module)
            for pos, value in enumerate(site.args):
                if value is None:
                    continue
                if site.raises:
                    raising.setdefault(key, {}).setdefault(pos, {}).setdefault(value, site.raises)
                else:
                    normal.setdefault(key, {}).setdefault(pos, set()).add(value)

    index = _target_index(ctx)
    out: list[PatchCandidate] = []
    for rel, source in ctx.files.items():
        if _is_test_file(rel, "tests"):
            continue
        tree = _parse(source)
        if tree is None:
            continue
        touched: list[str] = []
        for (function, _module), positions in raising.items():
            if function not in index:
                continue
            if index[function] and rel not in index[function]:
                continue
            found = _find_function(tree, rel, function)
            if found is None:
                continue
            params = _positional_params(found)
            guards: list[ast.stmt] = []
            for pos, values in positions.items():
                if pos >= len(params):
                    continue
                param = params[pos]
                if _has_raising_guard(found, param):
                    continue
                ordinary = normal.get((function, _module), {}).get(pos, set())
                invalid_values = [v for v in sorted(values) if v not in ordinary]
                if not invalid_values:
                    continue
                value = invalid_values[0]
                if value < 0:
                    op, bound = ast.Lt(), 0
                    clause = f">= {bound}"
                elif value == 0 and ordinary and min(ordinary) > 0:
                    op, bound = ast.Lt(), 1
                    clause = f">= {bound}"
                elif value > 0 and ordinary and max(ordinary) < value:
                    bound = max(ordinary)
                    op = ast.Gt()
                    clause = f"<= {bound}"
                else:
                    continue
                guard = ast.If(
                    test=ast.Compare(left=ast.Name(id=param, ctx=ast.Load()), ops=[op],
                                     comparators=[ast.Constant(value=bound)]),
                    body=[ast.Raise(exc=ast.Call(
                        func=ast.Name(id=values[value], ctx=ast.Load()),
                        args=[ast.Constant(value=f"{param} must be {clause}")], keywords=[]), cause=None)],
                    orelse=[],
                )
                guards.append(guard)
            if guards:
                at = _body_insert_index(found)
                found.body[at:at] = guards
                touched.append(found.name)
        if touched:
            out.append(_candidate(
                ctx, "boundary_guard", rel, _unparse(tree), ",".join(touched),
                f"raise on boundary values rejected by the tests in {', '.join(sorted(set(touched)))}",
            ))
    return out


# --- 6. off-by-one index arithmetic ---------------------------------------- #

def _name_uses(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _op_off_by_one(ctx: _Context) -> list[PatchCandidate]:
    index = _target_index(ctx)
    out: list[PatchCandidate] = []
    for rel, source in ctx.files.items():
        if _is_test_file(rel, "tests"):
            continue
        for fn_name, files in index.items():
            if files and rel not in files:
                continue
            for side in (0, 1):
                for delta in (-1, 1):
                    tree = _parse(source)
                    if tree is None:
                        continue
                    fn = _find_function(tree, rel, fn_name)
                    if fn is None:
                        continue
                    slice_vars: set[str] = set()
                    for node in ast.walk(fn):
                        if isinstance(node, ast.Subscript):
                            for name in _name_uses(node.slice):
                                slice_vars.add(name)
                    changed = False
                    for node in ast.walk(fn):
                        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                            continue
                        if not isinstance(node.targets[0], ast.Name):
                            continue
                        if node.targets[0].id not in slice_vars:
                            continue
                        value = node.value
                        if not (isinstance(value, ast.BinOp) and isinstance(value.op, ast.Mult)):
                            continue
                        operand = value.left if side == 0 else value.right
                        delta_node = ast.BinOp(left=copy.deepcopy(operand),
                                               op=ast.Add() if delta > 0 else ast.Sub(),
                                               right=ast.Constant(value=1))
                        if side == 0:
                            value.left = delta_node
                        else:
                            value.right = delta_node
                        changed = True
                    if changed:
                        sign = "+1" if delta > 0 else "-1"
                        out.append(_candidate(
                            ctx, "off_by_one", rel, _unparse(tree), fn_name,
                            f"shift index arithmetic in {fn_name} by {sign} (operand {side})",
                        ))
    return out


# --- 7. integer remainder distribution ------------------------------------- #

def _op_integer_distribution(ctx: _Context) -> list[PatchCandidate]:
    index = _target_index(ctx)
    out: list[PatchCandidate] = []
    for rel, source in ctx.files.items():
        if _is_test_file(rel, "tests"):
            continue
        tree = _parse(source)
        if tree is None:
            continue
        changed: list[str] = []
        for fn in _functions(tree):
            if not _is_target_function(fn, rel, index):
                continue
            bases: dict[str, ast.BinOp] = {}
            for node in ast.walk(fn):
                if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                        and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.BinOp) \
                        and isinstance(node.value.op, ast.FloorDiv):
                    bases[node.targets[0].id] = node.value
            if not bases:
                continue
            for node in ast.walk(fn):
                if not isinstance(node, ast.Return) or node.value is None:
                    continue
                value = node.value
                if not (isinstance(value, ast.BinOp) and isinstance(value.op, ast.Mult)):
                    continue
                list_side = value.left if isinstance(value.left, ast.List) else value.right
                other = value.right if list_side is value.left else value.left
                if not (isinstance(list_side, ast.List) and len(list_side.elts) == 1
                        and isinstance(list_side.elts[0], ast.Name)):
                    continue
                base_name = list_side.elts[0].id
                if base_name not in bases:
                    continue
                div = bases[base_name]
                used = _name_uses(fn)
                q_name = next(n for n in ("q", "_q", "__q") if n not in used)
                r_name = next(n for n in ("r", "_r", "__r") if n not in used)
                i_name = next(n for n in ("i", "_i", "__i") if n not in used)
                assign = ast.Assign(
                    targets=[ast.Tuple(elts=[ast.Name(id=q_name, ctx=ast.Store()),
                                             ast.Name(id=r_name, ctx=ast.Store())], ctx=ast.Store())],
                    value=ast.Call(func=ast.Name(id="divmod", ctx=ast.Load()),
                                   args=[copy.deepcopy(div.left), copy.deepcopy(div.right)],
                                   keywords=[]),
                )
                ret = ast.Return(value=ast.ListComp(
                    elt=ast.IfExp(
                        test=ast.Compare(left=ast.Name(id=i_name, ctx=ast.Load()), ops=[ast.Lt()],
                                         comparators=[ast.Name(id=r_name, ctx=ast.Load())]),
                        body=ast.BinOp(left=ast.Name(id=q_name, ctx=ast.Load()), op=ast.Add(),
                                       right=ast.Constant(value=1)),
                        orelse=ast.Name(id=q_name, ctx=ast.Load()),
                    ),
                    generators=[ast.comprehension(
                        target=ast.Name(id=i_name, ctx=ast.Store()),
                        iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()),
                                      args=[copy.deepcopy(div.right)], keywords=[]),
                        ifs=[], is_async=0)],
                ))
                if _replace_statement(tree, node, [assign, ret]):
                    changed.append(fn.name)
                    break
        if changed:
            out.append(_candidate(
                ctx, "integer_distribution", rel, _unparse(tree), ",".join(changed),
                f"distribute the integer remainder in {', '.join(sorted(set(changed)))}",
            ))
    return out


# --- 8. mapping fallback --------------------------------------------------- #

class _FallbackTransformer(ast.NodeTransformer):
    def __init__(self, dicts: dict[str, ast.Dict], targets: set[str]):
        self.dicts = dicts
        self.targets = targets
        self._stack: list[str] = []
        self.changed = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if not self._stack or self._stack[-1] not in self.targets:
            return node
        if not isinstance(node.value, ast.Name) or node.value.id not in self.dicts:
            return node
        literal = self.dicts[node.value.id]
        if not literal.values:
            return node
        self.changed = True
        return ast.Call(
            func=ast.Attribute(value=node.value, attr="get", ctx=ast.Load()),
            args=[node.slice, copy.deepcopy(literal.values[0])],
            keywords=[],
        )


def _op_mapping_fallback(ctx: _Context) -> list[PatchCandidate]:
    index = _target_index(ctx)
    out: list[PatchCandidate] = []
    for rel, source in ctx.files.items():
        if _is_test_file(rel, "tests"):
            continue
        tree = _parse(source)
        if tree is None:
            continue
        dicts: dict[str, ast.Dict] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Dict):
                if node.value.keys and all(key is not None for key in node.value.keys):
                    dicts[node.targets[0].id] = node.value
        if not dicts:
            continue
        targets = {name for name, files in index.items() if not files or rel in files}
        transformer = _FallbackTransformer(dicts, targets)
        tree = transformer.visit(tree)
        if transformer.changed:
            out.append(_candidate(
                ctx, "mapping_fallback", rel, _unparse(tree), ",".join(sorted(targets)),
                f"fall back to the declared default for missing keys in {', '.join(sorted(dicts))}",
            ))
    return out


_COMPARISON_FLIPS: dict[type[ast.cmpop], type[ast.cmpop]] = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
}


def _op_comparison_flip(ctx: _Context) -> list[PatchCandidate]:
    """Flip one control-flow comparison at a time (N1 plan: comparison family)."""
    index = _target_index(ctx)
    out: list[PatchCandidate] = []
    for rel, source in ctx.files.items():
        if _is_test_file(rel, "tests"):
            continue
        for fn_name, files in index.items():
            if files and rel not in files:
                continue
            base = _parse(source)
            if base is None:
                continue
            fn = _find_function(base, rel, fn_name)
            if fn is None:
                continue
            sites: list[tuple[int, int, int]] = []
            for node in ast.walk(fn):
                if not isinstance(node, (ast.If, ast.While)):
                    continue
                for cmp_node in (n for n in ast.walk(node.test) if isinstance(n, ast.Compare)):
                    for position, op in enumerate(cmp_node.ops):
                        if type(op) in _COMPARISON_FLIPS:
                            key = (cmp_node.lineno, cmp_node.col_offset, position)
                            if key not in sites:
                                sites.append(key)
            for lineno, col, position in sites[:8]:
                tree = _parse(source)
                if tree is None:
                    continue
                target_fn = _find_function(tree, rel, fn_name)
                if target_fn is None:
                    continue
                flipped = False
                for node in ast.walk(target_fn):
                    if not isinstance(node, (ast.If, ast.While)):
                        continue
                    for cmp_node in (n for n in ast.walk(node.test) if isinstance(n, ast.Compare)):
                        if cmp_node.lineno == lineno and cmp_node.col_offset == col \
                                and position < len(cmp_node.ops):
                            replacement = _COMPARISON_FLIPS.get(type(cmp_node.ops[position]))
                            if replacement is not None:
                                cmp_node.ops[position] = replacement()
                                flipped = True
                            break
                    if flipped:
                        break
                if flipped:
                    out.append(_candidate(
                        ctx, "comparison_flip", rel, _unparse(tree), fn_name,
                        f"flip comparison {position} at line {lineno} in {fn_name}",
                    ))
    return out


_OPERATOR_FUNCS = {
    "mutable_default": _op_mutable_default,
    "exception_scope": _op_exception_scope,
    "cache_key": _op_cache_key,
    "closure_binding": _op_closure_binding,
    "boundary_guard": _op_boundary_guard,
    "off_by_one": _op_off_by_one,
    "integer_distribution": _op_integer_distribution,
    "mapping_fallback": _op_mapping_fallback,
    "comparison_flip": _op_comparison_flip,
}


def generate_candidates(ctx: _Context, limit: int = 80) -> list[PatchCandidate]:
    """Generate every candidate patch the operator set can justify, best first."""
    candidates: list[PatchCandidate] = []
    for name in OPERATORS:
        try:
            candidates.extend(_OPERATOR_FUNCS[name](ctx))
        except (SyntaxError, ValueError, StopIteration, IndexError):
            continue
    seen: set[tuple[str, str]] = set()
    unique: list[PatchCandidate] = []
    for cand in candidates:
        if cand.source == ctx.files.get(cand.file):
            continue
        digest = hashlib.sha256(cand.source.encode("utf-8")).hexdigest()
        key = (cand.file, digest)
        if key in seen:
            continue
        seen.add(key)
        unique.append(cand)
    rank = {t.symbol.name: i for i, t in enumerate(ctx.targets)}
    unique.sort(key=lambda c: (c.priority, min((rank.get(name, 999) for name in c.target.split(",")), default=999),
                               c.description))
    return unique[:limit]


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #

@dataclass
class RepairReport:
    solved: bool
    root: str
    initial_failed: int
    final_failed: int
    verifier_calls: int
    elapsed_s: float
    edits: list[dict[str, Any]] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    diffs: list[str] = field(default_factory=list)
    targets: list[dict[str, Any]] = field(default_factory=list)
    budget_exhausted: bool = False
    applied: bool = False

    @property
    def accepted_without_execution(self) -> int:
        """Count edits that have no matching executed attempt (must always be 0).

        This is a real invariant check, not a restatement of ``solved``: every
        edit kept in the final workspace must correspond to a candidate that was
        written to the workspace and executed by ``pytest`` at least once.
        """
        executed = {(a.get("file"), a.get("description")) for a in self.attempts}
        return sum(1 for edit in self.edits
                   if (edit.get("file"), edit.get("description")) not in executed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "solved": self.solved,
            "root": self.root,
            "initial_failed": self.initial_failed,
            "final_failed": self.final_failed,
            "verifier_calls": self.verifier_calls,
            "elapsed_s": self.elapsed_s,
            "edits": self.edits,
            "attempts": self.attempts,
            "diffs": self.diffs,
            "targets": self.targets,
            "budget_exhausted": self.budget_exhausted,
            "applied": self.applied,
            "accepted_without_execution": self.accepted_without_execution,
        }


@dataclass
class _Node:
    files: dict[str, str]
    failed: int
    edits: list[dict[str, Any]]
    applied: frozenset[tuple[str, str, str]]


def _edit_record(cand: PatchCandidate) -> dict[str, Any]:
    return {
        "operator": cand.operator,
        "file": cand.file,
        "target": cand.target,
        "description": cand.description,
        "verified": True,
    }


def _iter_repo_files(root: Path, tests_dir: str) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if any(part in _IGNORED_DIRS for part in Path(rel).parts):
            continue
        if _is_test_file(rel, tests_dir):
            continue
        try:
            files[rel] = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    return files


def _copy_workspace(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(*_IGNORED_DIRS, ".sealed", "*.pyc"))


def _unified_diffs(original: dict[str, str], final: dict[str, str]) -> list[str]:
    out: list[str] = []
    for rel in sorted(set(original) | set(final)):
        before = original.get(rel, "")
        after = final.get(rel, "")
        if before == after:
            continue
        diff = difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                    fromfile=f"a/{rel}", tofile=f"b/{rel}")
        out.append("".join(diff))
    return out


class RepairEngine:
    """Bounded best-first search over AST patch candidates, verified by execution."""

    def __init__(
        self,
        *,
        tests_dir: str = "tests",
        timeout_s: float = 60.0,
        max_verifier_calls: int = 80,
        max_depth: int = 3,
        max_candidates: int = 80,
        confirm: bool = True,
    ):
        self.tests_dir = tests_dir
        self.timeout_s = timeout_s
        self.max_verifier_calls = max_verifier_calls
        self.max_depth = max_depth
        self.max_candidates = max_candidates
        self.confirm = confirm

    def repair(self, root: str | Path, *, apply: bool = True) -> RepairReport:
        root = Path(root).resolve()
        started = time.perf_counter()
        initial = run_pytest(root, self.tests_dir, self.timeout_s)
        calls = 1

        targets: list[RepairTarget] = []
        calls_by_test: dict[str, list[CallSite]] = {}
        if not initial.passed and initial.failures:
            graph = index_repository(root)
            targets, calls_by_test = analyze_failures(root, graph, initial, self.tests_dir)

        repo_files = _iter_repo_files(root, self.tests_dir)
        work = Path(tempfile.mkdtemp(prefix="omega_repair_"))
        report = RepairReport(
            solved=initial.passed,
            root=str(root),
            initial_failed=initial.failed,
            final_failed=initial.failed,
            verifier_calls=calls,
            elapsed_s=0.0,
            targets=[t.to_dict() for t in targets],
        )
        try:
            _copy_workspace(root, work)
            best = _Node(files=repo_files, failed=initial.failed, edits=[], applied=frozenset())
            solved = initial.passed

            while not solved and len(best.edits) < self.max_depth:
                ctx = _Context(workspace=work, files=best.files, targets=targets,
                               calls=calls_by_test)
                candidates = generate_candidates(ctx, limit=self.max_candidates)
                if not candidates:
                    break
                improved = False
                for cand in candidates:
                    key = (cand.operator, cand.file, cand.target)
                    if key in best.applied:
                        continue
                    if calls >= self.max_verifier_calls:
                        report.budget_exhausted = True
                        break
                    target_path = work / cand.file
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    target_path.write_text(cand.source, encoding="utf-8")
                    run = run_pytest(work, self.tests_dir, self.timeout_s)
                    calls += 1
                    record = {
                        "operator": cand.operator,
                        "file": cand.file,
                        "target": cand.target,
                        "description": cand.description,
                        "failed": run.failed,
                        "passed": run.passed,
                        "elapsed_s": run.elapsed_s,
                    }
                    report.attempts.append(record)

                    if not run.passed and run.failed >= best.failed:
                        target_path.write_text(best.files[cand.file], encoding="utf-8")
                        continue

                    if run.passed:
                        confirmed = True
                        if self.confirm:
                            confirm = run_pytest(work, self.tests_dir, self.timeout_s)
                            calls += 1
                            confirmed = confirm.passed
                            record["confirmed"] = confirmed
                        if confirmed:
                            best = _Node(files={**best.files, cand.file: cand.source}, failed=0,
                                         edits=[*best.edits, _edit_record(cand)],
                                         applied=best.applied | {key})
                            solved = True
                            break
                        target_path.write_text(best.files[cand.file], encoding="utf-8")
                        continue

                    # A strict reduction in the failing-test count is kept as the new base.
                    best = _Node(files={**best.files, cand.file: cand.source}, failed=run.failed,
                                 edits=[*best.edits, _edit_record(cand)],
                                 applied=best.applied | {key})
                    improved = True
                    break
                if not improved:
                    break

            report.solved = solved
            report.final_failed = 0 if solved else best.failed
            report.verifier_calls = calls
            report.diffs = _unified_diffs(repo_files, best.files)
            report.edits = best.edits
            if solved and apply:
                wrote = False
                for rel, source in best.files.items():
                    if repo_files.get(rel) == source:
                        continue
                    path = root / rel
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(source, encoding="utf-8")
                    wrote = True
                report.applied = wrote
        finally:
            shutil.rmtree(work, ignore_errors=True)
            report.elapsed_s = round(time.perf_counter() - started, 3)
        return report
