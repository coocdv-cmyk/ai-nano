"""AST-based Python repository symbol & call-graph indexer (M3).

Extracts classes, functions, methods, signatures, docstrings, inheritance,
and call edges using Python's `ast` module and optionally ingests compact
project records into ``MemoryFabric`` with ``kind="project"``.
"""
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any

DEFAULT_EXCLUDE_DIRS = frozenset({
    "__pycache__",
    ".git",
    ".hg",
    ".svn",
    ".omega",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
})


@dataclass
class SymbolRecord:
    """Single indexed symbol from a Python source file."""

    name: str
    qualname: str
    kind: str  # "class" | "function" | "method" | "async_function" | "async_method"
    file: str
    lineno: int
    end_lineno: int
    signature: str = ""
    docstring: str = ""
    calls: list[str] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "qualname": self.qualname,
            "kind": self.kind,
            "file": self.file,
            "lineno": self.lineno,
            "end_lineno": self.end_lineno,
            "signature": self.signature,
            "docstring": self.docstring,
            "calls": list(self.calls),
            "bases": list(self.bases),
        }

    def to_memory_text(self) -> str:
        """Format a compact searchable summary for MemoryFabric."""
        parts = [f"[symbol:{self.kind}] {self.qualname}{self.signature} in {self.file}:{self.lineno}-{self.end_lineno}"]
        if self.bases:
            parts.append(f"bases: {', '.join(self.bases)}")
        if self.calls:
            parts.append(f"calls: {', '.join(self.calls[:12])}")
        if self.docstring:
            first_line = self.docstring.strip().splitlines()[0][:140]
            parts.append(f"doc: {first_line}")
        return " | ".join(parts)


def _safe_unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _format_signature(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = fn.args
    pieces: list[str] = []
    n_pos_defaults = len(args.defaults)
    pos_all = list(args.posonlyargs) + list(args.args)
    default_start = len(pos_all) - n_pos_defaults

    for idx, arg in enumerate(pos_all):
        s = arg.arg
        if arg.annotation is not None:
            s += f": {_safe_unparse(arg.annotation)}"
        if idx >= default_start:
            d_val = _safe_unparse(args.defaults[idx - default_start])
            if len(d_val) > 24:
                d_val = "..."
            s += f" = {d_val}"
        pieces.append(s)
        if args.posonlyargs and idx == len(args.posonlyargs) - 1:
            pieces.append("/")

    if args.vararg is not None:
        s = f"*{args.vararg.arg}"
        if args.vararg.annotation is not None:
            s += f": {_safe_unparse(args.vararg.annotation)}"
        pieces.append(s)
    elif args.kwonlyargs:
        pieces.append("*")

    for kw, kw_def in zip(args.kwonlyargs, args.kw_defaults):
        s = kw.arg
        if kw.annotation is not None:
            s += f": {_safe_unparse(kw.annotation)}"
        if kw_def is not None:
            d_val = _safe_unparse(kw_def)
            if len(d_val) > 24:
                d_val = "..."
            s += f" = {d_val}"
        pieces.append(s)

    if args.kwarg is not None:
        s = f"**{args.kwarg.arg}"
        if args.kwarg.annotation is not None:
            s += f": {_safe_unparse(args.kwarg.annotation)}"
        pieces.append(s)

    sig = f"({', '.join(pieces)})"
    if fn.returns is not None:
        sig += f" -> {_safe_unparse(fn.returns)}"
    return sig


def _call_target_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_target_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def _collect_direct_calls(body: list[ast.stmt]) -> list[str]:
    """Collect called function/method names in `body` without descending into nested defs."""
    calls: list[str] = []
    seen: set[str] = set()
    stack: list[ast.AST] = list(body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Call):
            name = _call_target_name(node.func)
            if name and name not in seen:
                seen.add(name)
                calls.append(name)
        stack.extend(ast.iter_child_nodes(node))
    return calls


def _extract_file_symbols(path: Path, rel_path: str, module_name: str) -> list[SymbolRecord]:
    src = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src, filename=rel_path)
    records: list[SymbolRecord] = []

    def visit_block(stmts: list[ast.stmt], scope: list[str], in_class: bool) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.ClassDef):
                qual = ".".join([module_name, *scope, stmt.name]) if module_name else ".".join([*scope, stmt.name])
                bases = [b for b in (_safe_unparse(base) for base in stmt.bases) if b]
                doc = ast.get_docstring(stmt) or ""
                records.append(SymbolRecord(
                    name=stmt.name,
                    qualname=qual,
                    kind="class",
                    file=rel_path,
                    lineno=int(stmt.lineno),
                    end_lineno=int(getattr(stmt, "end_lineno", stmt.lineno) or stmt.lineno),
                    signature=f"({', '.join(bases)})" if bases else "",
                    docstring=doc,
                    calls=[],
                    bases=bases,
                ))
                visit_block(stmt.body, [*scope, stmt.name], in_class=True)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                is_async = isinstance(stmt, ast.AsyncFunctionDef)
                if in_class:
                    kind = "async_method" if is_async else "method"
                else:
                    kind = "async_function" if is_async else "function"
                qual = ".".join([module_name, *scope, stmt.name]) if module_name else ".".join([*scope, stmt.name])
                doc = ast.get_docstring(stmt) or ""
                calls = _collect_direct_calls(stmt.body)
                records.append(SymbolRecord(
                    name=stmt.name,
                    qualname=qual,
                    kind=kind,
                    file=rel_path,
                    lineno=int(stmt.lineno),
                    end_lineno=int(getattr(stmt, "end_lineno", stmt.lineno) or stmt.lineno),
                    signature=_format_signature(stmt),
                    docstring=doc,
                    calls=calls,
                    bases=[],
                ))
                visit_block(stmt.body, [*scope, stmt.name], in_class=False)

    visit_block(tree.body, [], in_class=False)
    return records


@dataclass
class RepoGraph:
    """In-memory symbol & call-edge index over a Python repository."""

    root: str
    symbols: list[SymbolRecord] = field(default_factory=list)
    files_indexed: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def find_symbol(self, query: str, limit: int = 20) -> list[SymbolRecord]:
        q = query.strip().lower()
        if not q:
            return []
        exact: list[SymbolRecord] = []
        suffix: list[SymbolRecord] = []
        partial: list[SymbolRecord] = []
        for sym in self.symbols:
            name_l = sym.name.lower()
            qual_l = sym.qualname.lower()
            if name_l == q or qual_l == q:
                exact.append(sym)
            elif qual_l.endswith(f".{q}") or name_l.startswith(q):
                suffix.append(sym)
            elif q in qual_l or q in sym.docstring.lower():
                partial.append(sym)
        return (exact + suffix + partial)[:limit]

    def callers_of(self, symbol_name: str, limit: int = 30) -> list[SymbolRecord]:
        target = symbol_name.strip().lower()
        if not target:
            return []
        out: list[SymbolRecord] = []
        for sym in self.symbols:
            for c in sym.calls:
                cl = c.lower()
                if cl == target or cl.endswith(f".{target}"):
                    out.append(sym)
                    break
            if len(out) >= limit:
                break
        return out

    def callees_of(self, symbol_name: str) -> list[str]:
        matches = self.find_symbol(symbol_name, limit=5)
        seen: set[str] = set()
        out: list[str] = []
        for sym in matches:
            for c in sym.calls:
                if c not in seen:
                    seen.add(c)
                    out.append(c)
        return out

    def summarize(self, top_k: int = 8) -> dict[str, Any]:
        kind_counts = Counter(s.kind for s in self.symbols)
        call_counts = Counter(c.split(".")[-1] for s in self.symbols for c in s.calls)
        return {
            "root": self.root,
            "files_indexed": self.files_indexed,
            "total_symbols": len(self.symbols),
            "kinds": dict(kind_counts),
            "top_called": [{"symbol": name, "call_sites": cnt} for name, cnt in call_counts.most_common(top_k)],
            "errors": len(self.errors),
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


def _iter_python_files(root: Path, exclude_dirs: frozenset[str], max_files: int) -> list[Path]:
    found: list[Path] = []
    if root.is_file():
        return [root] if root.suffix == ".py" else []

    stack = [root]
    while stack and len(found) < max_files:
        cur = stack.pop()
        try:
            entries = sorted(cur.iterdir(), key=lambda p: p.name)
        except OSError:
            continue
        subdirs: list[Path] = []
        for entry in entries:
            if entry.is_dir():
                if entry.name not in exclude_dirs and not entry.name.startswith("."):
                    subdirs.append(entry)
            elif entry.is_file() and entry.suffix == ".py":
                found.append(entry)
                if len(found) >= max_files:
                    break
        stack.extend(reversed(subdirs))
    return found


def index_repository(
    root: str | Path,
    memory: Any | None = None,
    *,
    max_files: int = 500,
    exclude_dirs: frozenset[str] | set[str] | None = None,
    source_prefix: str = "repo_graph",
) -> RepoGraph:
    """Walk `root`, extract AST symbols/calls, and optionally ingest into `memory`."""
    t0 = time.perf_counter()
    root_path = Path(root).resolve()
    excludes = frozenset(exclude_dirs) if exclude_dirs is not None else DEFAULT_EXCLUDE_DIRS

    symbols: list[SymbolRecord] = []
    errors: list[str] = []
    py_files = _iter_python_files(root_path, excludes, max_files=max_files)
    base_dir = root_path.parent if root_path.is_file() else root_path

    for py_file in py_files:
        try:
            rel = py_file.relative_to(base_dir).as_posix()
        except ValueError:
            rel = py_file.as_posix()
        mod_parts = list(Path(rel).with_suffix("").parts)
        if mod_parts and mod_parts[-1] == "__init__":
            mod_parts = mod_parts[:-1]
        module_name = ".".join(mod_parts)
        try:
            symbols.extend(_extract_file_symbols(py_file, rel, module_name))
        except Exception as exc:
            errors.append(f"{rel}: {type(exc).__name__}: {exc}")

    graph = RepoGraph(
        root=str(root_path),
        symbols=symbols,
        files_indexed=len(py_files),
        errors=errors,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )

    if memory is not None and symbols:
        _ingest_into_memory(graph, memory, source_prefix=source_prefix)

    return graph


def _ingest_into_memory(graph: RepoGraph, memory: Any, source_prefix: str = "repo_graph") -> int:
    """Persist compact project symbol records into MemoryFabric (`kind="project"`)."""
    items: list[tuple[str, str, list[str], str]] = []
    for sym in graph.symbols:
        file_stem = Path(sym.file).stem
        tags = ["repo_graph", sym.kind, sym.name.lower(), file_stem.lower()]
        items.append((sym.to_memory_text(), "project", tags, f"{source_prefix}:{sym.file}"))

    if hasattr(memory, "remember_many"):
        try:
            # Supports remember_many([(text, kind, tags, source), ...]) or dict items.
            memory.remember_many([
                {"text": t, "kind": k, "tags": tg, "source": src}
                for t, k, tg, src in items
            ])
            return len(items)
        except TypeError:
            pass
        except Exception:
            pass

    if hasattr(memory, "remember"):
        for t, k, tg, src in items:
            memory.remember(t, kind=k, tags=tg, source=src)
        return len(items)
    return 0
