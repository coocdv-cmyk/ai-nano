"""Independent verification of step postconditions.

The verifier never takes a tool's word for it when the world can be checked
directly: file expectations re-read the disk and recompute hashes. A step is
done only when every expectation passes.
"""

from __future__ import annotations

import hashlib
from typing import Any

from .security import PathEscape, guard_path
from .tools.base import ToolContext


def _field(output: dict | None, dotted: str) -> Any:
    cur: Any = output or {}
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return None
    return cur


def _file(ctx: ToolContext, path: str) -> bytes | None:
    try:
        p = guard_path(ctx.workspace, path)
    except PathEscape:
        return None
    return p.read_bytes() if p.is_file() else None


def check(ctx: ToolContext, exp: dict[str, Any], output: dict | None) -> tuple[bool, str]:
    kind = exp.get("kind")
    if kind == "ok":
        ok = bool(output and output.get("ok"))
        return ok, "tool reported ok" if ok else f"tool reported failure: {(output or {}).get('error')}"
    if kind == "file_exists":
        ok = _file(ctx, exp["path"]) is not None
        return ok, f"{exp['path']} {'exists' if ok else 'missing'} on disk"
    if kind == "file_absent":
        ok = _file(ctx, exp["path"]) is None
        return ok, f"{exp['path']} {'absent' if ok else 'still present'} on disk"
    if kind == "file_sha256":
        data = _file(ctx, exp["path"])
        actual = hashlib.sha256(data).hexdigest() if data is not None else None
        ok = actual == exp["sha256"]
        return ok, f"disk sha256 {'matches' if ok else 'differs'} ({(actual or 'missing')[:12]})"
    if kind in ("file_contains", "file_endswith"):
        data = _file(ctx, exp["path"])
        text = data.decode("utf-8", errors="replace") if data is not None else None
        if text is None:
            return False, f"{exp['path']} missing on disk"
        ok = exp["text"] in text if kind == "file_contains" else text.endswith(exp["text"])
        return ok, f"{exp['path']} {'has' if ok else 'lacks'} expected text"
    if kind == "exit_code":
        v = _field(output, "exit_code")
        return v == exp.get("equals", 0), f"exit code {v}"
    if kind == "output_equals":
        v = _field(output, exp["field"])
        ok = str(v) == str(exp["value"])
        return ok, f"{exp['field']} = {str(v)[:60]!r}"
    if kind == "output_contains":
        v = _field(output, exp["field"])
        ok = v is not None and str(exp["text"]).casefold() in str(v).casefold()
        return ok, f"{exp['field']} {'contains' if ok else 'lacks'} {exp['text']!r}"
    if kind == "min_count":
        v = _field(output, exp.get("field", "count"))
        ok = isinstance(v, int) and v >= exp.get("n", 1)
        return ok, f"{exp.get('field', 'count')} = {v} (need >= {exp.get('n', 1)})"
    return False, f"unknown expectation kind {kind!r}"


def verify(ctx: ToolContext, expectations: list[dict[str, Any]], output: dict | None) -> tuple[bool, list[dict]]:
    results = []
    for exp in expectations or [{"kind": "ok"}]:
        try:
            ok, detail = check(ctx, exp, output)
        except (KeyError, TypeError) as e:
            ok, detail = False, f"malformed expectation: {e}"
        results.append({"expect": exp, "passed": ok, "detail": detail})
    return all(r["passed"] for r in results), results


WORLD_KINDS = ("file_exists", "file_absent", "file_sha256", "file_contains", "file_endswith")


def world_checks(expectations: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """The expectations to re-check after a crash, or None if the world alone can't settle the step.

    "ok" only says the tool returned success, which is lost in a crash; the world
    postconditions are the real claim. Every other expectation must be checkable
    on disk, and there must be at least one.
    """
    rest = [e for e in expectations if e.get("kind") != "ok"]
    if rest and all(e.get("kind") in WORLD_KINDS for e in rest):
        return rest
    return None
