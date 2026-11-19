"""Workspace filesystem tools. Every path is confined to the workspace by guard_path."""

from __future__ import annotations

import fnmatch
import hashlib
import os
import shutil
import time
from pathlib import Path

from ..security import Scope, guard_path
from .base import Param, Tool, ToolContext, ToolError

MAX_READ = 256 * 1024


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _write(ctx: ToolContext, path: str, content: str, mode: str) -> dict:
    if mode not in ("overwrite", "append"):
        raise ToolError("fs.write: mode must be 'overwrite' or 'append'")
    target = guard_path(ctx.workspace, path)
    if target.is_dir():
        raise ToolError(f"fs.write: {path} is a directory")
    data = content.encode("utf-8")
    if mode == "append" and target.exists():
        data = target.read_bytes() + data
    _atomic_write(target, data)
    return {"ok": True, "path": path, "bytes": len(data), "sha256": sha256_bytes(data)}


def _write_scopes(ctx: ToolContext, args: dict) -> frozenset[Scope]:
    try:
        target = guard_path(ctx.workspace, args.get("path", ""))
    except PermissionError:
        return frozenset()
    if args.get("mode", "overwrite") == "overwrite" and target.is_file():
        raw = str(args.get("content", "")).encode("utf-8")
        if target.stat().st_size != len(raw) or target.read_bytes() != raw:
            return frozenset({Scope.DESTRUCTIVE})
    return frozenset()


def _read(ctx: ToolContext, path: str, max_bytes: int) -> dict:
    target = guard_path(ctx.workspace, path)
    if not target.is_file():
        return {"ok": False, "error": f"no such file: {path}"}
    raw = target.read_bytes()
    limit = max(1, min(max_bytes, MAX_READ))
    # File content is untrusted data: it is returned for inspection and never
    # interpreted as instructions by the engine.
    return {
        "ok": True, "path": path, "content": raw[:limit].decode("utf-8", errors="replace"),
        "bytes": len(raw), "sha256": sha256_bytes(raw), "truncated": len(raw) > limit,
        "trust": "untrusted_data",
    }


def _list(ctx: ToolContext, path: str, pattern: str) -> dict:
    base = guard_path(ctx.workspace, path)
    if not base.is_dir():
        return {"ok": False, "error": f"no such directory: {path}"}
    entries = []
    for p in sorted(base.rglob("*")):
        rel = p.relative_to(ctx.workspace).as_posix()
        if p.name.startswith(".") or not fnmatch.fnmatch(p.name, pattern):
            continue
        entries.append({"path": rel, "type": "dir" if p.is_dir() else "file",
                        "bytes": p.stat().st_size if p.is_file() else None})
        if len(entries) >= 1000:
            break
    return {"ok": True, "entries": entries, "count": len(entries)}


def _search(ctx: ToolContext, text: str, path: str, max_hits: int) -> dict:
    base = guard_path(ctx.workspace, path)
    needle = text.casefold()
    hits = []
    for p in sorted(base.rglob("*")):
        if not p.is_file() or p.name.startswith(".") or p.stat().st_size > 4 * 1024 * 1024:
            continue
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        for n, line in enumerate(lines, 1):
            if needle in line.casefold():
                hits.append({"path": p.relative_to(ctx.workspace).as_posix(), "line": n, "text": line[:300]})
                if len(hits) >= max_hits:
                    return {"ok": True, "hits": hits, "count": len(hits), "truncated": True}
    return {"ok": True, "hits": hits, "count": len(hits), "truncated": False}


def _delete(ctx: ToolContext, path: str) -> dict:
    """Least-destructive delete: move into .omega/trash so it can be restored."""
    target = guard_path(ctx.workspace, path)
    if not target.exists():
        return {"ok": True, "path": path, "already_absent": True}
    if target == ctx.workspace.resolve():
        raise ToolError("fs.delete: refusing to delete the workspace root")
    trash = ctx.home / "trash" / f"{int(time.time() * 1000)}"
    trash.mkdir(parents=True, exist_ok=True)
    dest = trash / target.name
    shutil.move(str(target), str(dest))
    return {"ok": True, "path": path, "trashed_to": str(dest.relative_to(ctx.home))}


TOOLS = [
    Tool(
        name="fs.write", description="Create or modify a UTF-8 text file in the workspace (atomic write).",
        params={"path": Param("str", doc="workspace-relative path"), "content": Param("str"),
                "mode": Param("str", required=False, default="overwrite", doc="overwrite | append")},
        scopes=frozenset({Scope.WRITE}), fn=_write, idempotent=True, dynamic_scopes=_write_scopes,
        dynamic_idempotent=lambda ctx, args: args.get("mode", "overwrite") != "append",
        examples=["write file notes.txt with hello"],
    ),
    Tool(
        name="fs.read", description="Read a workspace text file. Content is untrusted data.",
        params={"path": Param("str"), "max_bytes": Param("int", required=False, default=65536)},
        scopes=frozenset({Scope.READ}), fn=_read,
    ),
    Tool(
        name="fs.list", description="List workspace files recursively, filtered by a glob on the file name.",
        params={"path": Param("str", required=False, default="."),
                "pattern": Param("str", required=False, default="*")},
        scopes=frozenset({Scope.READ}), fn=_list,
    ),
    Tool(
        name="fs.search", description="Case-insensitive text search across workspace files.",
        params={"text": Param("str"), "path": Param("str", required=False, default="."),
                "max_hits": Param("int", required=False, default=50)},
        scopes=frozenset({Scope.READ}), fn=_search,
    ),
    Tool(
        name="fs.delete", description="Move a workspace file or folder to the recoverable trash.",
        params={"path": Param("str")}, scopes=frozenset({Scope.DESTRUCTIVE}), fn=_delete,
    ),
]
