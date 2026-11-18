"""Permission modes, scopes, and the workspace path guard.

A tool declares the scopes it needs. The task's mode decides, per scope,
whether the call is allowed outright, needs a human approval, or is denied.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path


class Scope(str, Enum):
    READ = "read"  # inspect workspace files and memory
    WRITE = "write"  # create or modify files / memories inside the workspace
    EXECUTE = "execute"  # run processes
    DESTRUCTIVE = "destructive"  # delete or overwrite irreversibly


class Decision(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


A, Q, D = Decision.ALLOW, Decision.ASK, Decision.DENY

MODES: dict[str, dict[Scope, Decision]] = {
    "read_only": {Scope.READ: A, Scope.WRITE: D, Scope.EXECUTE: D, Scope.DESTRUCTIVE: D},
    "ask": {Scope.READ: A, Scope.WRITE: Q, Scope.EXECUTE: Q, Scope.DESTRUCTIVE: Q},
    "workspace": {Scope.READ: A, Scope.WRITE: A, Scope.EXECUTE: Q, Scope.DESTRUCTIVE: Q},
    "autonomous": {Scope.READ: A, Scope.WRITE: A, Scope.EXECUTE: A, Scope.DESTRUCTIVE: Q},
}


def decide(mode: str, scopes: frozenset[Scope]) -> Decision:
    table = MODES[mode]
    worst = A
    for s in scopes:
        d = table[s]
        if d is D:
            return D
        if d is Q:
            worst = Q
    return worst


class PathEscape(PermissionError):
    pass


def guard_path(root: Path, rel: str) -> Path:
    """Resolve `rel` inside `root`, rejecting absolute paths, `..` and symlink escapes."""
    if not isinstance(rel, str) or not rel.strip():
        raise PathEscape("empty path")
    if "\x00" in rel:
        raise PathEscape("NUL byte in path")
    p = Path(rel)
    if p.is_absolute() or p.drive or rel.startswith(("/", "\\")):
        raise PathEscape(f"absolute paths are not allowed: {rel}")
    root = root.resolve()
    target = (root / p).resolve()
    if os.path.commonpath([str(root), str(target)]) != str(root):
        raise PathEscape(f"path escapes the workspace: {rel}")
    return target
