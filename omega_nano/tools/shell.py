"""Bounded process execution inside the workspace.

The program must be on the configured allowlist, dangerous patterns are
refused outright, there is a hard timeout, and output is capped. Running a
process is never idempotent, so a crash mid-run is escalated, not retried.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys

from ..security import Scope
from .base import Param, Tool, ToolContext, ToolError

MAX_OUT = 64 * 1024
DENY = re.compile(
    r"(\brm\s+-[a-z]*r|\bdel\s+/[sq]|\brmdir\s+/s|\bformat\b|\bshutdown\b|\bmkfs\b|\breg\s+(add|delete)\b"
    r"|\bcurl\b|\bwget\b|invoke-webrequest|\bnetsh\b|\bsc\s+(stop|delete)\b|>\s*/dev/sd|:\(\)\s*\{)",
    re.IGNORECASE,
)
WIN_BUILTINS = {"dir", "type", "echo", "ver"}
META = ("|", "&", ";", "`", "$(", ">", "<")
_QUOTED = re.compile(r"\"[^\"]*\"|'[^']*'")


def _unquote(arg: str) -> str:
    if len(arg) >= 2 and arg[0] == arg[-1] and arg[0] in "\"'":
        return arg[1:-1]
    return arg


def _run(ctx: ToolContext, cmd: str, timeout_s: float) -> dict:
    if DENY.search(cmd):
        raise ToolError("shell.run: command matches a denied pattern")
    # No shell is involved (argv exec), so metacharacters inside quoted arguments are inert;
    # outside quotes they signal an attempt at pipes or chaining and are refused.
    if any(ch in _QUOTED.sub("", cmd) for ch in META):
        raise ToolError("shell.run: pipes, chaining and substitution are not allowed")
    try:
        argv = [_unquote(a) for a in shlex.split(cmd, posix=os.name != "nt")]
    except ValueError as e:
        raise ToolError(f"shell.run: cannot parse command: {e}") from None
    if not argv:
        raise ToolError("shell.run: empty command")
    prog = os.path.basename(argv[0]).lower()
    prog = prog[:-4] if prog.endswith(".exe") else prog
    allow = getattr(ctx.config, "shell_allow", ())
    if prog not in allow:
        raise ToolError(f"shell.run: '{prog}' is not on the allowlist {sorted(allow)}")
    if prog in ("python", "py"):
        argv[0] = sys.executable
    if os.name == "nt" and prog in WIN_BUILTINS:
        if any(ch in cmd for ch in META + ("%", "^", "<", ">")):
            raise ToolError("shell.run: metacharacters are not allowed in cmd built-ins")
        argv = ["cmd", "/c", *argv]
    timeout_s = max(0.1, min(float(timeout_s), 600.0))
    try:
        p = subprocess.run(
            argv, cwd=str(ctx.workspace), capture_output=True, timeout=timeout_s,
            stdin=subprocess.DEVNULL, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {timeout_s}s", "exit_code": None, "timed_out": True}
    except FileNotFoundError:
        return {"ok": False, "error": f"program not found: {argv[0]}", "exit_code": None}
    out = p.stdout[:MAX_OUT].decode("utf-8", errors="replace")
    err = p.stderr[:MAX_OUT].decode("utf-8", errors="replace")
    return {"ok": p.returncode == 0, "exit_code": p.returncode, "stdout": out, "stderr": err,
            "trust": "untrusted_data"}


TOOLS = [
    Tool(
        name="shell.run",
        description="Run one allowlisted program in the workspace with a timeout (no pipes or chaining).",
        params={"cmd": Param("str"), "timeout_s": Param("float", required=False, default=20.0)},
        scopes=frozenset({Scope.EXECUTE}), fn=_run, idempotent=False,
    )
]
