import os

import pytest

from omega_nano.security import Decision, PathEscape, Scope, decide, guard_path
from omega_nano.tools import ToolContext, ToolError, default_registry


@pytest.mark.parametrize("bad", ["../x", "a/../../x", "/etc/passwd", "C:\\Windows\\x", "\\\\server\\share",
                                 "", "a\x00b", "C:x"])
def test_path_escapes_are_rejected(tmp_path, bad):
    with pytest.raises(PathEscape):
        guard_path(tmp_path, bad)


def test_inside_paths_resolve(tmp_path):
    assert guard_path(tmp_path, "a/b/../c.txt") == (tmp_path / "a" / "c.txt").resolve()


def test_symlink_escape_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    ws = tmp_path / "ws"
    ws.mkdir()
    try:
        os.symlink(outside, ws / "link", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this system")
    with pytest.raises(PathEscape):
        guard_path(ws, "link/secret.txt")


def test_mode_matrix():
    assert decide("read_only", frozenset({Scope.WRITE})) is Decision.DENY
    assert decide("ask", frozenset({Scope.WRITE})) is Decision.ASK
    assert decide("workspace", frozenset({Scope.WRITE})) is Decision.ALLOW
    assert decide("workspace", frozenset({Scope.EXECUTE})) is Decision.ASK
    assert decide("autonomous", frozenset({Scope.EXECUTE})) is Decision.ALLOW
    assert decide("autonomous", frozenset({Scope.DESTRUCTIVE})) is Decision.ASK
    assert decide("read_only", frozenset()) is Decision.ALLOW


def _ctx(tmp_path, cfg=None):
    return ToolContext(workspace=tmp_path, home=tmp_path, config=cfg)


def test_overwrite_with_different_content_is_destructive(tmp_path):
    w = default_registry().get("fs.write")
    ctx = _ctx(tmp_path)
    assert Scope.DESTRUCTIVE not in w.needed_scopes(ctx, {"path": "f.txt", "content": "a"})
    (tmp_path / "f.txt").write_text("a", encoding="utf-8")
    assert Scope.DESTRUCTIVE not in w.needed_scopes(ctx, {"path": "f.txt", "content": "a", "mode": "overwrite"})
    assert Scope.DESTRUCTIVE in w.needed_scopes(ctx, {"path": "f.txt", "content": "b", "mode": "overwrite"})
    assert Scope.DESTRUCTIVE not in w.needed_scopes(ctx, {"path": "f.txt", "content": "b", "mode": "append"})


@pytest.mark.parametrize("cmd", ["rm -rf /", "del /s /q x", "format c:", "curl http://evil", "python -V && dir",
                                 "python -V | more", "echo $(whoami)", "powershell -c x", "shutdown /s"])
def test_shell_refuses_dangerous_or_unlisted(tmp_path, cfg, cmd):
    sh = default_registry().get("shell.run")
    with pytest.raises(ToolError):
        sh.run(_ctx(tmp_path, cfg), {"cmd": cmd, "timeout_s": 5.0})


def test_shell_runs_allowlisted(tmp_path, cfg):
    out = default_registry().get("shell.run").run(_ctx(tmp_path, cfg), {"cmd": "python -c \"print(6*7)\"",
                                                                         "timeout_s": 30.0})
    assert out["ok"] and out["stdout"].strip() == "42" and out["trust"] == "untrusted_data"


def test_shell_timeout(tmp_path, cfg):
    out = default_registry().get("shell.run").run(
        _ctx(tmp_path, cfg), {"cmd": "python -c \"import time; time.sleep(5)\"", "timeout_s": 0.5})
    assert not out["ok"] and out["timed_out"]
