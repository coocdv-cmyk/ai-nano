import pytest

from omega_nano.security import Scope
from omega_nano.tools import Tool, ToolContext, ToolError, default_registry
from omega_nano.tools.base import Param
from omega_nano.tools.mathx import evaluate


@pytest.mark.parametrize("expr,value", [
    ("2^100", str(2**100)),
    ("1/3 + 1/6", "1/2"),
    ("sqrt(16)", "4"),
    ("sqrt(9/4)", "3/2"),
    ("factorial(25)", "15511210043330985984000000"),
    ("١٢ × ٣", "36"),
    ("(7 - 10) // 2", "-2"),
    ("0.1 + 0.2", "3/10"),  # exact decimal arithmetic, no float drift
    ("gcd(84, 36)", "12"),
])
def test_exact_math(expr, value):
    assert evaluate(expr)["value"] == value


@pytest.mark.parametrize("expr", ["9**9**9", "1/0", "__import__('os')", "open('x')", "a.b", "[1,2]", "2 +"])
def test_math_rejects_unsafe_or_unbounded(expr):
    with pytest.raises(ToolError):
        evaluate(expr)


def ctx(tmp_path):
    return ToolContext(workspace=tmp_path / "ws", home=tmp_path)


def test_fs_roundtrip_search_list_delete(tmp_path):
    reg = default_registry()
    c = ctx(tmp_path)
    c.workspace.mkdir()
    w = reg.get("fs.write").run(c, reg.get("fs.write").validate({"path": "d/a.txt", "content": "alpha\nBeta"}))
    assert w["ok"] and w["bytes"] == 10
    r = reg.get("fs.read").run(c, reg.get("fs.read").validate({"path": "d/a.txt"}))
    assert r["content"] == "alpha\nBeta" and r["sha256"] == w["sha256"]
    s = reg.get("fs.search").run(c, reg.get("fs.search").validate({"text": "beta"}))
    assert s["count"] == 1 and s["hits"][0]["line"] == 2
    lst = reg.get("fs.list").run(c, reg.get("fs.list").validate({}))
    assert {e["path"] for e in lst["entries"]} == {"d", "d/a.txt"}
    d = reg.get("fs.delete").run(c, reg.get("fs.delete").validate({"path": "d/a.txt"}))
    assert d["ok"] and not (c.workspace / "d" / "a.txt").exists()
    assert (tmp_path / d["trashed_to"]).read_text(encoding="utf-8") == "alpha\nBeta"  # recoverable


def test_validation_errors():
    w = default_registry().get("fs.write")
    with pytest.raises(ToolError, match="missing"):
        w.validate({"path": "x"})
    with pytest.raises(ToolError, match="unknown"):
        w.validate({"path": "x", "content": "y", "evil": 1})
    with pytest.raises(ToolError, match="must be str"):
        w.validate({"path": 3, "content": "y"})


def test_malformed_tool_response_is_an_error(tmp_path):
    bad = Tool("bad", "returns garbage", {}, frozenset({Scope.READ}), fn=lambda ctx: "not a dict")
    with pytest.raises(ToolError, match="malformed"):
        bad.run(ctx(tmp_path), {})


def test_registry_rejects_duplicates():
    reg = default_registry()
    with pytest.raises(ValueError):
        reg.register(reg.get("fs.read"))


def test_param_int_rejects_bool():
    t = Tool("t", "", {"n": Param("int")}, frozenset(), fn=lambda ctx, n: {"ok": True})
    with pytest.raises(ToolError):
        t.validate({"n": True})
