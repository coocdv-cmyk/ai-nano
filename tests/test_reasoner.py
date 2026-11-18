import pytest

from omega_nano.reasoner import RuleReasoner
from omega_nano.tools import default_registry

R = RuleReasoner()
TOOLS = default_registry()


def tools_for(goal):
    und, steps = R.plan(goal, TOOLS)
    return und, [s.tool for s in steps], steps


@pytest.mark.parametrize("goal,expected", [
    ("calculate 2^64", ["math.eval"]),
    ("احسب 5*5", ["math.eval"]),
    ("write file a.txt with hello", ["fs.write"]),
    ("اكتب ملف a.txt فيه مرحبا", ["fs.write"]),
    ("read file a.txt", ["fs.read"]),
    ("اقرأ الملف a.txt", ["fs.read"]),
    ("list files", ["fs.list"]),
    ("اعرض الملفات", ["fs.list"]),
    ("search for TODO", ["fs.search"]),
    ("ابحث عن مرحبا", ["fs.search"]),
    ("delete file a.txt", ["fs.delete"]),
    ("احذف الملف a.txt", ["fs.delete"]),
    ("remember that Sara likes tea", ["memory.remember"]),
    ("تذكر ان سارة تحب الشاي", ["memory.remember"]),
    ("what do you know about Sara", ["memory.recall"]),
    ("ماذا تعرف عن سارة", ["memory.recall"]),
    ("what is the capital of France", ["memory.recall"]),  # no digits -> recall, not math
    ("run python --version", ["shell.run"]),
    ("شغل python --version", ["shell.run"]),
    ("calculate 3*7 then save the result to r.txt then read file r.txt", ["math.eval", "fs.write", "fs.read"]),
    ("احسب ٣*٧ ثم احفظ النتيجة في r.txt", ["math.eval", "fs.write"]),
    ("append hi to file log.txt", ["fs.write"]),
    ("check that file a.txt contains hello", ["fs.read"]),
])
def test_intents(goal, expected):
    und, tools, _ = tools_for(goal)
    assert tools == expected and und["confidence"] == 1.0, und


def test_unparsed_is_reported_honestly():
    und, tools, _ = tools_for("calculate 1+1 then book me a flight to Tokyo")
    assert tools == ["math.eval"]
    assert und["unparsed"] == ["book me a flight to Tokyo"] and und["confidence"] == 0.5


def test_save_result_references_previous_output():
    _, _, steps = tools_for("calculate 2+2 then save the result to out.txt")
    assert steps[1].args["content"] == "${prev.value}"
    assert {"kind": "file_sha256", "path": "out.txt", "sha256": "@content_sha256"} in steps[1].expect


def test_save_result_without_previous_step_is_not_guessed():
    und, tools, _ = tools_for("save the result to out.txt")
    assert tools == [] and und["unparsed"]


def test_memory_key_extraction():
    _, _, steps = tools_for("remember that my favourite colour is blue")
    assert steps[0].args["key"] == "favourite colour"
    _, _, steps = tools_for("تذكر ان عاصمة مصر هي القاهرة")
    assert steps[0].args["key"] == "عاصمة مصر"


def test_language_detection():
    assert tools_for("احسب 1+1")[0]["language"] == "ar"
    assert tools_for("calculate 1+1")[0]["language"] == "en"
