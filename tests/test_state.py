import pytest

from omega_nano.model import IllegalTransition, Task, TaskStatus
from omega_nano.store import CorruptState, Store


def test_legal_and_illegal_transitions():
    t = Task(id="t1", goal="g")
    t.transition(TaskStatus.PLANNED)
    t.transition(TaskStatus.RUNNING)
    t.transition(TaskStatus.DONE)
    with pytest.raises(IllegalTransition):
        t.transition(TaskStatus.RUNNING)  # terminal states are final


def test_roundtrip_and_checkpoints(tmp_path):
    s = Store(tmp_path / "s.db")
    t = Task(id="abc", goal="hello")
    s.save(t, "received")
    t.transition(TaskStatus.PLANNED)
    s.save(t)
    loaded, recovered = s.load("abc")
    assert loaded.status == TaskStatus.PLANNED and not recovered
    assert [e["kind"] for e in s.events("abc")] == ["received"]


def test_corrupted_state_restores_latest_valid_checkpoint(tmp_path):
    s = Store(tmp_path / "s.db")
    t = Task(id="abc", goal="hello")
    s.save(t)
    t.transition(TaskStatus.PLANNED)
    s.save(t)
    s.conn.execute("UPDATE tasks SET state_json = replace(state_json, 'hello', 'HACKED') WHERE id='abc'")
    loaded, recovered = s.load("abc")
    assert recovered and loaded.goal == "hello" and loaded.status == TaskStatus.PLANNED
    kinds = [e["kind"] for e in s.events("abc")]
    assert "corruption_detected" in kinds and "restored_checkpoint" in kinds


def test_total_corruption_is_reported_not_hidden(tmp_path):
    s = Store(tmp_path / "s.db")
    s.save(Task(id="abc", goal="hello"))
    s.conn.execute("UPDATE tasks SET state_sha256='0'")
    s.conn.execute("UPDATE checkpoints SET state_sha256='0'")
    with pytest.raises(CorruptState):
        s.load("abc")


def test_prefix_lookup(tmp_path):
    s = Store(tmp_path / "s.db")
    s.save(Task(id="abcdef123456", goal="x"))
    assert s.load("abcdef")[0].id == "abcdef123456"
