from audit import EventLog, add_event


def test_default_is_not_shared_between_calls():
    first = add_event("a")
    second = add_event("b")
    assert first == ["a"]
    assert second == ["b"]


def test_explicit_list_is_mutated_in_place():
    bucket = ["x"]
    result = add_event("y", bucket)
    assert result is bucket
    assert bucket == ["x", "y"]


def test_event_log_is_isolated_between_instances():
    alpha = EventLog()
    beta = EventLog()
    alpha.add(1)
    beta.add(2)
    assert alpha.snapshot() == [1]
    assert beta.snapshot() == [2]
    assert alpha.add(3) == [1, 3]


def test_default_call_after_log_use_is_still_empty():
    EventLog().add("x")
    assert add_event("solo") == ["solo"]
