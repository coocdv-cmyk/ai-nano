from buffer import push


def test_default_calls_are_isolated():
    assert push(item="a") == ["a"]
    assert push(item="b") == ["b"]


def test_explicit_list_is_mutated_in_place():
    store = ["seed"]
    result = push(store, "next")
    assert result is store
    assert store == ["seed", "next"]


def test_default_after_explicit_use_is_still_empty():
    push(["other"], "z")
    assert push(item="solo") == ["solo"]
