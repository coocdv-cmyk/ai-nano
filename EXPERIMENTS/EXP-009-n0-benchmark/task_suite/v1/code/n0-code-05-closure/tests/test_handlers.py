from handlers import make_handlers, make_handlers_with_label


def test_each_handler_returns_its_own_code():
    handlers = make_handlers({"save": 10, "load": 20, "quit": 30})
    assert handlers["save"]() == 10
    assert handlers["load"]() == 20
    assert handlers["quit"]() == 30


def test_label_captures_name_and_code():
    handlers = make_handlers_with_label({"a": 1, "b": 2})
    assert handlers["a"]() == "a:1"
    assert handlers["b"]() == "b:2"


def test_empty_mapping():
    assert make_handlers({}) == {}
