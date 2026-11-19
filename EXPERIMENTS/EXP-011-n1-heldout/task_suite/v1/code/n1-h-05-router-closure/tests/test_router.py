from router import make_routes


def test_each_route_returns_its_own_id():
    routes = make_routes({"/a": 101, "/b": 202, "/c": 303})
    assert routes["/a"]() == 101
    assert routes["/b"]() == 202
    assert routes["/c"]() == 303


def test_empty_table():
    assert make_routes({}) == {}
