from search import find_last_at_most


def test_last_element_can_match():
    assert find_last_at_most([1, 3, 5, 7], 7) == 3


def test_middle_element():
    assert find_last_at_most([1, 3, 5, 7], 5) == 2


def test_below_everything():
    assert find_last_at_most([2, 4, 6], 1) == -1


def test_empty():
    assert find_last_at_most([], 10) == -1


def test_between_values():
    assert find_last_at_most([2, 4, 6], 5) == 1
