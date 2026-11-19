import pytest

from windows import window

ITEMS = list(range(100))


def test_first_page_starts_at_zero():
    assert window(ITEMS, 1, 10) == list(range(0, 10))


def test_third_page():
    assert window(ITEMS, 3, 10) == list(range(20, 30))


def test_page_zero_is_rejected():
    with pytest.raises(ValueError):
        window(ITEMS, 0, 10)


def test_zero_size_is_rejected():
    with pytest.raises(ValueError):
        window(ITEMS, 1, 0)


def test_page_beyond_end_is_empty():
    assert window(ITEMS, 20, 10) == []
