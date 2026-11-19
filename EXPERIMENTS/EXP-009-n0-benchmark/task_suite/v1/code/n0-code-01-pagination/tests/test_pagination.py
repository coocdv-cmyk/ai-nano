import pytest

from pagination import page

ITEMS = list(range(1, 26))


def test_first_page_is_one_based():
    assert page(ITEMS, 1, 10) == list(range(1, 11))


def test_second_page():
    assert page(ITEMS, 2, 10) == list(range(11, 21))


def test_last_partial_page():
    assert page(ITEMS, 3, 10) == [21, 22, 23, 24, 25]


def test_page_zero_is_rejected():
    with pytest.raises(ValueError):
        page(ITEMS, 0, 10)


def test_page_beyond_end_is_empty():
    assert page(ITEMS, 4, 10) == []


def test_non_positive_page_size_is_rejected():
    with pytest.raises(ValueError):
        page(ITEMS, 1, 0)
