import pytest

from money import split_amount


def test_sum_is_preserved():
    assert sum(split_amount(100, 3)) == 100
    assert sum(split_amount(7, 4)) == 7


def test_remainder_goes_to_the_first_parts():
    assert split_amount(100, 3) == [34, 33, 33]
    assert split_amount(10, 4) == [3, 3, 2, 2]


def test_even_split():
    assert split_amount(9, 3) == [3, 3, 3]


def test_negative_total_is_rejected():
    with pytest.raises(ValueError):
        split_amount(-1, 3)


def test_zero_parts_is_rejected():
    with pytest.raises(ValueError):
        split_amount(10, 0)
