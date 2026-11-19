import pytest

from points import distribute


def test_total_is_preserved():
    assert sum(distribute(100, 3)) == 100
    assert sum(distribute(7, 4)) == 7


def test_remainder_goes_to_the_first_buckets():
    assert distribute(100, 3) == [34, 33, 33]
    assert distribute(10, 4) == [3, 3, 2, 2]


def test_even_split():
    assert distribute(9, 3) == [3, 3, 3]


def test_negative_total_is_rejected():
    with pytest.raises(ValueError):
        distribute(-5, 3)
