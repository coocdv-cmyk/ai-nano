import pytest

from errors import FatalError, TransientError
from retry import with_retry


def test_transient_is_retried_until_success():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise TransientError("busy")
        return "ok"

    assert with_retry(flaky) == "ok"
    assert len(calls) == 3


def test_fatal_is_not_retried():
    calls = []

    def fatal():
        calls.append(1)
        raise FatalError("bad input")

    with pytest.raises(FatalError):
        with_retry(fatal)
    assert len(calls) == 1


def test_unexpected_exception_propagates_without_retry():
    calls = []

    def boom():
        calls.append(1)
        raise KeyError("missing")

    with pytest.raises(KeyError):
        with_retry(boom)
    assert len(calls) == 1


def test_transient_exhaustion_reraises_last():
    calls = []

    def always():
        calls.append(1)
        raise TransientError("still busy")

    with pytest.raises(TransientError):
        with_retry(always, attempts=4)
    assert len(calls) == 4
