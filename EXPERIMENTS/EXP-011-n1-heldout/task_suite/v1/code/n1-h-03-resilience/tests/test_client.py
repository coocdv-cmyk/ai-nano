import pytest

from client import call_with_retry
from faults import BadRequest, RecoverableError


def test_recoverable_is_retried():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RecoverableError("busy")
        return "ok"

    assert call_with_retry(flaky) == "ok"
    assert len(calls) == 3


def test_bad_request_is_not_retried():
    calls = []

    def bad():
        calls.append(1)
        raise BadRequest("no")

    with pytest.raises(BadRequest):
        call_with_retry(bad)
    assert len(calls) == 1


def test_unexpected_error_is_not_retried():
    calls = []

    def boom():
        calls.append(1)
        raise TypeError("bug")

    with pytest.raises(TypeError):
        call_with_retry(boom)
    assert len(calls) == 1


def test_exhaustion_reraises_last():
    calls = []

    def always():
        calls.append(1)
        raise RecoverableError("still busy")

    with pytest.raises(RecoverableError):
        call_with_retry(always, attempts=3)
    assert len(calls) == 3
