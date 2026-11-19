"""Retry wrapper."""

from faults import RecoverableError


def call_with_retry(fn, attempts=4):
    last = None
    for _ in range(attempts):
        try:
            return fn()
        except RecoverableError as exc:
            last = exc
    raise last
