"""Retry helper."""

from errors import TransientError


def with_retry(fn, attempts=3):
    last = None
    for _ in range(attempts):
        try:
            return fn()
        except TransientError as exc:
            last = exc
    raise last
