"""Retry wrapper."""


def call_with_retry(fn, attempts=4):
    last = None
    for _ in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last = exc
    raise last
