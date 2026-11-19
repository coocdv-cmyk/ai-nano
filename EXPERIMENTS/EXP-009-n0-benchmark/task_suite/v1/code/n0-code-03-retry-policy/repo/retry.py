"""Retry helper."""


def with_retry(fn, attempts=3):
    last = None
    for _ in range(attempts):
        try:
            return fn()
        except Exception as exc:  # retries everything
            last = exc
    raise last
