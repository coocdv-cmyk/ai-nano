"""Split an integer amount of cents into n parts."""


def split_amount(total_cents, n):
    if n <= 0:
        raise ValueError("n must be positive")
    if total_cents < 0:
        raise ValueError("total_cents must be non-negative")
    base, remainder = divmod(total_cents, n)
    return [base + 1 if index < remainder else base for index in range(n)]
