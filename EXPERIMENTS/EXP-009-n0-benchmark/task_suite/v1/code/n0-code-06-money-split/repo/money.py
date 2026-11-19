"""Split an integer amount of cents into n parts."""


def split_amount(total_cents, n):
    if n <= 0:
        raise ValueError("n must be positive")
    base = total_cents // n
    return [base] * n
