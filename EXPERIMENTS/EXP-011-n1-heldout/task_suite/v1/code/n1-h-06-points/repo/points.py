"""Distribute an integer total across buckets."""


def distribute(total, buckets):
    if buckets <= 0:
        raise ValueError("buckets must be positive")
    base = total // buckets
    return [base] * buckets
