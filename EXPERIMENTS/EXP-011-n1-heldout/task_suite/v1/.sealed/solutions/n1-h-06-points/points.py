"""Distribute an integer total across buckets."""


def distribute(total, buckets):
    if buckets <= 0:
        raise ValueError("buckets must be positive")
    if total < 0:
        raise ValueError("total must be non-negative")
    base, remainder = divmod(total, buckets)
    return [base + 1 if index < remainder else base for index in range(buckets)]
