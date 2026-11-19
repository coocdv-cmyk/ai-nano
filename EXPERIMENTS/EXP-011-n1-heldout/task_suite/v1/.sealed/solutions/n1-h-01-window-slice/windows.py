"""Fixed-size windows over a sequence."""


def window(items, page, size):
    if page < 1:
        raise ValueError("page must be >= 1")
    if size < 1:
        raise ValueError("size must be >= 1")
    start = (page - 1) * size
    return items[start:start + size]
