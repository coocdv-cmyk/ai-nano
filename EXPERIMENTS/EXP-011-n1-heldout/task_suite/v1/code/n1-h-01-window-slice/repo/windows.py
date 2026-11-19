"""Fixed-size windows over a sequence."""


def window(items, page, size):
    start = page * size
    return items[start:start + size]
