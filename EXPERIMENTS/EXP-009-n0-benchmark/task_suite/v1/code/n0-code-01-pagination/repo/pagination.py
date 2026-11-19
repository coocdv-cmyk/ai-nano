"""Simple pagination helper."""


def page(items, page_number, page_size):
    start = page_number * page_size
    end = start + page_size
    return items[start:end]
