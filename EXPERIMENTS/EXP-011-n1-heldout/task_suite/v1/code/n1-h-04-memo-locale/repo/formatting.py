"""Memoised text formatting."""

_CACHE = {}


def render(name, template):
    key = name
    if key not in _CACHE:
        _CACHE[key] = template.format(name=name)
    return _CACHE[key]
