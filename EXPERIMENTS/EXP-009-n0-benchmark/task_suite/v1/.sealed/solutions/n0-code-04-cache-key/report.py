from greeting import greeting

_CACHE = {}


def build_report(user_id, locale):
    key = (user_id, locale)
    if key not in _CACHE:
        _CACHE[key] = f"{greeting(locale)}, user {user_id}"
    return _CACHE[key]
