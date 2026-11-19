"""Build callbacks for a set of actions."""


def make_handlers(actions):
    handlers = {}
    for name, code in actions.items():
        handlers[name] = lambda: code
    return handlers


def make_handlers_with_label(actions):
    handlers = {}
    for name, code in actions.items():
        handlers[name] = lambda: f"{name}:{code}"
    return handlers
