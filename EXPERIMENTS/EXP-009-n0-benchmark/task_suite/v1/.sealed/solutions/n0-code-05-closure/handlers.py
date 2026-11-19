"""Build callbacks for a set of actions."""


def make_handlers(actions):
    handlers = {}
    for name, code in actions.items():
        handlers[name] = lambda code=code: code
    return handlers


def make_handlers_with_label(actions):
    handlers = {}
    for name, code in actions.items():
        handlers[name] = lambda name=name, code=code: f"{name}:{code}"
    return handlers
